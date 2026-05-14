"""
Pure tabular Q-learning for the hedging problem. Two state-space variants
both come up in the thesis:

  state_dim = 2:  s = (tau, moneyness)   -- minimal-state version
  state_dim = 3:  s = (tau, moneyness, H_prev)

The Q-learning update rule is exactly Sutton & Barto / thesis Algorithm 2:
    Q(s,a) <- Q(s,a) + alpha [r + gamma * max_a' Q(s',a') - Q(s,a)]

Optionally Double Q-learning (thesis Algorithm 3): two tables Q1, Q2;
randomly update one using the other's evaluation of the argmax action.

Three improvements over the baseline:
  (1) Robbins-Monro alpha decay:  alpha_t = alpha_0 / (1 + episode/tau_alpha)
      Standard stochastic-approximation schedule; ensures convergence.
  (2) Non-uniform moneyness binning: log-spaced bins concentrated near m=1.
      Most of the hedging action happens between m=0.85 and m=1.15; uniform
      bins waste resolution far away.
  (3) Bellman-residual logging: track mean |TD error| per episode as a true
      convergence diagnostic, independent of epsilon-greedy noise in the
      running cost average.
"""
import numpy as np


class QAgent:
    def __init__(self, N, dt,
                 state_dim=3,
                 tau_bins=26, m_bins=21, h_bins=11, A=11,
                 m_min=0.5, m_max=1.7,
                 m_binning="uniform",          # 'uniform' or 'log_centered'
                 alpha=0.1, gamma=1.0,
                 alpha_decay_episodes=None,    # None -> constant; int -> R-M decay
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=15000,
                 risk_c=0.0, optimistic_init=0.0,
                 double_q=False, seed=0):
        assert state_dim in (2, 3)
        assert m_binning in ("uniform", "log_centered")
        self.state_dim = state_dim
        self.N = N
        self.dt = dt
        self.tau_bins = tau_bins
        self.m_bins = m_bins
        self.h_bins = h_bins
        self.A = A
        self.m_min = m_min
        self.m_max = m_max
        self.m_binning = m_binning
        self.alpha_0 = alpha
        self.alpha_decay_episodes = alpha_decay_episodes
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.risk_c = risk_c
        self.episode = 0
        self.double_q = double_q

        # Action grid
        self.actions = np.linspace(0.0, 1.0, A)
        self.h_grid = np.linspace(0.0, 1.0, h_bins) if state_dim == 3 else None

        # Moneyness bin EDGES (length m_bins+1).
        # 'uniform' -> equally spaced.
        # 'log_centered' -> equally spaced in log space, naturally concentrating
        # bins near m=1 since log(m) is small there.
        if m_binning == "uniform":
            self.m_edges = np.linspace(m_min, m_max, m_bins + 1)
        else:
            log_min = np.log(m_min)
            log_max = np.log(m_max)
            self.m_edges = np.exp(np.linspace(log_min, log_max, m_bins + 1))

        if state_dim == 2:
            shape = (tau_bins, m_bins, A)
        else:
            shape = (tau_bins, m_bins, h_bins, A)

        self.Q = np.full(shape, optimistic_init, dtype=np.float64)
        if double_q:
            self.Q2 = np.full(shape, optimistic_init, dtype=np.float64)

        # Bellman-residual tracking (mean absolute TD error per episode)
        self._td_sum = 0.0
        self._td_count = 0
        self.td_history = []

        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    @property
    def epsilon(self):
        frac = min(1.0, self.episode / self.eps_decay_episodes)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @property
    def alpha(self):
        """Robbins-Monro decay if alpha_decay_episodes is set, else constant."""
        if self.alpha_decay_episodes is None:
            return self.alpha_0
        return self.alpha_0 / (1.0 + self.episode / self.alpha_decay_episodes)

    # -------------------------------------------------------------------------
    def _idx(self, tau, m, h):
        T = self.N * self.dt
        tc = min(max(tau, 0.0), T)
        ti = int(tc / T * (self.tau_bins - 1) + 0.5)
        ti = max(0, min(self.tau_bins - 1, ti))

        # searchsorted handles any monotone edges, uniform or log-spaced.
        mc = min(max(m, self.m_min), self.m_max - 1e-12)
        mi = int(np.searchsorted(self.m_edges, mc, side="right") - 1)
        mi = max(0, min(self.m_bins - 1, mi))

        if self.state_dim == 2:
            return (ti, mi)
        hc = min(max(h, 0.0), 1.0)
        hi = int(hc * (self.h_bins - 1) + 0.5)
        hi = max(0, min(self.h_bins - 1, hi))
        return (ti, mi, hi)

    def select_action(self, tau, m, h, greedy=False):
        idx = self._idx(tau, m, h)
        if (not greedy) and self.rng.random() < self.epsilon:
            a_idx = int(self.rng.integers(0, self.A))
        else:
            if self.double_q:
                q_sum = self.Q[idx] + self.Q2[idx]
                a_idx = int(np.argmax(q_sum))
            else:
                a_idx = int(np.argmax(self.Q[idx]))
        return a_idx, self.actions[a_idx]

    def greedy_action(self, tau, m, h):
        idx = self._idx(tau, m, h)
        if self.double_q:
            q_sum = self.Q[idx] + self.Q2[idx]
            return float(self.actions[int(np.argmax(q_sum))])
        return float(self.actions[int(np.argmax(self.Q[idx]))])

    # -------------------------------------------------------------------------
    def update(self, tau, m, h, a_idx, reward, tau_next, m_next, h_next, done):
        idx  = self._idx(tau, m, h)
        idx2 = self._idx(tau_next, m_next, h_next)
        r = reward - self.risk_c * reward * reward
        a = self.alpha

        if self.double_q:
            if self.rng.random() < 0.5:
                if done:
                    target = r
                else:
                    a_star = int(np.argmax(self.Q[idx2]))
                    target = r + self.gamma * self.Q2[idx2][a_star]
                q_idx = idx + (a_idx,)
                td = target - self.Q[q_idx]
                self.Q[q_idx] += a * td
            else:
                if done:
                    target = r
                else:
                    a_star = int(np.argmax(self.Q2[idx2]))
                    target = r + self.gamma * self.Q[idx2][a_star]
                q_idx = idx + (a_idx,)
                td = target - self.Q2[q_idx]
                self.Q2[q_idx] += a * td
        else:
            if done:
                target = r
            else:
                target = r + self.gamma * np.max(self.Q[idx2])
            q_idx = idx + (a_idx,)
            td = target - self.Q[q_idx]
            self.Q[q_idx] += a * td

        self._td_sum += abs(td)
        self._td_count += 1

    def end_episode(self):
        if self._td_count > 0:
            self.td_history.append(self._td_sum / self._td_count)
        else:
            self.td_history.append(0.0)
        self._td_sum = 0.0
        self._td_count = 0
        self.episode += 1
