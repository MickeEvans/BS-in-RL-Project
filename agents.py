"""
agents.py — Reinforcement learning agents for option hedging
=============================================================
Contains:
  - QHedger        : Tabular Q-learning   (thesis §2.4.4, Algorithm 2)
  - DoubleQHedger  : Double Q-learning    (thesis §2.4.4, Algorithm 3)
  - train()        : Training loop with geometric ε/α decay
  - evaluate()     : Greedy evaluation over n episodes

Both agents use:
  - ε-greedy exploration (geometric decay during training)
  - Cao c·R² variance penalty applied ONLY to the TD target
  - γ = 1 (finite horizon, no time preference)
  - Q-table initialised to zeros (no warm start)
"""

import time
import numpy as np

from environment import encode, run_episode


# ─── Q-Learning Agent (thesis Algorithm 2) ──────────────────────────────────

class QHedger:
    """
    Tabular Q-learning agent.

    Q-update (thesis Eq. 32):
        Q(S,A) ← Q(S,A) + α [ R + γ·max_a Q(S',a) − Q(S,A) ]

    The Cao c·R² regulariser is applied in the episode runner (environment.py)
    before calling update().  c=0 gives the pure thesis APL reward.
    """

    def __init__(self, params, name="QL", c=0.0):
        N_TIME  = params["N_TIME"]
        N_MONEY = params["N_MONEY"]
        N_ACT   = params["N_ACT"]
        self.name = name
        self.c    = c
        self.Q    = np.zeros((N_TIME, N_MONEY, N_ACT))

    def act(self, tau, S, params, eps):
        """ε-greedy action selection."""
        ti, mi = encode(tau, S, params)
        if np.random.random() < eps:
            return np.random.randint(params["N_ACT"])
        return int(np.argmax(self.Q[ti, mi]))

    def update(self, ti, mi, a, reward, ti_n, mi_n, alpha, gamma, done):
        """One-step Q-learning update."""
        target = reward if done else reward + gamma * self.Q[ti_n, mi_n].max()
        self.Q[ti, mi, a] += alpha * (target - self.Q[ti, mi, a])


# ─── Double Q-Learning Agent (thesis Algorithm 3) ───────────────────────────

class DoubleQHedger:
    """
    Double Q-learning agent — maintains two independent Q-tables to
    reduce maximisation bias (thesis §2.4.4, Algorithm 3).

    At each update, one table selects the best action and the other
    evaluates it, breaking the coupling that causes overestimation.
    """

    def __init__(self, params, name="DQL", c=0.0):
        N_TIME  = params["N_TIME"]
        N_MONEY = params["N_MONEY"]
        N_ACT   = params["N_ACT"]
        self.name = name
        self.c    = c
        self.Q1   = np.zeros((N_TIME, N_MONEY, N_ACT))
        self.Q2   = np.zeros((N_TIME, N_MONEY, N_ACT))

    @property
    def Q(self):
        """Combined Q for policy extraction and greedy action selection."""
        return self.Q1 + self.Q2

    def act(self, tau, S, params, eps):
        """ε-greedy using Q1 + Q2."""
        ti, mi = encode(tau, S, params)
        if np.random.random() < eps:
            return np.random.randint(params["N_ACT"])
        return int(np.argmax(self.Q1[ti, mi] + self.Q2[ti, mi]))

    def update(self, ti, mi, a, reward, ti_n, mi_n, alpha, gamma, done):
        """Double Q-learning update: randomly update Q1 or Q2."""
        if np.random.random() < 0.5:
            if done:
                target = reward
            else:
                a_star = int(np.argmax(self.Q1[ti_n, mi_n]))
                target = reward + gamma * self.Q2[ti_n, mi_n, a_star]
            self.Q1[ti, mi, a] += alpha * (target - self.Q1[ti, mi, a])
        else:
            if done:
                target = reward
            else:
                a_star = int(np.argmax(self.Q2[ti_n, mi_n]))
                target = reward + gamma * self.Q1[ti_n, mi_n, a_star]
            self.Q2[ti, mi, a] += alpha * (target - self.Q2[ti, mi, a])


# ─── Training loop ──────────────────────────────────────────────────────────

def train(agent, params, n_ep=30000, gamma=1.0, verbose=True):
    """
    Train an agent for n_ep episodes with geometric ε and α decay.

    ε decays from 1.0 → 0.05  (exploration → exploitation).
    α decays from 0.10 → 0.005 (large steps → fine tuning).
    """
    eps_start,   eps_end   = 1.00, 0.05
    alpha_start, alpha_end = 0.10, 0.005
    eps_decay   = (eps_end   / eps_start)   ** (1.0 / n_ep)
    alpha_decay = (alpha_end / alpha_start) ** (1.0 / n_ep)

    eps, alpha = eps_start, alpha_start
    log, t0    = [], time.time()

    for ep in range(n_ep):
        eps   *= eps_decay
        alpha *= alpha_decay
        pnl, tc, _ = run_episode(agent, params, eps, alpha, gamma,
                                 training=True)
        log.append((pnl, tc))

        if verbose and (ep + 1) % (n_ep // 6) == 0:
            recent = log[-(n_ep // 12):]
            print("  [%s] ep %5d  eps=%.3f  alpha=%.4f | PnL=%.4f  TC=%.4f"
                  % (agent.name, ep + 1, eps, alpha,
                     np.mean([x[0] for x in recent]),
                     np.mean([x[1] for x in recent])))

    if verbose:
        print("  Training done in %.1fs" % (time.time() - t0))
    return log


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(agent, params, n_ep=5000):
    """Run n_ep greedy episodes (no exploration, no learning)."""
    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        pnl, tc, nt = run_episode(agent, params, 0.0, 0.0, 1.0,
                                  training=False)
        pnls.append(pnl)
        tcs.append(tc)
        trades.append(nt)
    return np.array(pnls), np.array(tcs), np.array(trades)