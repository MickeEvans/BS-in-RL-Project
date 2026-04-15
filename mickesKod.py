"""
Tabular Q-Learning for European Call Option Hedging Under Transaction Costs
============================================================================

Key design choices in this version:
  - RELATIVE action space: agent picks a position *change* from
    {-1.0, -0.8, ..., 0.0, ..., +0.8, +1.0}, with 0.0 = "hold / don't trade".
    Result clipped to [0, 1]. This makes the no-trade decision explicit.
  - Reward scaling: step reward divided by K so Q-values are in percentage
    terms rather than dollar terms, improving numerical stability.
  - Log-normal GBM: S_next = S * exp((μ-σ²/2)Δt + σ√Δt Z), guaranteeing
    positive prices (consistent with thesis Eq. 2).
  - No risk-aversion penalty (c=0): the quadratic term was too aggressive
    for tabular Q-learning and made results worse. The transaction cost
    alone provides the incentive to avoid unnecessary trading.

Retained improvements:
  - Weekly time bins (52 instead of 252) → smaller Q-table
  - Randomized starting spot [80, 120] → better state coverage
  - Fast ε decay (1.0 → 0.02 over 10k episodes)
  - Decaying learning rate (α: 0.05 → 0.01)
  - Experience replay buffer

Authors: Carl Bergling, Noah El Abboudi, Michael Sederlund Evans
Project: Classical RL for Option Pricing in a Non-Perfect Market (Umeå University, 2026)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# =============================================================================
# 1. Black-Scholes Pricing
# =============================================================================

def bs_call_price_delta(S, K, tau, r, sigma):
    if tau <= 1e-12:
        payoff = max(S - K, 0.0)
        delta = 1.0 if S > K else (0.5 if abs(S - K) < 1e-12 else 0.0)
        return float(payoff), float(delta)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return float(price), float(delta)


def bs_call_price_vec(S, K, tau, r, sigma):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    d2 = d1 - sigma * np.sqrt(tau_safe)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    return np.where(tau <= 1e-12, np.maximum(S - K, 0.0), price)


def bs_delta_vec(S, K, tau, r, sigma):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    delta = norm.cdf(d1)
    return np.where(tau <= 1e-12, np.where(S > K, 1.0, 0.0), delta)


# =============================================================================
# 2. Hedging Environment — RELATIVE action space
# =============================================================================

class HedgingEnv:
    """
    The agent picks a position CHANGE (e.g. +0.2, 0.0, -0.4).
    The new position is clipped to [0, 1].
    Action 0.0 = "hold, don't trade" → zero transaction cost.

    Reward is the raw APL step reward scaled by 1/K for numerical stability.
    No quadratic risk-aversion penalty.
    
    GBM uses log-normal form to guarantee positive prices.
    """

    def __init__(self, strike, maturity, spot, vol, mu, r, dT, kappa, init_pos,
                 spot_range=None, seed=42):
        self.K = strike
        self.T = maturity
        self.S0 = spot
        self.sigma = vol
        self.mu = mu
        self.r = r
        self.dt = dT
        self.kappa = kappa
        self.init_pos = init_pos
        self.n_steps = int(round(self.T / self.dt))
        self.spot_range = spot_range

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        if self.spot_range is not None:
            lo, hi = self.spot_range
            self.S = self.rng.uniform(lo, hi)
        else:
            self.S = self.S0
        self.tau = self.T
        self.pos = self.init_pos
        self.ti = 0
        return self._obs()

    def _obs(self):
        return np.array([self.S / self.K, self.tau, self.pos], dtype=np.float32)

    def _option_price(self, S, tau):
        p, _ = bs_call_price_delta(S, self.K, tau, self.r, self.sigma)
        return p

    def _simulate_gbm(self):
        """Log-normal GBM (exact discretization of thesis Eq. 2)."""
        z = self.rng.standard_normal()
        return self.S * np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )

    def step(self, delta_pos):
        """
        delta_pos: the position CHANGE chosen by the agent.
        New position = clip(current + delta_pos, 0, 1).
        """
        pos_prev = self.pos
        S_prev = self.S
        tau_prev = self.tau
        self.ti += 1

        # Apply relative action, clip to valid range
        pos_next = float(np.clip(pos_prev + delta_pos, 0.0, 1.0))

        S_next = self._simulate_gbm()
        tau_next = max(0.0, tau_prev - self.dt)

        C_prev = self._option_price(S_prev, tau_prev)
        C_next = self._option_price(S_next, tau_next)

        # APL reward (Eq. 8), scaled by 1/K for numerical stability
        reward = (
            (S_next - S_prev) * pos_prev
            - abs(pos_next - pos_prev) * S_next * self.kappa
            - C_next
            + C_prev
        ) / self.K

        done = self.ti >= self.n_steps

        if done:
            reward -= pos_next * S_next * self.kappa / self.K

        self.S = S_next
        self.tau = tau_next
        self.pos = pos_next

        return self._obs(), reward, done, {}


# =============================================================================
# 3. Discretization
# =============================================================================

def make_discretizer(T, n_time, n_moneyness, n_pos, m_min, m_max):
    m_edges = np.linspace(m_min, m_max, n_moneyness)
    pos_grid = np.linspace(0, 1, n_pos)

    def discretize(obs):
        m, tau, pos = obs
        t_frac = 1 - tau / T
        t_idx = int(np.clip(np.floor(t_frac * n_time), 0, n_time - 1))
        m_idx = int(np.clip(np.digitize(m, m_edges) - 1, 0, len(m_edges) - 2))
        p_idx = int(np.argmin(np.abs(pos_grid - pos)))
        return (t_idx, m_idx, p_idx)

    return discretize, m_edges, pos_grid


# =============================================================================
# 4. Q-Learning with decaying α and experience replay
# =============================================================================

def q_learning(env, actions, discretize, n_time, n_money, n_pos,
               episodes, alpha_start, gamma, eps_start, eps_decay, eps_min,
               alpha_end=0.01, replay_size=50_000, replay_batch=64,
               print_every=500):
    n_actions = len(actions)
    Q = np.zeros((n_time, n_money, n_pos, n_actions))
    rewards_history = []
    replay_buf = []

    rng = np.random.default_rng(8)
    eps = eps_start

    for ep in range(episodes):
        alpha = alpha_start - (alpha_start - alpha_end) * (ep / episodes)

        obs = env.reset()
        s = discretize(obs)
        done = False
        ep_reward = 0.0

        while not done:
            if rng.random() < eps:
                a_idx = rng.integers(n_actions)
            else:
                a_idx = int(np.argmax(Q[s]))

            action = actions[a_idx]
            obs_next, reward, done, _ = env.step(action)
            s_next = discretize(obs_next)

            best_next_q = 0.0 if done else np.max(Q[s_next])
            td_error = reward + gamma * best_next_q - Q[s + (a_idx,)]
            Q[s + (a_idx,)] += alpha * td_error

            if len(replay_buf) >= replay_size:
                replay_buf[rng.integers(replay_size)] = (s, a_idx, reward, s_next, done)
            else:
                replay_buf.append((s, a_idx, reward, s_next, done))

            ep_reward += reward
            s = s_next

        # Experience replay
        if len(replay_buf) >= replay_batch:
            indices = rng.integers(len(replay_buf), size=replay_batch)
            for idx in indices:
                rs, ra, rr, rs2, rd = replay_buf[idx]
                best_q = 0.0 if rd else np.max(Q[rs2])
                Q[rs + (ra,)] += alpha * (rr + gamma * best_q - Q[rs + (ra,)])

        eps = max(eps_min, eps * eps_decay)
        rewards_history.append(ep_reward)

        if (ep + 1) % print_every == 0:
            avg_recent = np.mean(rewards_history[-print_every:])
            print(f"Episode {ep+1:>6d}/{episodes}  |  "
                  f"Avg Reward (last {print_every}): {avg_recent:>8.4f}  |  "
                  f"ε: {eps:.4f}  |  α: {alpha:.4f}")

    return Q, rewards_history


# =============================================================================
# 5. Double Q-Learning with decaying α and experience replay
# =============================================================================

def double_q_learning(env, actions, discretize, n_time, n_money, n_pos,
                      episodes, alpha_start, gamma, eps_start, eps_decay, eps_min,
                      alpha_end=0.01, replay_size=50_000, replay_batch=64,
                      print_every=500):
    n_actions = len(actions)
    Q1 = np.zeros((n_time, n_money, n_pos, n_actions))
    Q2 = np.zeros((n_time, n_money, n_pos, n_actions))
    rewards_history = []
    replay_buf = []

    rng = np.random.default_rng(8)
    eps = eps_start

    for ep in range(episodes):
        alpha = alpha_start - (alpha_start - alpha_end) * (ep / episodes)

        obs = env.reset()
        s = discretize(obs)
        done = False
        ep_reward = 0.0

        while not done:
            if rng.random() < eps:
                a_idx = rng.integers(n_actions)
            else:
                a_idx = int(np.argmax(Q1[s] + Q2[s]))

            action = actions[a_idx]
            obs_next, reward, done, _ = env.step(action)
            s_next = discretize(obs_next)

            if rng.random() < 0.5:
                if done:
                    target = reward
                else:
                    best_a = int(np.argmax(Q1[s_next]))
                    target = reward + gamma * Q2[s_next + (best_a,)]
                Q1[s + (a_idx,)] += alpha * (target - Q1[s + (a_idx,)])
            else:
                if done:
                    target = reward
                else:
                    best_a = int(np.argmax(Q2[s_next]))
                    target = reward + gamma * Q1[s_next + (best_a,)]
                Q2[s + (a_idx,)] += alpha * (target - Q2[s + (a_idx,)])

            if len(replay_buf) >= replay_size:
                replay_buf[rng.integers(replay_size)] = (s, a_idx, reward, s_next, done)
            else:
                replay_buf.append((s, a_idx, reward, s_next, done))

            ep_reward += reward
            s = s_next

        # Experience replay
        if len(replay_buf) >= replay_batch:
            indices = rng.integers(len(replay_buf), size=replay_batch)
            for idx in indices:
                rs, ra, rr, rs2, rd = replay_buf[idx]
                if rng.random() < 0.5:
                    best_q = 0.0 if rd else Q2[rs2 + (int(np.argmax(Q1[rs2])),)]
                    Q1[rs + (ra,)] += alpha * (rr + gamma * best_q - Q1[rs + (ra,)])
                else:
                    best_q = 0.0 if rd else Q1[rs2 + (int(np.argmax(Q2[rs2])),)]
                    Q2[rs + (ra,)] += alpha * (rr + gamma * best_q - Q2[rs + (ra,)])

        eps = max(eps_min, eps * eps_decay)
        rewards_history.append(ep_reward)

        if (ep + 1) % print_every == 0:
            avg_recent = np.mean(rewards_history[-print_every:])
            print(f"Episode {ep+1:>6d}/{episodes}  |  "
                  f"Avg Reward (last {print_every}): {avg_recent:>8.4f}  |  "
                  f"ε: {eps:.4f}  |  α: {alpha:.4f}")

    return Q1 + Q2, rewards_history


# =============================================================================
# 6. Evaluation
# =============================================================================

def simulate_paths(S0, mu, sigma, dT, n_steps, n_trials, seed=0):
    """Log-normal GBM paths (matches env dynamics)."""
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_steps + 1, n_trials))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_trials)
        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dT + sigma * np.sqrt(dT) * z
        )
    times = np.arange(n_steps + 1) * dT
    return paths, times


def compute_costs(policy_fn, n_trials, n_steps, S0, K, T, r, sigma,
                  init_pos, dT, mu, kappa):
    """Evaluate hedge costs (raw APL, no scaling, no penalty)."""
    paths, times = simulate_paths(S0, mu, sigma, dT, n_steps, n_trials)

    costs = np.zeros(n_trials)
    pos_prev = np.ones(n_trials) * init_pos
    pos_next = policy_fn(paths[0] / K, T * np.ones(n_trials), pos_prev)

    for t in range(1, n_steps + 1):
        S_now = paths[t]
        S_prev = paths[t - 1]
        tau_now = np.maximum(0.0, T - times[t])
        tau_prev = np.maximum(0.0, T - times[t - 1])

        C_now = bs_call_price_vec(S_now, K, tau_now, r, sigma)
        C_prev = bs_call_price_vec(S_prev, K, tau_prev, r, sigma)

        step_pnl = (
            (S_now - S_prev) * pos_prev
            - np.abs(pos_next - pos_prev) * S_now * kappa
            - C_now
            + C_prev
        )

        if t == n_steps:
            step_pnl -= pos_next * S_now * kappa

        costs += step_pnl

        if t < n_steps:
            pos_prev = pos_next
            pos_next = policy_fn(
                paths[t] / K,
                np.maximum(0.0, T - times[t]) * np.ones(n_trials),
                pos_prev,
            )

    return costs


# =============================================================================
# 7. Policies
# =============================================================================

def make_policy_bsm(K, r, sigma):
    def policy(moneyness, ttm, pos):
        S = moneyness * K
        return bs_delta_vec(S, K, ttm, r, sigma)
    return policy


def make_policy_rl(Q, actions, discretize):
    """
    Greedy RL policy for RELATIVE action space.
    The Q-table stores values for position changes.
    The policy returns absolute target positions.
    """
    def policy(moneyness, ttm, pos):
        out = np.zeros_like(moneyness)
        for i in range(len(moneyness)):
            obs = np.array([moneyness[i], ttm[i], pos[i]])
            s = discretize(obs)
            a_idx = int(np.argmax(Q[s]))
            delta_pos = actions[a_idx]
            out[i] = np.clip(pos[i] + delta_pos, 0.0, 1.0)
        return out
    return policy


# =============================================================================
# 8. Plotting
# =============================================================================

def plot_learning_curve(rewards, title="Q-Learning: Episode Rewards", window=200):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    if len(rewards) >= window:
        rolling = pd.Series(rewards).rolling(window=window).mean()
        ax.plot(rolling, color="darkblue", linewidth=2,
                label=f"Moving Average ({window} episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cost_histogram(costs_bsm, costs_rl, label_rl="Q-Learning"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(-costs_bsm, bins=40, alpha=0.5, color="blue", label="BS Delta Hedge")
    ax.hist(-costs_rl, bins=40, alpha=0.5, color="red", label=f"{label_rl} Hedge")
    ax.set_xlabel("Hedge Cost")
    ax.set_ylabel("Number of Trials")
    ax.set_title("Hedge Cost Distribution: RL vs. Black-Scholes")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hedge_ratio(Q, actions, discretize, K, r, sigma,
                     ttm_plot=2/12, m_range=None):
    """
    For relative actions: given a starting position, the policy returns
    pos + actions[argmax Q[s]], clipped to [0,1].
    """
    if m_range is None:
        m_range = np.arange(0.80, 1.2001, 0.01)

    bsm_line = []
    rl_atm, rl_selling, rl_buying = [], [], []

    for m in m_range:
        _, d_atm = bs_call_price_delta(m, 1.0, ttm_plot, r, sigma)
        _, d_sell = bs_call_price_delta(m + 0.1, 1.0, ttm_plot, r, sigma)
        _, d_buy = bs_call_price_delta(m - 0.1, 1.0, ttm_plot, r, sigma)

        bsm_line.append(d_atm)

        for pos_start, store in [(d_atm, rl_atm), (d_sell, rl_selling), (d_buy, rl_buying)]:
            obs = np.array([m, ttm_plot, pos_start])
            s = discretize(obs)
            a_idx = int(np.argmax(Q[s]))
            new_pos = np.clip(pos_start + actions[a_idx], 0.0, 1.0)
            store.append(new_pos)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m_range, bsm_line, "k--", linewidth=2, label="BS Delta")
    ax.plot(m_range, rl_atm, label="RL Hedge – ATM position", linewidth=1.5)
    ax.plot(m_range, rl_selling, label="RL Hedge – Selling position", linewidth=1.5)
    ax.plot(m_range, rl_buying, label="RL Hedge – Buying position", linewidth=1.5)
    ax.set_xlabel("Moneyness (S / K)")
    ax.set_ylabel("Hedge Ratio")
    ax.set_title(f"Learned Hedge Ratio vs. BS Delta  (TTM = {ttm_plot:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# 9. Main Experiment
# =============================================================================

def main():
    # ----- Market & Option Parameters ---------------------------------------
    Strike = 100
    Maturity = 1
    SpotPrice = 100
    ExpVol = 0.2
    ExpReturn = 0.05
    rfRate = 0
    dT = 1 / 252
    kappa = 0.01
    InitPosition = 0

    nSteps = int(round(Maturity / dT))

    # ----- Discretization ---------------------------------------------------
    n_time_bins = 52         # Weekly time bins
    n_m_bins = 41            # 40 moneyness bins over [0.8, 1.2]
    n_pos_bins = 11          # 11 position levels: 0%, 10%, ..., 100%
    m_min, m_max = 0.8, 1.2

    # ----- RELATIVE action space --------------------------------------------
    # 11 actions: position changes from -1.0 to +1.0 in steps of 0.2
    # Action 0.0 = hold (no trade, no cost)
    # Full range allows day-1 hedge establishment in one step
    # Result clipped to [0, 1] by the environment
    actions = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_actions = len(actions)

    # ----- Hyperparameters --------------------------------------------------
    episodes = 10_000
    alpha_start = 0.05
    alpha_end = 0.01
    gamma = 0.9995
    eps_start = 1.0
    eps_decay = 0.9997
    eps_min = 0.02
    replay_size = 50_000
    replay_batch = 64

    # ----- Environment ------------------------------------------------------
    env = HedgingEnv(
        strike=Strike, maturity=Maturity, spot=SpotPrice,
        vol=ExpVol, mu=ExpReturn, r=rfRate, dT=dT,
        kappa=kappa, init_pos=InitPosition,
        spot_range=(80, 120),
        seed=42,
    )

    discretize, m_edges, pos_grid = make_discretizer(
        Maturity, n_time_bins, n_m_bins, n_pos_bins, m_min, m_max
    )
    n_money_bins = n_m_bins - 1

    print("=" * 70)
    print("  Q-Learning for European Call Hedging Under Transaction Costs")
    print("=" * 70)
    print(f"  Strike={Strike}, Spot={SpotPrice}, T={Maturity}, σ={ExpVol}, "
          f"μ={ExpReturn}, r={rfRate}")
    print(f"  κ={kappa}, Δt=1/{int(1/dT)}, N={nSteps}")
    print(f"  Action space: RELATIVE {actions.tolist()}")
    print(f"  State grid: {n_time_bins} time × {n_money_bins} moneyness × "
          f"{n_pos_bins} position × {n_actions} actions")
    print(f"  Q-table size: {n_time_bins * n_money_bins * n_pos_bins * n_actions:,} entries")
    print(f"  Episodes: {episodes}, α: {alpha_start}→{alpha_end}, γ={gamma}")
    print(f"  ε: {eps_start} → {eps_min} (decay={eps_decay})")
    print(f"  Replay buffer: {replay_size:,} / batch={replay_batch}")
    print(f"  Reward scaling: 1/K = 1/{Strike}")
    print(f"  GBM: log-normal (exact)")
    print("=" * 70)

    # ===== TRAIN =============================================================
    print("\n>>> Training Q-Learning ...")
    Q, rewards = q_learning(
        env, actions, discretize,
        n_time_bins, n_money_bins, n_pos_bins,
        episodes=episodes, alpha_start=alpha_start, gamma=gamma,
        eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min,
        alpha_end=alpha_end, replay_size=replay_size, replay_batch=replay_batch,
        print_every=1000,
    )

    print("\n>>> Training Double Q-Learning ...")
    Q_double, rewards_double = double_q_learning(
        env, actions, discretize,
        n_time_bins, n_money_bins, n_pos_bins,
        episodes=episodes, alpha_start=alpha_start, gamma=gamma,
        eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min,
        alpha_end=alpha_end, replay_size=replay_size, replay_batch=replay_batch,
        print_every=1000,
    )

    # ===== EVALUATE ==========================================================
    print("\n>>> Evaluating hedge costs on 1,000 Monte Carlo paths ...")
    n_trials = 1_000

    policy_bsm = make_policy_bsm(Strike, rfRate, ExpVol)
    policy_rl = make_policy_rl(Q, actions, discretize)
    policy_dql = make_policy_rl(Q_double, actions, discretize)

    costs_bsm = compute_costs(
        policy_bsm, n_trials, nSteps, SpotPrice, Strike,
        Maturity, rfRate, ExpVol, InitPosition, dT, ExpReturn, kappa,
    )
    costs_ql = compute_costs(
        policy_rl, n_trials, nSteps, SpotPrice, Strike,
        Maturity, rfRate, ExpVol, InitPosition, dT, ExpReturn, kappa,
    )
    costs_dql = compute_costs(
        policy_dql, n_trials, nSteps, SpotPrice, Strike,
        Maturity, rfRate, ExpVol, InitPosition, dT, ExpReturn, kappa,
    )

    option_price, _ = bs_call_price_delta(SpotPrice, Strike, Maturity, rfRate, ExpVol)

    table = pd.DataFrame(
        {
            "BS Delta": 100 * np.array([-np.mean(costs_bsm), np.std(costs_bsm)]) / option_price,
            "Q-Learning": 100 * np.array([-np.mean(costs_ql), np.std(costs_ql)]) / option_price,
            "Double Q": 100 * np.array([-np.mean(costs_dql), np.std(costs_dql)]) / option_price,
        },
        index=[
            "Avg Hedge Cost (% of Option Price)",
            "Std Hedge Cost (% of Option Price)",
        ],
    )

    print("\n" + "=" * 70)
    print("  Hedge Cost Comparison (as % of BS Option Price)")
    print("=" * 70)
    print(table.to_string())
    print("=" * 70)

    # Coverage diagnostics
    total = Q.size
    visited = np.count_nonzero(Q)
    print(f"\n  Q-table coverage: {visited:,} / {total:,} ({100*visited/total:.1f}%)")
    total_d = Q_double.size
    visited_d = np.count_nonzero(Q_double)
    print(f"  Double-Q coverage: {visited_d:,} / {total_d:,} ({100*visited_d/total_d:.1f}%)")

    # ===== PLOTS =============================================================
    fig1 = plot_learning_curve(rewards, title="Q-Learning: Episode Rewards")
    fig1.savefig("q_learning_curve.png", dpi=150)

    fig1b = plot_learning_curve(rewards_double, title="Double Q-Learning: Episode Rewards")
    fig1b.savefig("double_q_learning_curve.png", dpi=150)

    fig2 = plot_cost_histogram(costs_bsm, costs_ql, label_rl="Q-Learning")
    fig2.savefig("q_learning_cost_histogram.png", dpi=150)

    fig2b = plot_cost_histogram(costs_bsm, costs_dql, label_rl="Double Q-Learning")
    fig2b.savefig("double_q_cost_histogram.png", dpi=150)

    fig3 = plot_hedge_ratio(Q, actions, discretize, Strike, rfRate, ExpVol)
    fig3.savefig("q_learning_hedge_ratio.png", dpi=150)

    fig3b = plot_hedge_ratio(Q_double, actions, discretize, Strike, rfRate, ExpVol)
    fig3b.savefig("double_q_hedge_ratio.png", dpi=150)

    plt.show()

    print("\nAll plots saved. Done!")
    return Q, Q_double, table


if __name__ == "__main__":
    Q, Q_double, table = main()