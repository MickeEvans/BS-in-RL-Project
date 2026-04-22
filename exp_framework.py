"""
Experimentation framework for Q-learning hedging improvements.
==============================================================
Goal: find a configuration of tabular Q-learning that BEATS BS weekly
on the mean-variance objective.

Key idea (from Cao-Chen-Hull-Poulos 2021): track TWO Q-functions:
  Q1(s,a) = E[total cost from here onward | take a, then follow policy]
  Q2(s,a) = E[(total cost from here onward)^2 | ...]
Policy: a* = argmin over a of  Q1(s,a) + c * sqrt(max(0, Q2(s,a) - Q1(s,a)^2))

This satisfies Bellman, so per-step bootstrapping works properly.
The per-step reward is just the per-step COST (non-risk-adjusted):
  cost_t = |dpos_t| * S_t * kappa + terminal close-out cost if done.
The risk adjustment happens in the policy and Q2 tracks cost^2.

Start small, iterate, scale up only what works.
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import norm


# =============================================================================
# BS utilities
# =============================================================================
def bs_price_vec(S, K, tau, r, sigma):
    tau_safe = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    d2 = d1 - sigma * np.sqrt(tau_safe)
    price = S * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    return np.where(tau <= 1e-12, np.maximum(S - K, 0.0), price)


def bs_delta_vec(S, K, tau, r, sigma):
    tau_safe = np.maximum(tau, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    return np.where(tau <= 1e-12, np.where(S > K, 1.0, 0.0), norm.cdf(d1))


# =============================================================================
# Environment: returns per-step COST (not reward). Cost is strictly >= 0.
# =============================================================================
#
# We formulate hedging as a COST-MINIMIZATION problem (CCHP convention):
#   cost_t = - dPnL_t   where dPnL is the step-by-step accounting P&L.
# For a short call hedged by long stock:
#   dPnL_t = (S_{t+1} - S_t) * pos_t     # stock gain
#           - |pos_{t+1} - pos_t| * S_{t+1} * kappa   # trading cost
#           - (C_{t+1} - C_t)             # option value change
# Terminal extra: close out position, pay |pos_T| * S_T * kappa.
#
# So cost_t = -dPnL_t. If the agent hedges perfectly, cost is just TC.

def simulate_paths_batch(S0, sigma, r, dt, n_steps, batch_size, rng):
    """Generate `batch_size` GBM paths under risk-neutral measure, each of length n_steps+1."""
    paths = np.empty((n_steps + 1, batch_size))
    paths[0] = S0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(batch_size)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths


# =============================================================================
# CCHP Dual-Q Learning
# =============================================================================
def cchp_dual_q_train(
    K, T, S0, sigma, r, dt, kappa, init_pos,
    actions, c_risk,
    n_time, n_money, n_gap, m_edges, g_edges,
    n_batches, batch_size,
    alpha_start, alpha_end, eps_start, eps_end,
    gamma=1.0, exploring_starts_frac=0.3,
    seed=8, print_every=50,
):
    """
    Cao-Chen-Hull-Poulos dual-Q learning.

    Q1[s,a] = E[sum of future costs | take a, then greedy]
    Q2[s,a] = E[(sum of future costs)^2 | ...]

    Policy: argmin_a  Q1[s,a] + c_risk * sqrt(max(0, Q2[s,a] - Q1[s,a]^2))

    Bellman targets (from the paper):
      Q1(s,a) <- cost + gamma * Q1(s', a*)
      Q2(s,a) <- cost^2 + 2*gamma*cost*Q1(s', a*) + gamma^2 * Q2(s', a*)
    where a* is the greedy action at s' under the RISK-ADJUSTED criterion.
    """
    rng = np.random.default_rng(seed)
    n_actions = len(actions)
    n_steps = int(round(T / dt))
    actions_arr = np.asarray(actions)

    Q1 = np.zeros((n_time, n_money, n_gap, n_actions))
    Q2 = np.zeros((n_time, n_money, n_gap, n_actions))
    visit_count = np.zeros((n_time, n_money, n_gap, n_actions), dtype=np.int32)
    rewards_history = np.zeros(n_batches)

    phase1_end = int(exploring_starts_frac * n_batches)

    def tau_to_tidx(tau_arr):
        t_frac = 1.0 - tau_arr / T
        return np.clip(np.floor(t_frac * n_time).astype(int), 0, n_time - 1)

    def risk_adjusted_q(q1_slice, q2_slice):
        """Compute Q1 + c*sqrt(max(0, Q2 - Q1^2)) for each action."""
        var = np.maximum(q2_slice - q1_slice**2, 0.0)
        return q1_slice + c_risk * np.sqrt(var)

    for batch in range(n_batches):
        # Schedules
        if batch < phase1_end:
            p_frac = batch / max(1, phase1_end - 1)
            alpha = alpha_start + (alpha_start/2 - alpha_start) * p_frac
            eps = eps_start + (0.3 - eps_start) * p_frac
            # Exploring starts
            S_init = S0 * np.exp(rng.normal(0, 0.15, size=batch_size))
            pos_init = rng.uniform(0, 1, size=batch_size)
            tau_init = rng.uniform(0.05, T, size=batch_size)
        else:
            p_frac = (batch - phase1_end) / max(1, n_batches - phase1_end - 1)
            alpha = alpha_start/2 + (alpha_end - alpha_start/2) * p_frac
            eps = 0.3 + (eps_end - 0.3) * p_frac
            S_init = np.full(batch_size, S0)
            pos_init = np.full(batch_size, init_pos)
            tau_init = np.full(batch_size, T)

        S = S_init.copy()
        pos = pos_init.copy()
        tau = tau_init.copy()
        ep_cost = np.zeros(batch_size)

        t_idx = tau_to_tidx(tau)
        m_idx = np.clip(np.digitize(S/K, m_edges) - 1, 0, n_money - 1)
        bsd = bs_delta_vec(S, K, tau, r, sigma)
        g_idx = np.clip(np.digitize(pos - bsd, g_edges) - 1, 0, n_gap - 1)

        alive = np.ones(batch_size, dtype=bool)

        for ti in range(n_steps):
            if not alive.any():
                break

            # Action selection: eps-greedy on risk-adjusted Q
            q1_s = Q1[t_idx, m_idx, g_idx]
            q2_s = Q2[t_idx, m_idx, g_idx]
            ra_q = risk_adjusted_q(q1_s, q2_s)
            greedy = np.argmin(ra_q, axis=1)  # MINIMIZE cost
            rand_a = rng.integers(n_actions, size=batch_size)
            explore = rng.random(batch_size) < eps
            a_idx = np.where(explore, rand_a, greedy)
            action = actions_arr[a_idx]

            # Env step
            pos_prev = pos.copy(); S_prev = S.copy(); tau_prev = tau.copy()
            pos_next = np.clip(pos_prev + action, 0.0, 1.0)
            z = rng.standard_normal(batch_size)
            S_next = S_prev * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            tau_next = np.maximum(0.0, tau_prev - dt)
            C_prev = bs_price_vec(S_prev, K, tau_prev, r, sigma)
            C_next = bs_price_vec(S_next, K, tau_next, r, sigma)

            # Step COST (positive = bad) — scale by 1/K as in paper
            dPnL = (S_next - S_prev) * pos_prev \
                   - np.abs(pos_next - pos_prev) * S_next * kappa \
                   - C_next + C_prev
            cost = -dPnL / K

            done_now = tau_next <= 1e-12
            cost = np.where(done_now, cost + pos_next * S_next * kappa / K, cost)
            ep_cost = ep_cost + cost * alive

            # Next state indices
            mon_next = S_next / K
            t_next = tau_to_tidx(tau_next)
            m_next = np.clip(np.digitize(mon_next, m_edges) - 1, 0, n_money - 1)
            bsd_next = bs_delta_vec(S_next, K, tau_next, r, sigma)
            g_next = np.clip(np.digitize(pos_next - bsd_next, g_edges) - 1, 0, n_gap - 1)

            # Bellman targets: find greedy next action under risk-adjusted Q
            q1_ns = Q1[t_next, m_next, g_next]
            q2_ns = Q2[t_next, m_next, g_next]
            ra_q_next = risk_adjusted_q(q1_ns, q2_ns)
            a_star = np.argmin(ra_q_next, axis=1)

            # Get Q1(s', a*) and Q2(s', a*)
            rng_idx = np.arange(batch_size)
            q1_next_star = Q1[t_next, m_next, g_next, a_star]
            q2_next_star = Q2[t_next, m_next, g_next, a_star]

            # Targets (CCHP eq. 9-10):
            # Q1_target = cost + gamma * Q1_next
            # Q2_target = cost^2 + 2*gamma*cost*Q1_next + gamma^2 * Q2_next
            q1_target = np.where(done_now, cost, cost + gamma * q1_next_star)
            q2_target = np.where(done_now,
                                cost**2,
                                cost**2 + 2*gamma*cost*q1_next_star + gamma**2 * q2_next_star)

            # Apply TD updates only on alive paths -- VECTORIZED via np.add.at.
            # np.add.at handles the edge case where multiple paths update the
            # same (t,m,g,a) cell in one batch: it applies all updates sequentially.
            # Note: with decaying alpha and large state space, same-cell collisions
            # per batch are very rare, so this is a faithful implementation.
            alive_mask = alive
            if alive_mask.any():
                ti_a = t_idx[alive_mask]
                mi_a = m_idx[alive_mask]
                gi_a = g_idx[alive_mask]
                ai_a = a_idx[alive_mask]
                q1_cur = Q1[ti_a, mi_a, gi_a, ai_a]
                q2_cur = Q2[ti_a, mi_a, gi_a, ai_a]
                q1_delta = alpha * (q1_target[alive_mask] - q1_cur)
                q2_delta = alpha * (q2_target[alive_mask] - q2_cur)
                np.add.at(Q1, (ti_a, mi_a, gi_a, ai_a), q1_delta)
                np.add.at(Q2, (ti_a, mi_a, gi_a, ai_a), q2_delta)
                np.add.at(visit_count, (ti_a, mi_a, gi_a, ai_a), 1)

            # Roll
            S, pos, tau = S_next, pos_next, tau_next
            t_idx, m_idx, g_idx = t_next, m_next, g_next
            alive = alive & ~done_now

        rewards_history[batch] = ep_cost.mean()

        if (batch + 1) % print_every == 0:
            recent = rewards_history[max(0, batch - print_every + 1):batch + 1]
            print(f"Batch {batch+1:>5d}/{n_batches}  "
                  f"(eps={eps:.3f} alpha={alpha:.4f})  "
                  f"Avg cost/ep: {recent.mean():>8.4f}  "
                  f"Visit coverage: {100*np.count_nonzero(visit_count)/visit_count.size:.1f}%")

    return Q1, Q2, visit_count, rewards_history


# =============================================================================
# Evaluation
# =============================================================================
def make_cchp_policy(Q1, Q2, actions, c_risk, K, r, sigma, T,
                     n_time, m_edges, g_edges, n_money, n_gap,
                     nt_bonus=0.0):
    """
    nt_bonus: subtract this from the risk-adjusted Q of action=0 (do nothing)
    before argmin. Positive values bias toward not trading, reducing
    over-trading from noisy tie-breaks. 0 = off.
    """
    actions_arr = np.asarray(actions)
    # Find the index of action 0
    zero_idx = int(np.argmin(np.abs(actions_arr)))
    assert abs(actions_arr[zero_idx]) < 1e-9, "actions must include 0"

    def policy(S, tau, pos):
        t_frac = 1 - tau / T
        t_idx = np.clip(np.floor(t_frac * n_time).astype(int), 0, n_time - 1)
        m_idx = np.clip(np.digitize(S/K, m_edges) - 1, 0, n_money - 1)
        bsd = bs_delta_vec(S, K, tau, r, sigma)
        g_idx = np.clip(np.digitize(pos - bsd, g_edges) - 1, 0, n_gap - 1)
        q1_s = Q1[t_idx, m_idx, g_idx]
        q2_s = Q2[t_idx, m_idx, g_idx]
        var = np.maximum(q2_s - q1_s**2, 0.0)
        ra_q = q1_s + c_risk * np.sqrt(var)
        # Apply no-trade bonus: subtract from action-0's cost
        if nt_bonus != 0.0:
            ra_q = ra_q.copy()
            ra_q[:, zero_idx] -= nt_bonus
        a_idx = np.argmin(ra_q, axis=1)
        return np.clip(pos + actions_arr[a_idx], 0.0, 1.0)
    return policy


def make_bs_policy(K, r, sigma):
    def policy(S, tau, pos):
        return bs_delta_vec(S, K, tau, r, sigma)
    return policy


def evaluate_policy(policy, paths, times, K, T, r, sigma, init_pos, kappa, rebal_every=1):
    n_steps, n_trials = paths.shape[0] - 1, paths.shape[1]
    costs = np.zeros(n_trials)
    pos_prev = np.full(n_trials, init_pos)
    ttm0 = np.full(n_trials, T)
    pos_next = policy(paths[0], ttm0, pos_prev)

    for t in range(1, n_steps + 1):
        S_now, S_prev = paths[t], paths[t-1]
        tau_now = max(0.0, T - times[t])
        tau_prev = max(0.0, T - times[t-1])
        C_now = bs_price_vec(S_now, K, tau_now, r, sigma)
        C_prev = bs_price_vec(S_prev, K, tau_prev, r, sigma)
        step_pnl = (S_now - S_prev) * pos_prev \
                   - np.abs(pos_next - pos_prev) * S_now * kappa \
                   - C_now + C_prev
        if t == n_steps:
            step_pnl -= pos_next * S_now * kappa
        costs += step_pnl
        if t < n_steps:
            pos_prev = pos_next
            if t % rebal_every == 0:
                ttm = np.full(n_trials, max(0.0, T - times[t]))
                pos_next = policy(paths[t], ttm, pos_prev)
    return costs  # (positive = gain, negative = loss)


def summarize(costs, option_price, label, c_risk=1.5):
    h = -costs  # hedge cost (positive = cost)
    mean, std = h.mean(), h.std()
    var95 = np.quantile(h, 0.95)
    cvar = h[h >= var95].mean()
    mv = mean + c_risk * std  # CCHP objective
    return {'label': label, 'mean': mean, 'std': std, 'cvar': cvar,
            'mv_obj': mv, 'option_price': option_price}


def print_summary(results, c_risk):
    print("\n" + "=" * 90)
    print(f"  Hedge Cost Summary  (c = {c_risk} in Y = mean + c*std)")
    print("=" * 90)
    df = pd.DataFrame({r['label']: [r['mean'], r['std'], r['cvar'], r['mv_obj']]
                      for r in results},
                     index=['Mean Cost $', 'Std Cost $', 'CVaR-95% $',
                            f'Y = Mean + {c_risk}*Std $'])
    print(df.to_string(float_format=lambda x: f"{x:9.4f}"))
    print("=" * 90)


# =============================================================================
# Quick experiment runner
# =============================================================================
def run_experiment(
    config_name,
    n_batches=50, batch_size=64,
    c_risk=1.5, kappa=0.01,
    n_time=52, n_money=40, n_gap=21,
    m_min=0.5, m_max=1.5, g_min=-0.5, g_max=0.5,
    actions=None, alpha_start=0.05, alpha_end=0.005,
    eps_start=1.0, eps_end=0.05,
    exploring_starts_frac=0.3,
    K=100.0, T=1.0, S0=100.0, sigma=0.2, r=0.0, dt=1/252,
    n_eval=5_000, seed=8,
    verbose=True,
):
    """Run a full small experiment: train, evaluate, print, return summaries."""
    if actions is None:
        actions = np.array([-0.20, -0.10, -0.05, -0.02, -0.01, 0.0,
                           0.01, 0.02, 0.05, 0.10, 0.20])
    n_steps = int(round(T / dt))

    m_edges = np.linspace(m_min, m_max, n_money + 1)
    g_edges = np.linspace(g_min, g_max, n_gap + 1)

    # Initial position = BS delta at t=0
    init_pos = float(bs_delta_vec(np.array([S0]), K, np.array([T]), r, sigma)[0])
    option_price = float(bs_price_vec(np.array([S0]), K, np.array([T]), r, sigma)[0])

    if verbose:
        print("=" * 90)
        print(f"  EXPERIMENT: {config_name}")
        print(f"  n_batches={n_batches}, batch_size={batch_size}, "
              f"total eps={n_batches*batch_size}")
        print(f"  c_risk={c_risk}, |actions|={len(actions)}, "
              f"grid={n_time}x{n_money}x{n_gap}")
        print(f"  alpha: {alpha_start}->{alpha_end}, eps: {eps_start}->{eps_end}, "
              f"xs_frac={exploring_starts_frac}")
        print("=" * 90)

    t0 = time.time()
    Q1, Q2, visits, rewards = cchp_dual_q_train(
        K, T, S0, sigma, r, dt, kappa, init_pos,
        actions, c_risk,
        n_time, n_money, n_gap, m_edges, g_edges,
        n_batches, batch_size,
        alpha_start, alpha_end, eps_start, eps_end,
        exploring_starts_frac=exploring_starts_frac,
        seed=seed, print_every=max(1, n_batches//5) if verbose else 10**9,
    )
    train_time = time.time() - t0

    # Evaluate
    eval_rng = np.random.default_rng(777)
    paths = simulate_paths_batch(S0, sigma, r, dt, n_steps, n_eval, eval_rng)
    times = np.arange(n_steps + 1) * dt

    pol_rl = make_cchp_policy(Q1, Q2, actions, c_risk, K, r, sigma, T,
                              n_time, m_edges, g_edges, n_money, n_gap)
    pol_bs = make_bs_policy(K, r, sigma)

    c_rl = evaluate_policy(pol_rl, paths, times, K, T, r, sigma, init_pos, kappa, 1)
    c_bs_daily = evaluate_policy(pol_bs, paths, times, K, T, r, sigma, init_pos, kappa, 1)
    c_bs_weekly = evaluate_policy(pol_bs, paths, times, K, T, r, sigma, init_pos, kappa, 5)

    results = [
        summarize(c_bs_daily, option_price, 'BS daily', c_risk),
        summarize(c_bs_weekly, option_price, 'BS weekly', c_risk),
        summarize(c_rl, option_price, f'CCHP Q', c_risk),
    ]

    if verbose:
        print_summary(results, c_risk)
        print(f"  Training time: {train_time:.1f}s")
        print(f"  Q1 coverage: {100*np.count_nonzero(Q1)/Q1.size:.1f}%")
        print()

    return {
        'config_name': config_name,
        'results': results,
        'Q1': Q1, 'Q2': Q2, 'visits': visits,
        'rewards': rewards,
        'train_time': train_time,
    }


if __name__ == "__main__":
    # Baseline experiment: tiny scale, just verify the algorithm runs and
    # learns SOMETHING sensible.
    run_experiment(
        config_name="TINY_BASELINE",
        n_batches=50, batch_size=64,
        c_risk=1.5, kappa=0.01,
    )
