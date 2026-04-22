"""
Sweep no-trade bonus: load E24 Q-tables, evaluate with varying nt_bonus.
This tests post-hoc whether tie-breaking toward action=0 helps.

nt_bonus subtracts a value from action 0's risk-adjusted Q. The scale of
Q values is on the order of the per-episode cost, ~0.05 in K-units
(roughly $5). So nt_bonus values in [0, 0.02] scan roughly 0% to 40% of
a typical Q-value gap.
"""
from exp_framework import (run_experiment, make_cchp_policy, make_bs_policy,
                           simulate_paths_batch, bs_delta_vec, bs_price_vec,
                           evaluate_policy, summarize)
import numpy as np

coarse = np.array([-0.1, -0.02, 0.0, 0.02, 0.1])

print("Training E24 (c_risk=1.5, 500 batches, narrow+fine gap)...")
r = run_experiment(config_name='E24_bank',
                   n_batches=500, batch_size=64, c_risk=1.5, actions=coarse,
                   g_min=-0.2, g_max=0.2, n_gap=31, verbose=False)
Q1, Q2 = r['Q1'], r['Q2']

K, T, S0, sigma, rr, dt, kappa = 100.0, 1.0, 100.0, 0.2, 0.0, 1/252, 0.01
n_steps = int(round(T/dt))
init_pos = float(bs_delta_vec(np.array([S0]), K, np.array([T]), rr, sigma)[0])
n_time, n_money, n_gap = 52, 40, 31
m_edges = np.linspace(0.5, 1.5, n_money + 1)
g_edges = np.linspace(-0.2, 0.2, n_gap + 1)

# Generate eval paths ONCE, reuse across all bonuses
rng = np.random.default_rng(777)
paths = simulate_paths_batch(S0, sigma, rr, dt, n_steps, 5000, rng)
times = np.arange(n_steps + 1) * dt

# BS benchmarks
pol_bs = make_bs_policy(K, rr, sigma)
c_bs_weekly = evaluate_policy(pol_bs, paths, times, K, T, rr, sigma, init_pos, kappa, 5)
bs_weekly_sum = summarize(c_bs_weekly, 7.9656, 'BS weekly', 1.5)
print(f"\nBS weekly: mean={bs_weekly_sum['mean']:.3f}  std={bs_weekly_sum['std']:.3f}  "
      f"Y={bs_weekly_sum['mv_obj']:.3f}")

# Sweep no-trade bonuses
print(f"\n{'nt_bonus':<10} {'Mean':>8} {'Std':>8} {'CVaR':>8} {'Y':>8}  {'vs_BSw':>8}  {'trades':>8}")
print("-" * 70)

for nt_bonus in [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]:
    pol_rl = make_cchp_policy(Q1, Q2, coarse, 1.5, K, rr, sigma, T,
                              n_time, m_edges, g_edges, n_money, n_gap,
                              nt_bonus=nt_bonus)
    # Count trades and cost in one pass
    n_trials = paths.shape[1]
    costs = np.zeros(n_trials)
    n_trades = np.zeros(n_trials, dtype=int)
    pos_prev = np.full(n_trials, init_pos)
    ttm0 = np.full(n_trials, T)
    pos_next = pol_rl(paths[0], ttm0, pos_prev)
    n_trades += (np.abs(pos_next - pos_prev) > 1e-8).astype(int)
    for t in range(1, n_steps + 1):
        S_now, S_prev = paths[t], paths[t-1]
        tau_now = max(0.0, T - times[t])
        tau_prev = max(0.0, T - times[t-1])
        C_now = bs_price_vec(S_now, K, tau_now, rr, sigma)
        C_prev = bs_price_vec(S_prev, K, tau_prev, rr, sigma)
        step = (S_now - S_prev)*pos_prev - np.abs(pos_next - pos_prev)*S_now*kappa - C_now + C_prev
        if t == n_steps:
            step -= pos_next * S_now * kappa
        costs += step
        if t < n_steps:
            pos_prev = pos_next
            ttm = np.full(n_trials, max(0.0, T - times[t]))
            pos_next = pol_rl(paths[t], ttm, pos_prev)
            n_trades += (np.abs(pos_next - pos_prev) > 1e-8).astype(int)
    h = -costs
    mean, std = h.mean(), h.std()
    cvar = h[h >= np.quantile(h, 0.95)].mean()
    Y = mean + 1.5*std
    gap = Y - bs_weekly_sum['mv_obj']
    print(f"{nt_bonus:<10.4f} {mean:>8.3f} {std:>8.3f} {cvar:>8.3f} "
          f"{Y:>8.3f}  {gap:>+8.3f}  {n_trades.mean():>8.1f}")
