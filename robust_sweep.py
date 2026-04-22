"""
Robustness check: retrain with multiple seeds, fine-grained nt_bonus sweep.
"""
from exp_framework import (run_experiment, make_cchp_policy, make_bs_policy,
                           simulate_paths_batch, bs_delta_vec, bs_price_vec,
                           evaluate_policy, summarize)
import numpy as np

coarse = np.array([-0.1, -0.02, 0.0, 0.02, 0.1])
K, T, S0, sigma, rr, dt, kappa = 100.0, 1.0, 100.0, 0.2, 0.0, 1/252, 0.01
n_steps = int(round(T/dt))
init_pos = float(bs_delta_vec(np.array([S0]), K, np.array([T]), rr, sigma)[0])
n_time, n_money, n_gap = 52, 40, 31
m_edges = np.linspace(0.5, 1.5, n_money + 1)
g_edges = np.linspace(-0.2, 0.2, n_gap + 1)
option_price = 7.9656

# BS weekly reference (reused)
rng = np.random.default_rng(777)
paths = simulate_paths_batch(S0, sigma, rr, dt, n_steps, 5000, rng)
times = np.arange(n_steps + 1) * dt
pol_bs = make_bs_policy(K, rr, sigma)
c_bs_weekly = evaluate_policy(pol_bs, paths, times, K, T, rr, sigma, init_pos, kappa, 5)
bs_mean, bs_std = (-c_bs_weekly).mean(), (-c_bs_weekly).std()
bs_Y = bs_mean + 1.5*bs_std
print(f"BS weekly (reference):  mean={bs_mean:.3f}  std={bs_std:.3f}  Y={bs_Y:.3f}\n")

def evaluate_rl(Q1, Q2, nt_bonus):
    pol = make_cchp_policy(Q1, Q2, coarse, 1.5, K, rr, sigma, T,
                           n_time, m_edges, g_edges, n_money, n_gap,
                           nt_bonus=nt_bonus)
    n_trials = paths.shape[1]
    costs = np.zeros(n_trials)
    n_trades = np.zeros(n_trials, dtype=int)
    pos_prev = np.full(n_trials, init_pos)
    ttm0 = np.full(n_trials, T)
    pos_next = pol(paths[0], ttm0, pos_prev)
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
            pos_next = pol(paths[t], ttm, pos_prev)
            n_trades += (np.abs(pos_next - pos_prev) > 1e-8).astype(int)
    h = -costs
    return h.mean(), h.std(), h.mean()+1.5*h.std(), n_trades.mean()

# Train three seeds, sweep nt_bonus on each
seeds = [8, 42, 2026]
nt_values = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0015, 0.002]

print(f"{'Seed':<6} {'nt_bonus':<10} {'Mean':>8} {'Std':>8} {'Y':>8}  {'vs_BSw':>8}  {'trades':>8}")
print("-" * 72)

all_results = {nt: [] for nt in nt_values}
for seed in seeds:
    r = run_experiment(config_name=f'seed{seed}', seed=seed,
                      n_batches=500, batch_size=64, c_risk=1.5, actions=coarse,
                      g_min=-0.2, g_max=0.2, n_gap=31, verbose=False)
    Q1, Q2 = r['Q1'], r['Q2']
    for nt in nt_values:
        mean, std, Y, trades = evaluate_rl(Q1, Q2, nt)
        all_results[nt].append((mean, std, Y, trades))
        gap = Y - bs_Y
        mark = "  <-- BEATS" if gap < 0 else ""
        print(f"{seed:<6} {nt:<10.4f} {mean:>8.3f} {std:>8.3f} {Y:>8.3f}  "
              f"{gap:>+8.3f}  {trades:>8.1f}{mark}")
    print("-" * 72)

print("\nMean across seeds:")
print(f"{'nt_bonus':<10} {'Mean_avg':>10} {'Std_avg':>10} {'Y_avg':>10}  {'vs_BSw':>8}  {'trades':>8}")
print("-" * 72)
for nt in nt_values:
    arr = np.array(all_results[nt])
    m, s, y, t = arr.mean(axis=0)
    gap = y - bs_Y
    mark = "  <-- BEATS" if gap < 0 else ""
    print(f"{nt:<10.4f} {m:>10.3f} {s:>10.3f} {y:>10.3f}  {gap:>+8.3f}  {t:>8.1f}{mark}")
