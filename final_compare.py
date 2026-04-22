"""
Final decisive: Single vs Double CCHP at full scale.
800 batches * 64 = 51,200 episodes per run.
3 seeds per algo, focus on nt=0.0002 (best config).

Key question: is DoubleQ undertrained at 400 batches because it splits
updates across 4 tables? Does it catch up at 800?
"""
import sys, pickle, os
from exp_framework import (cchp_dual_q_train, make_cchp_policy, make_bs_policy,
                          simulate_paths_batch, bs_delta_vec, bs_price_vec,
                          evaluate_policy)
from double_cchp import double_cchp_train
import numpy as np
import pandas as pd
import time

K, T, S0, sigma, r, dt, kappa = 100.0, 1.0, 100.0, 0.2, 0.0, 1/252, 0.01
n_steps = int(round(T/dt))
init_pos = float(bs_delta_vec(np.array([S0]), K, np.array([T]), r, sigma)[0])
option_price = float(bs_price_vec(np.array([S0]), K, np.array([T]), r, sigma)[0])
coarse = np.array([-0.1, -0.02, 0.0, 0.02, 0.1])
n_time, n_money, n_gap = 52, 40, 31
m_edges = np.linspace(0.5, 1.5, n_money + 1)
g_edges = np.linspace(-0.2, 0.2, n_gap + 1)

c_risk = 1.5
n_batches = 800
batch_size = 64
seeds = [42, 2026, 8]

# Eval setup (BIGGER eval: 10k paths)
rng = np.random.default_rng(777)
paths = simulate_paths_batch(S0, sigma, r, dt, n_steps, 10000, rng)
times = np.arange(n_steps + 1) * dt

pol_bs = make_bs_policy(K, r, sigma)
c_bsd = evaluate_policy(pol_bs, paths, times, K, T, r, sigma, init_pos, kappa, 1)
c_bsw = evaluate_policy(pol_bs, paths, times, K, T, r, sigma, init_pos, kappa, 5)
c_bsbw = evaluate_policy(pol_bs, paths, times, K, T, r, sigma, init_pos, kappa, 10)

def sum_stats(h):
    return h.mean(), h.std(), h.mean()+c_risk*h.std(), h[h >= np.quantile(h, 0.95)].mean()

bs_weekly_m, bs_weekly_s, bs_weekly_Y, bs_weekly_c = sum_stats(-c_bsw)
bs_bi_m, bs_bi_s, bs_bi_Y, bs_bi_c = sum_stats(-c_bsbw)
bs_daily_m, bs_daily_s, bs_daily_Y, bs_daily_c = sum_stats(-c_bsd)

print(f"BS daily:    Y={bs_daily_Y:.3f}  mean=${bs_daily_m:.3f}  std=${bs_daily_s:.3f}  CVaR=${bs_daily_c:.3f}")
print(f"BS weekly:   Y={bs_weekly_Y:.3f}  mean=${bs_weekly_m:.3f}  std=${bs_weekly_s:.3f}  CVaR=${bs_weekly_c:.3f}")
print(f"BS biweekly: Y={bs_bi_Y:.3f}  mean=${bs_bi_m:.3f}  std=${bs_bi_s:.3f}  CVaR=${bs_bi_c:.3f}")

def eval_full(Q1, Q2, nt_bonus):
    pol = make_cchp_policy(Q1, Q2, coarse, c_risk, K, r, sigma, T,
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
        C_now = bs_price_vec(S_now, K, tau_now, r, sigma)
        C_prev = bs_price_vec(S_prev, K, tau_prev, r, sigma)
        step = (S_now - S_prev)*pos_prev - np.abs(pos_next - pos_prev)*S_now*kappa - C_now + C_prev
        if t == n_steps:
            step -= pos_next * S_now * kappa
        costs += step
        if t < n_steps:
            pos_prev = pos_next
            ttm = np.full(n_trials, max(0.0, T - times[t]))
            pos_next = pol(paths[t], ttm, pos_prev)
            n_trades += (np.abs(pos_next - pos_prev) > 1e-8).astype(int)
    return -costs, n_trades

CACHE = 'dq_final_results.pkl'
if os.path.exists(CACHE):
    with open(CACHE, 'rb') as f:
        results = pickle.load(f)
    print(f"\n[Loaded cache] keys: {list(results.keys())}")
else:
    results = {}

nt_bonuses_eval = [0.0, 0.0002, 0.0005, 0.001]

# Train: Single first, then Double
for algo_name, train_fn in [("SingleQ", cchp_dual_q_train), ("DoubleQ", double_cchp_train)]:
    for seed in seeds:
        key = (algo_name, seed)
        if key in results:
            print(f"[SKIP] {key} cached")
            continue
        print(f"\n\n{'#'*70}\n# {algo_name} seed={seed}  (800 batches)\n{'#'*70}")
        t0 = time.time()
        if algo_name == "SingleQ":
            Q1, Q2, visits, hist = train_fn(
                K, T, S0, sigma, r, dt, kappa, init_pos,
                coarse, c_risk, n_time, n_money, n_gap, m_edges, g_edges,
                n_batches, batch_size,
                alpha_start=0.05, alpha_end=0.002,
                eps_start=1.0, eps_end=0.02,
                exploring_starts_frac=0.3,
                seed=seed, print_every=200,
            )
        else:
            Q1, Q2, visits, hist, _ = train_fn(
                K, T, S0, sigma, r, dt, kappa, init_pos,
                coarse, c_risk, n_time, n_money, n_gap, m_edges, g_edges,
                n_batches, batch_size,
                alpha_start=0.05, alpha_end=0.002,
                eps_start=1.0, eps_end=0.02,
                exploring_starts_frac=0.3,
                seed=seed, print_every=200,
            )
        elapsed = time.time() - t0
        print(f"Time: {elapsed:.0f}s  Coverage: {100*np.count_nonzero(Q1)/Q1.size:.1f}%")

        evals = {}
        for nt in nt_bonuses_eval:
            h, tr = eval_full(Q1, Q2, nt)
            m, s, y, cvar = sum_stats(h)
            evals[nt] = {'mean': m, 'std': s, 'Y': y, 'cvar': cvar, 'trades': tr.mean()}
            print(f"  nt={nt:.4f}: mean=${m:.3f}  std=${s:.3f}  Y={y:.3f}  "
                 f"CVaR=${cvar:.3f}  trades={tr.mean():.1f}")
        results[key] = {'eval': evals, 'hist': hist, 'time': elapsed}
        with open(CACHE, 'wb') as f:
            pickle.dump(results, f)
        print(f"[Saved]")

# Aggregate
print("\n\n" + "="*100)
print("  FINAL COMPARISON @ 51,200 episodes x 3 seeds, 10,000 eval paths")
print(f"  BS daily Y={bs_daily_Y:.3f}  BS weekly Y={bs_weekly_Y:.3f}  BS biweekly Y={bs_bi_Y:.3f}")
print("="*100)
print(f"{'Algo':<10} {'nt_bonus':<10} {'Mean':>10} {'Std':>10} {'CVaR':>10} "
      f"{'Y':>9} {'Y_range':>10} {'Trades':>8}")
print("-"*100)
for algo in ["SingleQ", "DoubleQ"]:
    for nt in nt_bonuses_eval:
        Ys = [results[(algo, s)]['eval'][nt]['Y'] for s in seeds]
        ms = [results[(algo, s)]['eval'][nt]['mean'] for s in seeds]
        ss = [results[(algo, s)]['eval'][nt]['std'] for s in seeds]
        cs = [results[(algo, s)]['eval'][nt]['cvar'] for s in seeds]
        ts = [results[(algo, s)]['eval'][nt]['trades'] for s in seeds]
        print(f"{algo:<10} {nt:<10.4f} {np.mean(ms):>10.3f} {np.mean(ss):>10.3f} "
              f"{np.mean(cs):>10.3f} {np.mean(Ys):>9.3f} {max(Ys)-min(Ys):>10.3f} "
              f"{np.mean(ts):>8.1f}")
    print("-"*100)
