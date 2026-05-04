"""
main.py — Run the full hedging experiment.

This is the entry point. It orchestrates:
  1. Black-Scholes benchmark evaluation
  2. No-trade band sweep (analytical baselines)
  3. Q-Learning λ-sweep (find the best risk-aversion)
  4. Extended fine-tuning of the best agent
  5. Double Q-Learning ablation
  6. Out-of-sample evaluation on 20,000 fresh paths
  7. Plot generation

Usage:
    python main.py

To tweak parameters, edit config.py (market params, seeds, episode counts).
"""

import numpy as np

from config import (
    S0, K, T, N, sigma, kappa,
    LAMBDA_SWEEP, BAND_SWEEP,
    EPISODES_SWEEP, EPISODES_EXT, N_PARALLEL,
    EVAL_PATHS, OOS_PATHS,
    SEED_EVAL, SEED_OOS,
)
from black_scholes import V0
from environment import N_T, N_M, N_E, N_A, sim_paths
from agent import warm_init, train, train_double
from simulate import sim_bs, sim_band, sim_Q, sim_double_Q, mets


def run_full_experiment():
    """Run the complete experiment pipeline and return a results dict."""

    print("=" * 70)
    print("EUROPEAN CALL OPTION HEDGING — TABULAR Q-LEARNING")
    print("=" * 70)
    print(f"Parameters: S0={S0}, K={K}, T={T}, N={N}, σ={sigma}, κ={kappa}")
    print(f"State space: {N_T}×{N_M}×{N_E} × {N_A} actions = {N_T*N_M*N_E*N_A} cells")
    print(f"Initial BS price V0 = {V0:.4f}")
    print()

    eval_paths = sim_paths(EVAL_PATHS, seed=SEED_EVAL)

    # ── 1. BS Benchmark ──────────────────────────────────────────────────────
    print("── Benchmark: Black-Scholes delta hedge ──")
    p, t, n = sim_bs(eval_paths)
    m_bs    = mets(p, t, n, "BS-delta")
    print(f"  mean P&L = {m_bs['mean_pnl']:+.4f}  std = {m_bs['std_pnl']:.4f}  "
          f"TC = {m_bs['mean_tc']:.4f}  CVaR5% = {m_bs['CVaR_5']:+.4f}")

    # ── 2. No-trade band sweep ───────────────────────────────────────────────
    print("\n── Benchmark: No-trade bands ──")
    band_results = {}
    for hw in BAND_SWEEP:
        p, t, n = sim_band(hw, eval_paths)
        m       = mets(p, t, n, f"Band-{hw}")
        band_results[hw] = m
        print(f"  hw={hw:.2f}: std={m['std_pnl']:.3f} TC={m['mean_tc']:.3f} "
              f"CVaR5%={m['CVaR_5']:+.3f}")

    # ── 3. Q-Learning λ-sweep ────────────────────────────────────────────────
    print(f"\n── Q-Learning: risk-aversion (λ) sweep, {EPISODES_SWEEP} episodes each ──")
    lambda_results = {}
    for lam in LAMBDA_SWEEP:
        Q = np.zeros((N_T, N_M, N_E, N_A))
        v = np.zeros_like(Q, dtype=int)
        warm_init(Q)
        train(Q, v, n_episodes=EPISODES_SWEEP, risk_lambda=lam,
              eps_start=0.4, eps_end=0.03,
              seed=100 + int(lam * 100), n_parallel=N_PARALLEL)
        p, t, n = sim_Q(Q, eval_paths)
        m       = mets(p, t, n, f"QL-λ{lam}")
        lambda_results[lam] = {'m': m, 'Q': Q}
        print(f"  λ={lam:.2f}: std={m['std_pnl']:.3f} TC={m['mean_tc']:.3f} "
              f"CVaR5%={m['CVaR_5']:+.3f}")

    # Pick best λ (lowest TC while keeping std close to BS)
    best_lam = min(lambda_results,
                   key=lambda x: lambda_results[x]['m']['mean_tc']
                                 + 1.5 * max(0, lambda_results[x]['m']['std_pnl']
                                              - m_bs['std_pnl']))
    print(f"  → Best λ: {best_lam}")

    # ── 4. Extended fine-tuning ──────────────────────────────────────────────
    print(f"\n── Extended training of best agent (λ={best_lam}, +{EPISODES_EXT} eps) ──")
    Q_best = lambda_results[best_lam]['Q'].copy()
    v_best = np.ones_like(Q_best, dtype=int) * 20
    train(Q_best, v_best, n_episodes=EPISODES_EXT, risk_lambda=best_lam,
          lr0=0.08, eps_start=0.10, eps_end=0.005, seed=5555,
          n_parallel=N_PARALLEL)
    p, t, n = sim_Q(Q_best, eval_paths)
    m_best  = mets(p, t, n, "QL-best-extended")
    print(f"  extended: std={m_best['std_pnl']:.3f} TC={m_best['mean_tc']:.3f} "
          f"CVaR5%={m_best['CVaR_5']:+.3f}")

    # ── 5. Double Q-Learning ablation ────────────────────────────────────────
    print(f"\n── Double Q-Learning ablation (λ={best_lam}, {EPISODES_SWEEP} eps) ──")
    Q_A = np.zeros((N_T, N_M, N_E, N_A))
    Q_B = np.zeros((N_T, N_M, N_E, N_A))
    vA  = np.zeros_like(Q_A, dtype=int)
    vB  = np.zeros_like(Q_B, dtype=int)
    warm_init(Q_A); warm_init(Q_B)
    train_double(Q_A, Q_B, vA, vB, n_episodes=EPISODES_SWEEP,
                 risk_lambda=best_lam,
                 eps_start=0.4, eps_end=0.03, seed=42,
                 n_parallel=N_PARALLEL)
    p, t, n = sim_double_Q(Q_A, Q_B, eval_paths)
    m_dq    = mets(p, t, n, "Double-QL")
    print(f"  Double Q: std={m_dq['std_pnl']:.3f} TC={m_dq['mean_tc']:.3f} "
          f"CVaR5%={m_dq['CVaR_5']:+.3f}")

    # ── 6. Out-of-sample evaluation ──────────────────────────────────────────
    print(f"\n── Out-of-sample evaluation ({OOS_PATHS:,} fresh paths, different seed) ──")
    oos = sim_paths(OOS_PATHS, seed=SEED_OOS)

    p, t, n = sim_bs(oos);                                  m_bs_oos    = mets(p, t, n, "BS")
    p, t, n = sim_band(0.20, oos);                           m_b20_oos   = mets(p, t, n, "Band0.20")
    p, t, n = sim_Q(lambda_results[best_lam]['Q'], oos);     m_ql15_oos  = mets(p, t, n, f"QL-λ{best_lam}-15k")
    p, t, n = sim_Q(Q_best, oos);                            m_qlext_oos = mets(p, t, n, "QL-best-ext")
    p, t, n = sim_double_Q(Q_A, Q_B, oos);                   m_dq_oos    = mets(p, t, n, "Double-QL")

    print(f"\n{'Method':<22} {'Mean PnL':>10} {'Std PnL':>9} {'Mean TC':>9} "
          f"{'Sharpe':>8} {'CVaR 5%':>9}")
    print('-' * 78)
    for name, m in [
        ("BS-delta",                 m_bs_oos),
        ("Band hw=0.20",             m_b20_oos),
        (f"QL λ={best_lam} (15k)",   m_ql15_oos),
        (f"QL λ={best_lam} extended", m_qlext_oos),
        (f"Double-QL λ={best_lam}",  m_dq_oos),
    ]:
        print(f"{name:<22} {m['mean_pnl']:>+10.4f} {m['std_pnl']:>9.4f} "
              f"{m['mean_tc']:>9.4f} {m['sharpe']:>+8.4f} {m['CVaR_5']:>+9.4f}")

    tc_save = (m_bs_oos['mean_tc'] - m_qlext_oos['mean_tc']) / m_bs_oos['mean_tc'] * 100
    std_chg = (m_qlext_oos['std_pnl'] - m_bs_oos['std_pnl']) / m_bs_oos['std_pnl'] * 100
    cvar_imp = m_qlext_oos['CVaR_5'] - m_bs_oos['CVaR_5']
    print(f"\nQ-Learning (extended) vs Black-Scholes:")
    print(f"  Transaction cost reduction: {tc_save:+.1f}%")
    print(f"  Hedging risk (std) change:  {std_chg:+.1f}%")
    print(f"  Tail risk (CVaR) improved:  {cvar_imp:+.3f}")

    return {
        'bs_oos':         m_bs_oos,
        'band_oos':       m_b20_oos,
        'ql_15k_oos':     m_ql15_oos,
        'ql_ext_oos':     m_qlext_oos,
        'dql_oos':        m_dq_oos,
        'band_results':   band_results,
        'lambda_results': lambda_results,
        'best_lam':       best_lam,
        'Q_best':         Q_best,
        'Q_A':            Q_A,
        'Q_B':            Q_B,
        'oos_paths':      oos,
    }


if __name__ == "__main__":
    np.random.seed(0)
    results = run_full_experiment()

    print("\n── Generating plots... ──")
    from plotting import plot_results
    import matplotlib.pyplot as plt

    figs = plot_results(results)
    plt.show()

    print("Done.")