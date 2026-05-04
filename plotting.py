"""
Plotting — generate the four result figures from experiment outputs.

Figure 1: Efficient frontier (hedging risk vs transaction cost)
Figure 2: P&L distribution histogram + box plot
Figure 3: Heat-map of the learned policy (no-trade band)
Figure 4: Final out-of-sample bar chart comparison
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import ACTIONS, M_EDGES, E_EDGES
from simulate import sim_bs, sim_band, sim_Q, sim_double_Q


def plot_results(results):
    """
    Generate four matplotlib figures from the experiment outputs dict.
    Returns (fig1, fig2, fig3, fig4). Call plt.show() to display.
    """
    m_bs     = results['bs_oos']
    m_band   = results['band_oos']
    m_ql15   = results['ql_15k_oos']
    m_qle    = results['ql_ext_oos']
    m_dq     = results['dql_oos']
    band_r   = results['band_results']
    lam_r    = results['lambda_results']
    best_lam = results['best_lam']
    Q_best   = results['Q_best']
    oos      = results['oos_paths']

    # Re-run simulators to get P&L vectors for distribution plots
    pnl_bs,   _, _ = sim_bs(oos)
    pnl_band, _, _ = sim_band(0.20, oos)
    pnl_ql,   _, _ = sim_Q(Q_best, oos)
    pnl_dq,   _, _ = sim_double_Q(results['Q_A'], results['Q_B'], oos)

    # ── Figure 1: Efficient frontier ─────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(10, 6))

    band_tcs  = [m['mean_tc']  for _, m in sorted(band_r.items())]
    band_stds = [m['std_pnl']  for _, m in sorted(band_r.items())]
    ax.plot(band_tcs, band_stds, 's-', color='green', alpha=0.7,
            label='No-trade band', markersize=10)
    for hw, m in sorted(band_r.items()):
        ax.annotate(f'{hw}', (m['mean_tc'], m['std_pnl']),
                    xytext=(5, -10), textcoords='offset points',
                    fontsize=8, color='darkgreen')

    lam_tcs  = [d['m']['mean_tc'] for _, d in sorted(lam_r.items())]
    lam_stds = [d['m']['std_pnl'] for _, d in sorted(lam_r.items())]
    ax.plot(lam_tcs, lam_stds, 'o-', color='C0',
            label='Q-Learning (λ sweep)', markersize=10)
    for lam, d in sorted(lam_r.items()):
        ax.annotate(f'λ={lam}', (d['m']['mean_tc'], d['m']['std_pnl']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='navy')

    ax.scatter([m_qle['mean_tc']], [m_qle['std_pnl']],
               s=250, marker='*', color='gold', edgecolor='navy',
               zorder=10, label='QL best (extended)')
    ax.scatter([m_dq['mean_tc']], [m_dq['std_pnl']],
               s=200, marker='D', color='orange', edgecolor='black',
               zorder=10, label='Double Q-Learning')
    ax.scatter([m_bs['mean_tc']], [m_bs['std_pnl']],
               s=250, marker='X', color='red',
               zorder=10, label='BS-delta hedge')

    ax.set_xlabel('Mean Transaction Cost per option', fontsize=12)
    ax.set_ylabel('Hedging error (std of P&L)', fontsize=12)
    ax.set_title('Efficient frontier: Hedging risk vs Transaction cost\n'
                 '(20,000 OOS paths)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig1.tight_layout()

    # ── Figure 2: P&L distributions ──────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(-14, 4, 50)
    axes[0].hist(pnl_bs,   bins=bins, alpha=0.5, density=True, color='red',
                 label=f"BS (μ={m_bs['mean_pnl']:.2f}, σ={m_bs['std_pnl']:.2f})")
    axes[0].hist(pnl_band, bins=bins, alpha=0.5, density=True, color='green',
                 label=f"Band 0.20 (μ={m_band['mean_pnl']:.2f}, σ={m_band['std_pnl']:.2f})")
    axes[0].hist(pnl_ql,   bins=bins, alpha=0.5, density=True, color='blue',
                 label=f"QL λ={best_lam} (μ={m_qle['mean_pnl']:.2f}, σ={m_qle['std_pnl']:.2f})")
    axes[0].hist(pnl_dq,   bins=bins, alpha=0.5, density=True, color='orange',
                 label=f"Double-QL (μ={m_dq['mean_pnl']:.2f}, σ={m_dq['std_pnl']:.2f})")
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Terminal P&L per option', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('P&L distribution comparison (20k OOS paths)', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    bp = axes[1].boxplot(
        [pnl_bs, pnl_band, pnl_ql, pnl_dq],
        tick_labels=['BS-delta', 'Band 0.20', f'QL λ={best_lam}', 'Double-QL'],
        patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'],
                        ['salmon', 'lightgreen', 'steelblue', 'sandybrown']):
        patch.set_facecolor(c)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('P&L', fontsize=12)
    axes[1].set_title('P&L dispersion (box plot)', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3)
    fig2.tight_layout()

    # ── Figure 3: Learned policy heat-map ────────────────────────────────────
    fig3, axes = plt.subplots(2, 3, figsize=(15, 9))
    t_buckets   = [0, 2, 4, 6, 8]
    time_labels = ['0–5 days', '15–30 days', '55–90 days',
                   '130–175 days', '215–240 days']

    for idx, (ti, tlbl) in enumerate(zip(t_buckets, time_labels)):
        ax = axes.flat[idx]
        best_a = np.argmax(Q_best[ti, :, :, :], axis=2)
        action_vals = ACTIONS[best_a]
        im = ax.imshow(action_vals.T, origin='lower', aspect='auto',
                       cmap='RdBu_r', vmin=-0.2, vmax=0.2,
                       extent=[M_EDGES[0], M_EDGES[-1],
                               E_EDGES[0], E_EDGES[-1]])
        ax.set_title(f'Time to maturity: {tlbl}', fontsize=11)
        ax.set_xlabel('Moneyness S/K')
        ax.set_ylabel('Position error (pos − δ_BS)')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Δ position')

    axes.flat[5].axis('off')
    axes.flat[5].text(0.05, 0.85, 'LEARNED HEDGING POLICY',
                      fontsize=15, weight='bold',
                      transform=axes.flat[5].transAxes)
    axes.flat[5].text(0.05, 0.65,
                      'x: moneyness (S/K)\n'
                      'y: position error (pos − δ_BS)\n\n'
                      'Red  (+)  → buy stock\n'
                      'Blue (−)  → sell stock\n'
                      'White (0) → do nothing\n\n'
                      'The white band around y=0 is the\n'
                      'learned NO-TRADE REGION — the agent\n'
                      'has rediscovered the classical band\n'
                      'structure.',
                      fontsize=10, transform=axes.flat[5].transAxes, va='top')

    fig3.suptitle(f'Q-Learning policy (λ={best_lam}, extended training)',
                  fontsize=14)
    fig3.tight_layout()

    # ── Figure 4: Bar chart comparison ───────────────────────────────────────
    fig4, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    methods = [
        ('BS-delta',                m_bs,    'red'),
        ('Band hw=0.20',            m_band,  'green'),
        (f'QL λ={best_lam} (15k)',  m_ql15,  'lightsteelblue'),
        (f'QL λ={best_lam} ext.',   m_qle,   'navy'),
        ('Double-QL',               m_dq,    'orange'),
    ]
    x = np.arange(len(methods))
    labels  = [m[0] for m in methods]
    colours = [m[2] for m in methods]
    stds    = [m[1]['std_pnl'] for m in methods]
    tcs     = [m[1]['mean_tc'] for m in methods]
    cvars   = [-m[1]['CVaR_5'] for m in methods]

    axes[0].bar(x, stds, color=colours)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_ylabel('Std P&L'); axes[0].set_title('Hedging risk (lower = better)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, tcs, color=colours)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_ylabel('Mean TC'); axes[1].set_title('Transaction cost (lower = better)')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x, cvars, color=colours)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20, ha='right')
    axes[2].set_ylabel('|CVaR 5%|'); axes[2].set_title('Tail risk (lower = better)')
    axes[2].grid(axis='y', alpha=0.3)

    fig4.suptitle('Final out-of-sample comparison', fontsize=14, y=1.02)
    fig4.tight_layout()

    return fig1, fig2, fig3, fig4
