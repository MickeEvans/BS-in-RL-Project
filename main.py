"""
main.py — Run the RL hedging experiment
========================================
All tuneable parameters are defined in PARAMS below.
Trains Q-learning and/or Double Q-learning agents at various c-values,
benchmarks against Black-Scholes, and produces a 6-panel diagnostic plot.

Usage:
    python main.py                     (interactive — shows plot)
    MPLBACKEND=Agg python main.py      (headless  — console table only)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from environment import (
    bs_benchmark, bs_quantised_benchmark,
    extract_policy, bs_policy_grid,
)
from agents import QHedger, DoubleQHedger, train, evaluate

warnings.filterwarnings("ignore")
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit these to change the experiment
# ═════════════════════════════════════════════════════════════════════════════
PARAMS = dict(
    # ── Market ───────────────────────────────────────────────────────────
    S0    = 100.0,      # initial stock price
    K     = 100.0,      # strike price (ATM)
    T     = 1.0,        # maturity in years
    N     = 252,        # trading days (rebalancing steps per episode)
    sigma = 0.20,       # annual volatility
    r     = 0.0,        # risk-free rate
    kappa = 0.01,       # proportional transaction-cost half-spread

    # ── State grid ───────────────────────────────────────────────────────
    N_TIME  = 5,        # bins for time-to-maturity
    N_MONEY = 15,       # bins for log-moneyness
    M_LO    = -0.5,     # lower bound of log-moneyness grid
    M_HI    =  0.5,     # upper bound of log-moneyness grid

    # ── Action grid ──────────────────────────────────────────────────────
    N_ACT = 5,          # number of discrete holdings in [-1, 0]
                        # 5 → H ∈ {-1.0, -0.75, -0.5, -0.25, 0.0}
)

# ── Training / evaluation settings ──────────────────────────────────────────
TRAIN_EPISODES = 30_000
EVAL_EPISODES  = 5_000

# ── Agent configurations to train ───────────────────────────────────────────
# Each dict: agent class, Cao c-value, display name.
# Set agent_class to DoubleQHedger to use double Q-learning instead.
CONFIGS = [
    dict(agent_class=QHedger, c=0.0, name="QL_c0",  label="QL   c=0   "),
    dict(agent_class=QHedger, c=0.7, name="QL_c07", label="QL   c=0.7 "),
    dict(agent_class=QHedger, c=1.5, name="QL_c15", label="QL   c=1.5 "),
    dict(agent_class=QHedger, c=2.0, name="QL_c20", label="QL   c=2.0 "),
    # Uncomment to add Double Q-learning agents:
    # dict(agent_class=DoubleQHedger, c=4.0, name="DQL_c40", label="DQL  c=4.0"),
]


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("RL Hedging  —  APL reward, no warm start")
    print("=" * 78)

    # ── BS benchmarks ────────────────────────────────────────────────────
    print("\n[BS delta benchmark]")
    bs_pnl, bs_tc, bs_tr = bs_benchmark(PARAMS, EVAL_EPISODES)
    print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
          % (bs_tc.mean(), bs_pnl.std(), bs_pnl.mean(), bs_tr.mean()))

    print("\n[BS quantised benchmark]")
    bsq_pnl, bsq_tc, bsq_tr = bs_quantised_benchmark(PARAMS, EVAL_EPISODES)
    print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
          % (bsq_tc.mean(), bsq_pnl.std(), bsq_pnl.mean(), bsq_tr.mean()))

    # ── Train & evaluate each agent config ───────────────────────────────
    agents, results = [], {}
    for cfg in CONFIGS:
        print("\n[Train %s]" % cfg["label"])
        AgentClass = cfg["agent_class"]
        ag = AgentClass(PARAMS, name=cfg["name"], c=cfg["c"])
        train(ag, PARAMS, n_ep=TRAIN_EPISODES, gamma=1.0)
        pnl, tc, tr = evaluate(ag, PARAMS, n_ep=EVAL_EPISODES)
        results[cfg["name"]] = dict(pnl=pnl, tc=tc, trades=tr,
                                    label=cfg["label"])
        agents.append(ag)
        print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
              "   (BS TC=%.4f, trades=%.1f)"
              % (tc.mean(), pnl.std(), pnl.mean(), tr.mean(),
                 bs_tc.mean(), bs_tr.mean()))

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("%-28s  %10s  %10s  %10s  %10s"
          % ("Strategy", "Mean TC", "Std PnL", "Mean PnL", "Trades/ep"))
    print("-" * 78)
    print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
          % ("BS Delta", bs_tc.mean(), bs_pnl.std(),
             bs_pnl.mean(), bs_tr.mean()))
    print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
          % ("BS quantised", bsq_tc.mean(), bsq_pnl.std(),
             bsq_pnl.mean(), bsq_tr.mean()))
    for name, res in results.items():
        print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
              % (res["label"][:28], res["tc"].mean(), res["pnl"].std(),
                 res["pnl"].mean(), res["trades"].mean()))
    print("=" * 78)

    # ── 6-panel plot ─────────────────────────────────────────────────────
    plot_results(bs_pnl, bs_tc, bsq_pnl, bsq_tc, results, agents, CONFIGS)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(bs_pnl, bs_tc, bsq_pnl, bsq_tc, results, agents, configs):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("RL Hedging  —  APL reward, no warm start",
                 fontsize=14, fontweight="bold")

    palette = ["tomato", "darkorange", "green", "purple", "deeppink",
               "teal", "brown", "olive"]

    # ── PnL distribution ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(bs_pnl, bins=70, alpha=0.55, label="BS Delta",
            color="steelblue", density=True)
    for (name, res), c in zip(results.items(), palette):
        ax.hist(res["pnl"], bins=70, alpha=0.45, label=res["label"],
                color=c, density=True)
    ax.set_title("PnL distribution")
    ax.set_xlabel("PnL"); ax.set_ylabel("Density")
    ax.legend(fontsize=7)

    # ── TC distribution ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(bs_tc, bins=70, alpha=0.55, label="BS Delta",
            color="steelblue", density=True)
    for (name, res), c in zip(results.items(), palette):
        ax.hist(res["tc"], bins=70, alpha=0.45, label=res["label"],
                color=c, density=True)
    ax.set_title("Transaction-cost distribution")
    ax.set_xlabel("TC"); ax.set_ylabel("Density")
    ax.legend(fontsize=7)

    # ── TC vs Risk scatter ──────────────────────────────────────────────
    ax = axes[0, 2]
    ax.scatter(bs_tc.mean(), bs_pnl.std(), s=200, marker="*",
               color="steelblue", zorder=5, label="BS Delta")
    ax.scatter(bsq_tc.mean(), bsq_pnl.std(), s=120, marker="P",
               color="grey", zorder=5, label="BS quantised")
    for (name, res), c in zip(results.items(), palette):
        ax.scatter(res["tc"].mean(), res["pnl"].std(), s=90, marker="o",
                   color=c, zorder=5, label=res["label"])
    ax.set_title("TC  vs  Risk (Std PnL)")
    ax.set_xlabel("Mean transaction cost"); ax.set_ylabel("Std(PnL)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # ── Policy heatmaps (pick lowest-TC agent) ──────────────────────────
    best_name = min(results, key=lambda k: results[k]["tc"].mean())
    best_idx  = next(i for i, cfg in enumerate(configs)
                     if cfg["name"] == best_name)
    best_agent = agents[best_idx]
    M_LO, M_HI = PARAMS["M_LO"], PARAMS["M_HI"]

    ax = axes[1, 0]
    im = ax.imshow(bs_policy_grid(PARAMS), origin="lower", aspect="auto",
                   vmin=-1, vmax=0, cmap="RdYlGn",
                   extent=[M_LO, M_HI, 0, 1])
    ax.set_title("BS Hedge  H = −δ")
    ax.set_xlabel("Log-Moneyness log(S/K)")
    ax.set_ylabel("Time-to-Maturity τ/T")
    plt.colorbar(im, ax=ax, label="H")

    ax = axes[1, 1]
    im = ax.imshow(extract_policy(best_agent, PARAMS),
                   origin="lower", aspect="auto",
                   vmin=-1, vmax=0, cmap="RdYlGn",
                   extent=[M_LO, M_HI, 0, 1])
    ax.set_title("RL Policy  (%s)" % best_agent.name)
    ax.set_xlabel("Log-Moneyness log(S/K)")
    ax.set_ylabel("Time-to-Maturity τ/T")
    plt.colorbar(im, ax=ax, label="H")

    ax = axes[1, 2]
    diff = extract_policy(best_agent, PARAMS) - bs_policy_grid(PARAMS)
    vmax = max(np.abs(diff).max(), 1e-6)
    im = ax.imshow(diff, origin="lower", aspect="auto",
                   vmin=-vmax, vmax=vmax, cmap="bwr",
                   extent=[M_LO, M_HI, 0, 1])
    ax.set_title("RL  −  BS Hedge")
    ax.set_xlabel("Log-Moneyness log(S/K)")
    ax.set_ylabel("Time-to-Maturity τ/T")
    plt.colorbar(im, ax=ax, label="H diff")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nPlot saved to results.png")
    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()