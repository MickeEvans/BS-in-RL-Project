"""
main.py — Run the RL hedging experiment
========================================
All tuneable parameters are defined in PARAMS below.
Trains Q-learning and/or Double Q-learning agents at various c-values,
benchmarks against Black-Scholes, and produces diagnostic plots.

Summary table reports:
  - Mean PnL
  - Mean hedge cost (% of initial option price)
  - Std  hedge cost (% of initial option price)
  - Mean terminal hedging error (Portfolio − discounted option value)

Benchmarks:
  - BS Delta         : continuous delta hedge (frictionless upper bound)
  - BS Band          : delta hedge with no-trade band (cost-aware benchmark)

Plots produced:
  1. PnL distribution histogram   — all strategies overlaid
  2. Hedge-cost distribution      — all strategies overlaid (% of V0)
  3. Hedging-error progression    — mean |HE_t| over time, per strategy
  4. Hedge ratio vs moneyness     — at fixed TTM, RL (ATM/Selling/Buying)
                                    vs theoretical BLS delta

Usage:
    python main.py                     (interactive — shows plot)
    MPLBACKEND=Agg python main.py      (headless  — console table only)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import warnings

from environment import (
    bs_benchmark, bs_band_benchmark, bs_price, bs_delta,
    rl_hedge_ratio, build_money_edges,
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
    N_MONEY = 15,       # bins for moneyness dimension

    # Moneyness binning convention — choose ONE of:
    #   "log"    : m = log(S/K)  — log-moneyness (original).
    #              M_LO / M_HI are log-moneyness bounds, e.g. -0.5 / +0.5.
    #              Gives finer resolution near ATM; symmetric around 0.
    #
    #   "linear" : m = S/K       — simple price ratio.
    #              M_LO / M_HI are linear-moneyness bounds, e.g. 0.5 / 1.5.
    #              Equal bin width in price space; intuitive to interpret.
    MONEY_BINS = "linear",

    # Grid bounds — make sure these match the chosen MONEY_BINS convention:
    #   "log"    → e.g.  M_LO=-0.5,  M_HI=0.5   (covers ~60%–165% of K)
    #   "linear" → e.g.  M_LO=0.5,   M_HI=1.5   (covers 50%–150% of K)
    M_LO    = -0.5,
    M_HI    =  0.5,

    # Moneyness bin-spacing — controls how bin edges are distributed:
    #
    #   BIN_ALPHA = 1.0          uniform edges  (original behaviour)
    #   BIN_ALPHA = 0.35 – 0.45  geometric compression toward ATM
    #                            → narrow bins near the money where gamma
    #                              is large; wide bins in the rarely-visited
    #                              tails.  Only applies when MONEY_BINS="log".
    #
    # Rule of thumb:  start with 0.40 and adjust:
    #   lower  (e.g. 0.30) → even more resolution at ATM, fewer bins for tails
    #   higher (e.g. 0.55) → mild compression, closer to uniform
    BIN_ALPHA = 1.0,

    # ── Action grid ──────────────────────────────────────────────────────
    N_ACT = 5,          # number of discrete holdings in [-1, 0]
                        # 5 → H ∈ {-1.0, -0.75, -0.5, -0.25, 0.0}
)

# ── Training / evaluation settings ──────────────────────────────────────────
TRAIN_EPISODES = 30_000
EVAL_EPISODES  = 5_000

# ── No-trade band benchmark settings ────────────────────────────────────────
# band_type : "fixed" — constant half-width = BAND_WIDTH (good for a sweep).
#             "ww"    — Whalley–Wilmott analytical width (calibrated to C_BAND).
# Tip: set band_type="ww" and C_BAND to the same c you use for RL agents so
#      the WW band and the RL Cao objective share the same risk aversion.
BAND_TYPE  = "fixed"   # "fixed" or "ww"
BAND_WIDTH = 0.10      # half-width used when BAND_TYPE="fixed"
C_BAND     = 1.0       # risk-aversion inside the WW formula (band_type="ww")

# ── Moneyness-slice plot settings ───────────────────────────────────────────
TTM_SLICE = 2.0 / 12.0          # time-to-maturity for the slice plot (months/12)
M_GRID    = np.arange(0.8, 1.201, 0.01)   # moneyness S/K range

# ── Agent configurations to train ───────────────────────────────────────────
CONFIGS = [
    dict(agent_class=QHedger,        c=0.0, name="QL_c0",   label="QL   c=0   "),
    dict(agent_class=QHedger,        c=0.7, name="QL_c07",  label="QL   c=0.7 "),
    dict(agent_class=QHedger,        c=1.5, name="QL_c15",  label="QL   c=1.5 "),
    dict(agent_class=QHedger,        c=2.0, name="QL_c20",  label="QL   c=2.0 "),
    dict(agent_class=DoubleQHedger,  c=4.0, name="DQL_c4",  label="DQL  c=4.0 "),
    dict(agent_class=DoubleQHedger,  c=0.7, name="DQL_c07", label="DQL  c=0.7 "),
    dict(agent_class=DoubleQHedger,  c=1.5, name="DQL_c15", label="DQL  c=1.5 "),
    dict(agent_class=DoubleQHedger,  c=2.0, name="DQL_c20", label="DQL  c=2.0 "),
]


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 92)
    print("RL Hedging  —  APL reward, no warm start")
    print("=" * 92)

    # Build moneyness bin edges (must be called before any encode() call)
    edges = build_money_edges(PARAMS)
    alpha = PARAMS.get("BIN_ALPHA", 1.0)
    bins  = PARAMS.get("MONEY_BINS", "log")
    print("\nMoneyness grid: %s  BIN_ALPHA=%.2f  M_LO=%.2f  M_HI=%.2f  N_MONEY=%d"
          % (bins, alpha, PARAMS["M_LO"], PARAMS["M_HI"], PARAMS["N_MONEY"]))
    if abs(alpha - 1.0) > 1e-9:
        print("  Bin edges: " + "  ".join("%.3f" % e for e in edges))

    # Initial option price (used to normalise hedge cost into a percentage)
    V0 = bs_price(PARAMS["S0"], PARAMS["K"], PARAMS["T"],
                  PARAMS["r"], PARAMS["sigma"])
    print("\nInitial option price V_0 = %.4f" % V0)

    # ── BS benchmarks ────────────────────────────────────────────────────
    print("\n[BS delta benchmark]")
    bs_pnl, bs_tc, bs_tr, bs_he = bs_benchmark(PARAMS, EVAL_EPISODES)
    _print_row("BS Delta", bs_pnl, bs_tc, bs_he, V0)

    print("\n[BS band benchmark  (type=%s  width=%.3f  c_band=%.2f)]"
          % (BAND_TYPE, BAND_WIDTH, C_BAND))
    bsb_pnl, bsb_tc, bsb_tr, bsb_he = bs_band_benchmark(
        PARAMS, EVAL_EPISODES,
        band_type=BAND_TYPE, band_width=BAND_WIDTH, c_band=C_BAND)
    _print_row("BS Band", bsb_pnl, bsb_tc, bsb_he, V0)

    # ── Train & evaluate each agent config ───────────────────────────────
    agents, results = [], {}
    for cfg in CONFIGS:
        print("\n[Train %s]" % cfg["label"])
        AgentClass = cfg["agent_class"]
        ag = AgentClass(PARAMS, name=cfg["name"], c=cfg["c"])
        train(ag, PARAMS, n_ep=TRAIN_EPISODES, gamma=1.0)
        pnl, tc, tr, he = evaluate(ag, PARAMS, n_ep=EVAL_EPISODES)
        results[cfg["name"]] = dict(pnl=pnl, tc=tc, trades=tr, he=he,
                                    label=cfg["label"])
        agents.append(ag)
        _print_row(cfg["label"].strip(), pnl, tc, he, V0)

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("%-20s  %10s  %14s  %14s  %14s"
          % ("Strategy",
             "Mean PnL",
             "Avg HC (% V0)",
             "Std HC (% V0)",
             "Mean HE"))
    print("-" * 92)
    print(_table_row("BS Delta",     bs_pnl,  bs_tc,  bs_he,  V0))
    print(_table_row("BS Band", bsb_pnl, bsb_tc, bsb_he, V0))
    for name, res in results.items():
        print(_table_row(res["label"].strip(),
                         res["pnl"], res["tc"], res["he"], V0))
    print("=" * 92)
    print("Note: HC = hedge cost (transaction costs); HE = hedging error")
    print("      HE_t = Portfolio_t − e^{-rτ_t}·V_t, mean taken at t = N (terminal)")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_results(bs_pnl, bs_tc, bs_he,
                 bsb_pnl, bsb_tc, bsb_he,
                 results, agents, CONFIGS, V0)


# ═════════════════════════════════════════════════════════════════════════════
# Table-row helpers
# ═════════════════════════════════════════════════════════════════════════════
def _hedge_cost_pct(tc_array, V0):
    """Hedge cost expressed as % of initial option price."""
    return 100.0 * tc_array / V0


def _terminal_he(he_paths):
    """Terminal (i.e. step N) hedging error per episode."""
    return he_paths[:, -1]


def _print_row(name, pnl, tc, he, V0):
    """Concise per-strategy line shown during the run."""
    hc_pct = _hedge_cost_pct(tc, V0)
    term_he = _terminal_he(he)
    print("  [%s]  PnL=%.4f  HC=%.2f%% (std %.2f%%)  HE=%.4f"
          % (name, pnl.mean(), hc_pct.mean(), hc_pct.std(), term_he.mean()))


def _table_row(name, pnl, tc, he, V0):
    hc_pct = _hedge_cost_pct(tc, V0)
    term_he = _terminal_he(he)
    return ("%-20s  %10.4f  %14.2f  %14.2f  %14.4f"
            % (name[:20], pnl.mean(),
               hc_pct.mean(), hc_pct.std(), term_he.mean()))


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(bs_pnl, bs_tc, bs_he,
                 bsb_pnl, bsb_tc, bsb_he,
                 results, agents, configs, V0):
    """
    Four-panel figure (2 × 2):
      Top-left  : PnL distribution histogram  — all strategies overlaid
      Top-right : Hedge-cost distribution histogram — all strategies overlaid
      Bot-left  : Mean |HE_t| over time
      Bot-right : Hedge ratio vs moneyness at fixed TTM
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("RL Hedging  —  APL reward, no warm start",
                 fontsize=14, fontweight="bold")

    palette = ["tomato", "darkorange", "green", "purple",
               "deeppink", "teal", "brown", "olive"]

    # ── Panel 1 (top-left): PnL distribution ────────────────────────────
    ax = axes[0, 0]
    ax.hist(bs_pnl,  bins=70, alpha=0.55, density=True,
            color="steelblue", label="BS Delta")
    ax.hist(bsb_pnl, bins=70, alpha=0.45, density=True,
            color="grey",      label="BS Band", linestyle="--",
            histtype="step", linewidth=1.6)
    for (name, res), c in zip(results.items(), palette):
        ax.hist(res["pnl"], bins=70, alpha=0.40, density=True,
                color=c, label=res["label"].strip())
    ax.set_title("PnL distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 2 (top-right): Hedge-cost distribution ─────────────────────
    ax = axes[0, 1]
    ax.hist(_hedge_cost_pct(bs_tc,  V0), bins=70, alpha=0.55, density=True,
            color="steelblue", label="BS Delta")
    ax.hist(_hedge_cost_pct(bsb_tc, V0), bins=70, alpha=0.45, density=True,
            color="grey",      label="BS Band", linestyle="--",
            histtype="step", linewidth=1.6)
    for (name, res), c in zip(results.items(), palette):
        ax.hist(_hedge_cost_pct(res["tc"], V0), bins=70, alpha=0.40,
                density=True, color=c, label=res["label"].strip())
    ax.set_title("Hedge-cost distribution  (% of $V_0$)")
    ax.set_xlabel("Hedge cost  (% of $V_0$)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 3 (bot-left): Mean absolute hedging error over time ────────
    ax = axes[1, 0]
    N = PARAMS["N"]
    t_axis = np.arange(N + 1) * (PARAMS["T"] / N)

    ax.plot(t_axis, np.abs(bs_he).mean(axis=0),
            color="steelblue", lw=2.0, label="BS Delta")
    ax.plot(t_axis, np.abs(bsb_he).mean(axis=0),
            color="grey", lw=1.8, linestyle="--", label="BS Band")
    for (name, res), c in zip(results.items(), palette):
        ax.plot(t_axis, np.abs(res["he"]).mean(axis=0),
                color=c, lw=1.4, label=res["label"].strip())
    ax.set_title("Hedging error progression")
    ax.set_xlabel("Time  $t$  (years)")
    ax.set_ylabel(r"Mean $|HE_t|$  =  $|$Portfolio$_t -  e^{-r\tau_t}V_t|$")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 4 (bot-right): Hedge ratio vs moneyness at fixed TTM ───────
    ax = axes[1, 1]
    _plot_hedge_ratio_slice(ax, agents, configs)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nPlot saved to results.png")
    plt.show()
    print("\nDone.")


def _plot_hedge_ratio_slice(ax, agents, configs):
    """
    Mirror of the MATLAB plot:
        policy_RL_mR(mR, TTM, Pos) = getAction(agent, [mR TTM Pos])
        plot blsdelta vs RL at ATM, RL "Selling" (mR+0.1), RL "Buying" (mR-0.1).

    Our agents condition on (τ, S), not on a separate position dimension.
    We replicate the MATLAB curves by querying the RL policy at:
        ATM     : the actual moneyness mR
        Selling : moneyness mR + 0.1   (agent sees a more-ITM state)
        Buying  : moneyness mR − 0.1   (agent sees a more-OTM state)
    A single representative agent (the lowest-TC trained one) is used.
    """
    K     = PARAMS["K"]
    r     = PARAMS["r"]
    sigma = PARAMS["sigma"]
    tau   = TTM_SLICE

    # Pick the trained agent with the lowest mean hedge cost.
    # (Identical selection rule as the old policy-heatmap panel.)
    best_idx = 0
    best_tc  = float("inf")
    # We don't have results here, so re-derive from agents' evaluations:
    # Instead, pick the last QL agent by default — simple and predictable.
    # Override: pick the agent whose label starts with "QL   c=0" if present.
    for i, cfg in enumerate(configs):
        if cfg["name"] == "QL_c0":
            best_idx = i
            break
    best_agent = agents[best_idx]
    best_label = configs[best_idx]["label"].strip()

    # ── BLS delta curve ─────────────────────────────────────────────────
    bls = np.array([bs_delta(mR * K, K, tau, r, sigma) for mR in M_GRID])
    ax.plot(M_GRID, bls, "b-", lw=2.0, label="Theoretical BLS Delta")

    # ── RL Hedge — ATM, Selling, Buying ─────────────────────────────────
    rl_atm     = np.array([rl_hedge_ratio(best_agent, mR * K,        K, tau, PARAMS)
                           for mR in M_GRID])
    rl_selling = np.array([rl_hedge_ratio(best_agent, (mR + 0.1) * K, K, tau, PARAMS)
                           for mR in M_GRID])
    rl_buying  = np.array([rl_hedge_ratio(best_agent, (mR - 0.1) * K, K, tau, PARAMS)
                           for mR in M_GRID])

    ax.plot(M_GRID, rl_atm,     "r-", lw=1.8, label="RL Hedge -- ATM")
    ax.plot(M_GRID, rl_selling, "g-", lw=1.8, label="RL Hedge -- Selling")
    ax.plot(M_GRID, rl_buying,  "m-", lw=1.8, label="RL Hedge -- Buying")

    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Hedge Ratio")
    ax.set_title("RL Hedge vs. BLS Delta for TTM = %.3f  (%s)"
                 % (tau, best_label))
    ax.set_xlim([M_GRID[0], M_GRID[-1]])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    main()
