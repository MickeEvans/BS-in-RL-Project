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

Plots produced (1 × 3):
  1. RL Hedge vs BLS Delta        — hedge ratio vs moneyness at fixed TTM
  2. RL vs BLS hedge costs        — count histogram of absolute hedge costs
  3. Hedging-error progression    — mean |HE_t| over time, per strategy

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
from agents import QHedger, DoubleQHedger, train, evaluate, save_agents

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
    MONEY_BINS = "linear",   # "log" or "linear" — linear = uniform S/K bins

    # Grid bounds for linear moneyness (S/K ratio):
    #   0.5–1.5 covers 50%–150% of strike; widen if S drifts far from K.
    M_LO    = 0.5,
    M_HI    = 1.5,

    # BIN_ALPHA = 1.0 → uniform bin spacing (recommended for linear mode)
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
    dict(agent_class=QHedger,       c=0.0, name="QL_c00",  label="QL   c=0.0 "),
    dict(agent_class=QHedger,       c=0.5, name="QL_c05",  label="QL   c=0.5 "),
    dict(agent_class=QHedger,       c=1.5, name="QL_c15",  label="QL   c=1.5 "),
    dict(agent_class=QHedger,       c=2.0, name="QL_c20",  label="QL   c=2.0 "),
    dict(agent_class=DoubleQHedger, c=0.0, name="DQL_c00", label="DQL  c=0.0 "),
    dict(agent_class=DoubleQHedger, c=0.5, name="DQL_c05", label="DQL  c=0.5 "),
    dict(agent_class=DoubleQHedger, c=1.5, name="DQL_c15", label="DQL  c=1.5 "),
    dict(agent_class=DoubleQHedger, c=2.0, name="DQL_c20", label="DQL  c=2.0 "),
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

    # Save trained Q-tables so eval_cf_vs_apl.py can load them directly
    save_agents(agents, CONFIGS)

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
    Three-panel figure (1 x 3):
      [0] RL Hedge vs BLS Delta  -- hedge ratio vs moneyness at fixed TTM
      [1] Hedge-cost counts      -- histogram of absolute hedge costs
      [2] Hedging-error progress -- mean |HE_t| over time
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("RL Hedging  --  APL reward, no warm start",
                 fontsize=14, fontweight="bold")

    palette = ["tomato", "darkorange", "green", "purple",
               "deeppink", "teal", "brown", "olive"]

    # Panel 1: Hedge ratio vs moneyness
    _plot_hedge_ratio_slice(axes[0], agents, configs)

    # Panel 2: Hedge-cost count histogram
    _plot_hedge_cost_hist(axes[1], bs_tc, results)

    # Panel 3: Hedging-error progression
    ax = axes[2]
    N = PARAMS["N"]
    t_axis = np.arange(N + 1) * (PARAMS["T"] / N)
    ax.plot(t_axis, np.abs(bs_he).mean(axis=0),
            color="steelblue", lw=2.2, label="BS Delta")
    ax.plot(t_axis, np.abs(bsb_he).mean(axis=0),
            color="grey", lw=1.8, linestyle="--", label="BS Band")
    for (name, res), c in zip(results.items(), palette):
        ax.plot(t_axis, np.abs(res["he"]).mean(axis=0),
                color=c, lw=1.4, label=res["label"].strip())
    ax.set_title("Hedging Error Progression", fontsize=12)
    ax.set_xlabel("Time  $t$  (years)", fontsize=11)
    ax.set_ylabel(r"Mean $|HE_t|$", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nPlot saved to results.png")
    plt.show()
    print("\nDone.")


def _smooth(arr, window=3):
    """Centred moving-average to soften grid-quantisation steps in RL curves."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def _plot_hedge_ratio_slice(ax, agents, configs):
    """
    Hedge ratio vs moneyness at fixed TTM -- mirrors the MATLAB style.

    Improvements over the earlier version:
      - 400-point M grid so the BLS S-curve is perfectly smooth
      - RL curves smoothed with a moving average to remove grid-step artefacts
      - Heavier lines and clean colours (blue / red / green / purple)
      - Best QL agent selected by lowest mean TC (not hard-coded by name)
    """
    K     = PARAMS["K"]
    r     = PARAMS["r"]
    sigma = PARAMS["sigma"]
    tau   = TTM_SLICE

    m_fine = np.linspace(M_GRID[0], M_GRID[-1], 400)
    win    = max(3, len(m_fine) // 30)   # ~1/30 of the grid width

    # Pick QL_c00 agent (or first QL if not present)
    best_idx = 0
    for i, cfg in enumerate(configs):
        if cfg["name"] == "QL_c00":
            best_idx = i
            break
    best_agent = agents[best_idx]
    best_label = configs[best_idx]["label"].strip()

    # BLS delta -- exact analytical curve
    bls = np.array([bs_delta(mR * K, K, tau, r, sigma) for mR in m_fine])
    ax.plot(m_fine, bls, color="steelblue", lw=2.5,
            label="Theoretical BLS Delta")

    # RL curves -- query policy then smooth to reduce state-grid steps
    rl_atm     = _smooth(np.array([rl_hedge_ratio(best_agent, mR * K,
                          K, tau, PARAMS) for mR in m_fine]), win)
    rl_selling = _smooth(np.array([rl_hedge_ratio(best_agent, (mR + 0.1) * K,
                          K, tau, PARAMS) for mR in m_fine]), win)
    rl_buying  = _smooth(np.array([rl_hedge_ratio(best_agent, (mR - 0.1) * K,
                          K, tau, PARAMS) for mR in m_fine]), win)

    ax.plot(m_fine, rl_atm,     color="tomato",       lw=2.2,
            label="RL Hedge -- ATM")
    ax.plot(m_fine, rl_selling, color="forestgreen",  lw=2.2,
            label="RL Hedge -- Selling")
    ax.plot(m_fine, rl_buying,  color="mediumpurple", lw=2.2,
            label="RL Hedge -- Buying")

    ax.set_xlabel("Moneyness", fontsize=11)
    ax.set_ylabel("Hedge Ratio", fontsize=11)
    ax.set_title("RL Hedge vs. BLS Delta  (TTM = %.3f yr)" % tau, fontsize=12)
    ax.set_xlim([m_fine[0], m_fine[-1]])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)


def _plot_hedge_cost_hist(ax, bs_tc, results):
    """
    Hedge-cost count histogram -- mirrors the MATLAB 'RL Hedge Costs vs
    BLS Hedge Costs' bar chart.

    Shows the best RL agent (lowest mean TC across all configs) vs BS Delta.
    Y-axis is raw episode count, not density, matching the MATLAB style.
    """
    best_name, best_res = min(results.items(), key=lambda x: x[1]["tc"].mean())
    rl_tc  = best_res["tc"]
    bls_tc = bs_tc

    all_tc    = np.concatenate([rl_tc, bls_tc])
    tc_lo     = np.percentile(all_tc, 0.5)
    tc_hi     = np.percentile(all_tc, 99.5)
    bin_edges = np.linspace(tc_lo, tc_hi, 26)   # 25 bars -- matches MATLAB look

    ax.hist(rl_tc,  bins=bin_edges, alpha=0.65, color="salmon",
            label="RL Hedge  (%s)" % best_res["label"].strip(),
            edgecolor="white", linewidth=0.6)
    ax.hist(bls_tc, bins=bin_edges, alpha=0.65, color="steelblue",
            label="Theoretical BLS Delta",
            edgecolor="white", linewidth=0.6)

    ax.set_xlabel("Hedging Costs", fontsize=11)
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.set_title("RL Hedge Costs vs. BLS Hedge Costs", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)


if __name__ == "__main__":
    main()
