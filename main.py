"""
main.py — Run the RL hedging experiment
========================================
All tuneable parameters are defined in PARAMS below.
Trains Q-learning and Double Q-learning agents, benchmarks against
Black-Scholes, and produces diagnostic plots.

Summary table reports:
  - Mean PnL
  - Mean hedge cost (% of initial option price)
  - Std  hedge cost (% of initial option price)
  - Mean terminal hedging error (Portfolio − discounted option value)

Benchmarks:
  - BS Delta : continuous delta hedge (frictionless upper bound)
  - BS Band  : delta hedge with no-trade band (cost-aware benchmark)

Plots produced — each saved as a SEPARATE PNG file:
  1. fig_ql_hedge_ratio.png   — QL  hedge ratio vs moneyness (ATM/Sell/Buy)
  2. fig_dql_hedge_ratio.png  — DQL hedge ratio vs moneyness (ATM/Sell/Buy)
  3. fig_hedge_costs.png      — RL (QL+DQL) vs BLS hedge-cost count histogram
  4. fig_hedging_error.png    — mean |HE_t| over time, per strategy

Usage:
    python main.py                     (interactive — shows plots)
    MPLBACKEND=Agg python main.py      (headless  — saves PNGs only)
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
# band_type : "fixed" — constant half-width = BAND_WIDTH.
#             "ww"    — Whalley–Wilmott analytical width (risk aversion C_BAND).
BAND_TYPE  = "fixed"   # "fixed" or "ww"
BAND_WIDTH = 0.10      # half-width used when BAND_TYPE="fixed"
C_BAND     = 1.0       # risk-aversion inside the WW band formula (band_type="ww")

# ── Moneyness-slice plot settings ───────────────────────────────────────────
TTM_SLICE = 2.0 / 12.0          # time-to-maturity for the slice plot (months/12)
M_GRID    = np.arange(0.8, 1.201, 0.01)   # moneyness S/K range

# ── Agent configurations to train ───────────────────────────────────────────
# One Q-learning and one Double Q-learning agent (no Cao penalty).
CONFIGS = [
    dict(agent_class=QHedger,       name="QL",  label="Q-Learning       "),
    dict(agent_class=DoubleQHedger, name="DQL", label="Double Q-Learning"),
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

    print("\n[BS band benchmark  (type=%s  width=%.3f)]"
          % (BAND_TYPE, BAND_WIDTH))
    bsb_pnl, bsb_tc, bsb_tr, bsb_he = bs_band_benchmark(
        PARAMS, EVAL_EPISODES,
        band_type=BAND_TYPE, band_width=BAND_WIDTH, c_band=C_BAND)
    _print_row("BS Band", bsb_pnl, bsb_tc, bsb_he, V0)

    # ── Train & evaluate each agent config ───────────────────────────────
    agents, results = [], {}
    for cfg in CONFIGS:
        print("\n[Train %s]" % cfg["label"])
        AgentClass = cfg["agent_class"]
        ag = AgentClass(PARAMS, name=cfg["name"])
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
# PLOTTING — each plot is saved as a SEPARATE figure / PNG
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(bs_pnl, bs_tc, bs_he,
                 bsb_pnl, bsb_tc, bsb_he,
                 results, agents, configs, V0):
    """
    Produce four separate figures, each saved as its own PNG so they can be
    downloaded individually:

      fig_ql_hedge_ratio.png   — QL  hedge ratio vs moneyness (ATM/Sell/Buy)
      fig_dql_hedge_ratio.png  — DQL hedge ratio vs moneyness (ATM/Sell/Buy)
      fig_hedge_costs.png      — QL + DQL vs BLS hedge-cost count histogram
      fig_hedging_error.png    — mean |HE_t| over time, all strategies
    """
    # ── Figure 1: QL hedge ratio vs moneyness ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_hedge_ratio_slice(ax, agents, configs, agent_kind="QHedger",
                            title_prefix="Q-Learning")
    fig.tight_layout()
    fig.savefig("fig_ql_hedge_ratio.png", dpi=150)
    print("\nSaved fig_ql_hedge_ratio.png")

    # ── Figure 2: DQL hedge ratio vs moneyness ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_hedge_ratio_slice(ax, agents, configs, agent_kind="DoubleQHedger",
                            title_prefix="Double Q-Learning")
    fig.tight_layout()
    fig.savefig("fig_dql_hedge_ratio.png", dpi=150)
    print("Saved fig_dql_hedge_ratio.png")

    # ── Figure 3: Hedge-cost count histogram (QL + DQL vs BLS) ───────────
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_hedge_cost_hist(ax, bs_tc, results)
    fig.tight_layout()
    fig.savefig("fig_hedge_costs.png", dpi=150)
    print("Saved fig_hedge_costs.png")

    # ── Figure 4: Hedging-error progression ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_hedging_error(ax, bs_he, bsb_he, results)
    fig.tight_layout()
    fig.savefig("fig_hedging_error.png", dpi=150)
    print("Saved fig_hedging_error.png")

    plt.show()
    print("\nDone.")


def _smooth(arr, window=3):
    """Centred moving-average to soften grid-quantisation steps in RL curves."""
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def _find_agent(agents, configs, agent_kind):
    """Return the (agent, label) whose class name matches agent_kind."""
    for ag, cfg in zip(agents, configs):
        if type(ag).__name__ == agent_kind:
            return ag, cfg["label"].strip()
    raise ValueError("No agent of type %s found in CONFIGS" % agent_kind)


def _plot_hedge_ratio_slice(ax, agents, configs, agent_kind, title_prefix):
    """
    Hedge ratio vs moneyness at fixed TTM, for one agent type (QL or DQL).

    Mirrors the MATLAB style:
      - 400-point M grid so the BLS S-curve is smooth
      - RL curves smoothed with a moving average to remove grid-step artefacts
      - Heavy lines, clean colours (blue / red / green / purple)
      - ATM / Selling (mR+0.1) / Buying (mR-0.1) RL curves
    """
    K     = PARAMS["K"]
    r     = PARAMS["r"]
    sigma = PARAMS["sigma"]
    tau   = TTM_SLICE

    m_fine = np.linspace(M_GRID[0], M_GRID[-1], 400)
    win    = max(3, len(m_fine) // 30)

    agent, label = _find_agent(agents, configs, agent_kind)

    # BLS delta — exact analytical S-curve
    bls = np.array([bs_delta(mR * K, K, tau, r, sigma) for mR in m_fine])
    ax.plot(m_fine, bls, color="steelblue", lw=2.5,
            label="Theoretical BLS Delta")

    # RL curves — query policy then smooth
    rl_atm     = _smooth(np.array([rl_hedge_ratio(agent, mR * K,
                          K, tau, PARAMS) for mR in m_fine]), win)
    rl_selling = _smooth(np.array([rl_hedge_ratio(agent, (mR + 0.1) * K,
                          K, tau, PARAMS) for mR in m_fine]), win)
    rl_buying  = _smooth(np.array([rl_hedge_ratio(agent, (mR - 0.1) * K,
                          K, tau, PARAMS) for mR in m_fine]), win)

    ax.plot(m_fine, rl_atm,     color="tomato",       lw=2.2,
            label="RL Hedge -- ATM")
    ax.plot(m_fine, rl_selling, color="forestgreen",  lw=2.2,
            label="RL Hedge -- Selling")
    ax.plot(m_fine, rl_buying,  color="mediumpurple", lw=2.2,
            label="RL Hedge -- Buying")

    ax.set_xlabel("Moneyness", fontsize=11)
    ax.set_ylabel("Hedge Ratio", fontsize=11)
    ax.set_title("%s Hedge vs. BLS Delta  (TTM = %.3f yr)"
                 % (title_prefix, tau), fontsize=12)
    ax.set_xlim([m_fine[0], m_fine[-1]])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)


def _plot_hedge_cost_hist(ax, bs_tc, results):
    """
    Hedge-cost count histogram — mirrors MATLAB's 'RL Hedge Costs vs
    BLS Hedge Costs' chart, now with BOTH RL agents.

    Y-axis = number of trials (raw episode count, not density).
    X-axis = absolute hedge cost.
    Shows QL, DQL and BS Delta overlaid.
    """
    # Identify QL and DQL results by name
    ql_res  = next((r for n, r in results.items() if n == "QL"),  None)
    dql_res = next((r for n, r in results.items() if n == "DQL"), None)

    series = [("Theoretical BLS Delta", bs_tc, "steelblue")]
    if ql_res is not None:
        series.append(("RL Hedge -- QL",  ql_res["tc"],  "salmon"))
    if dql_res is not None:
        series.append(("RL Hedge -- DQL", dql_res["tc"], "mediumseagreen"))

    # Common bin edges across all shown series
    all_tc    = np.concatenate([s[1] for s in series])
    tc_lo     = np.percentile(all_tc, 0.5)
    tc_hi     = np.percentile(all_tc, 99.5)
    bin_edges = np.linspace(tc_lo, tc_hi, 26)

    for label, data, color in series:
        ax.hist(data, bins=bin_edges, alpha=0.55, color=color,
                label=label, edgecolor="white", linewidth=0.6)

    ax.set_xlabel("Hedging Costs", fontsize=11)
    ax.set_ylabel("Number of Trials", fontsize=11)
    ax.set_title("RL Hedge Costs vs. BLS Hedge Costs", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)


def _plot_hedging_error(ax, bs_he, bsb_he, results):
    """Mean |HE_t| over time for all strategies."""
    palette = ["tomato", "mediumseagreen", "darkorange", "purple",
               "deeppink", "teal", "brown", "olive"]
    N = PARAMS["N"]
    t_axis = np.arange(N + 1) * (PARAMS["T"] / N)

    ax.plot(t_axis, np.abs(bs_he).mean(axis=0),
            color="steelblue", lw=2.2, label="BS Delta")
    ax.plot(t_axis, np.abs(bsb_he).mean(axis=0),
            color="grey", lw=1.8, linestyle="--", label="BS Band")
    for (name, res), c in zip(results.items(), palette):
        ax.plot(t_axis, np.abs(res["he"]).mean(axis=0),
                color=c, lw=1.6, label=res["label"].strip())

    ax.set_title("Hedging Error Progression", fontsize=12)
    ax.set_xlabel("Time  $t$  (years)", fontsize=11)
    ax.set_ylabel(r"Mean $|HE_t|$", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)


if __name__ == "__main__":
    main()
