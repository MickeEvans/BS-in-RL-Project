"""
eval_cf_vs_apl.py — Compare APL vs Cash-Flow evaluation on trained agents
==========================================================================
Loads Q-tables saved by main.py (no retraining) and evaluates each agent
under two accounting formulations on fresh simulated paths:

  1. APL formulation  (thesis Eq. 8)  — same signal used during training
  2. CF  formulation  (thesis Eq. 10) — realised cash flows + terminal payoff

Prerequisite
------------
Run main.py first.  It saves one .npz file per agent to saved_agents/.
This script raises FileNotFoundError with a clear message if any file is
missing so you know exactly what to do.

PARAMS and CONFIGS here must match main.py exactly — they define which
files to look for and how to reconstruct the Q-tables.

Metrics reported for both formulations, all agents + BS benchmarks:
  Mean PnL | Std PnL | Mean TC (% V0) | Trades/ep

Usage:
    python main.py              # train and save agents first
    python eval_cf_vs_apl.py   # then evaluate
"""

import math
import numpy as np
import warnings

from environment import (
    bs_price, bs_delta,
    encode, a2h,
    bs_benchmark, bs_band_benchmark,
    build_money_edges,
)
from agents import QHedger, DoubleQHedger, evaluate, load_agents

warnings.filterwarnings("ignore")
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETERS — must match main.py exactly
# ═════════════════════════════════════════════════════════════════════════════
PARAMS = dict(
    S0    = 100.0,
    K     = 100.0,
    T     = 1.0,
    N     = 252,
    sigma = 0.20,
    r     = 0.0,
    kappa = 0.01,
    N_TIME  = 5,
    N_MONEY = 15,
    MONEY_BINS = "linear",
    BIN_ALPHA  = 1.0,
    M_LO    = 0.5,
    M_HI    = 1.5,
    N_ACT   = 5,
)

EVAL_EPISODES = 5_000

# No-trade band settings — must match main.py
BAND_TYPE  = "fixed"
BAND_WIDTH = 0.10
C_BAND     = 1.0

# CONFIGS — must match main.py exactly (names must line up with saved files)
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
# CASH-FLOW EPISODE RUNNER  (thesis Eq. 10)
# ═════════════════════════════════════════════════════════════════════════════

def run_episode_cf(agent, params):
    """
    Evaluate one episode under the Cash-Flow formulation (thesis Eq. 10).

    The agent's greedy policy is followed (eps=0, no learning).
    No option pricing model is consulted mid-episode — only the terminal
    payoff phi(S_n) = max(S_n - K, 0) is model-dependent.

    CF per-step reward (short-stock convention):
        R_{i+1} = S_{i+1}*(H_i - H_{i+1}) - kappa*S_{i+1}*|H_{i+1} - H_i|

    Initial cash flow  : -S_0*H_0 - kappa*|S_0*H_0|
    Terminal cash flow : S_n*H_n  - kappa*|S_n*H_n| - phi(S_n)

    Returns (total_cf, total_tc, n_trades).
    """
    S0    = params["S0"]
    K     = params["K"]
    T     = params["T"]
    N     = params["N"]
    sigma = params["sigma"]
    r     = params["r"]
    kappa = params["kappa"]
    N_ACT = params["N_ACT"]
    dt    = T / N

    S, tau = S0, T

    a        = agent.act(tau, S, params, eps=0.0)
    H        = a2h(a, N_ACT)
    tc_init  = kappa * abs(S * H)
    total_cf = -(S * H) - tc_init
    total_tc = tc_init
    n_trades = 1 if abs(H) > 1e-10 else 0

    for step in range(N):
        z       = np.random.randn()
        S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                               + sigma * math.sqrt(dt) * z)
        tau_new = max(T - (step + 1) * dt, 0.0)
        done    = (step == N - 1)

        if not done:
            a_new  = agent.act(tau_new, S_new, params, eps=0.0)
            H_new  = a2h(a_new, N_ACT)
            tc_now = kappa * abs(S_new * (H_new - H))
            if abs(H_new - H) > 1e-10:
                n_trades += 1
            R = S_new * (H - H_new) - tc_now
        else:
            H_new  = H
            tc_now = kappa * abs(S_new * H)
            if abs(H) > 1e-10:
                n_trades += 1
            payoff = bs_price(S_new, K, 0.0, r, sigma)
            R = S_new * H - tc_now - payoff

        total_cf += R
        total_tc += tc_now
        S, tau, H = S_new, tau_new, H_new

    return total_cf, total_tc, n_trades


def evaluate_cf(agent, params, n_ep=5000):
    """Run n_ep greedy episodes under the CF formulation."""
    cfs, tcs, trades = [], [], []
    for _ in range(n_ep):
        cf, tc, nt = run_episode_cf(agent, params)
        cfs.append(cf)
        tcs.append(tc)
        trades.append(nt)
    return np.array(cfs), np.array(tcs), np.array(trades)


# ═════════════════════════════════════════════════════════════════════════════
# CF BENCHMARKS
# ═════════════════════════════════════════════════════════════════════════════

def _cf_step(S, H, S_new, H_new, kappa, done, K, r, sigma):
    """Shared per-step CF accounting. Returns (R, tc_now, traded)."""
    if not done:
        tc_now = kappa * abs(S_new * (H_new - H))
        traded = abs(H_new - H) > 1e-10
        R      = S_new * (H - H_new) - tc_now
    else:
        tc_now = kappa * abs(S_new * H)
        traded = abs(H) > 1e-10
        payoff = bs_price(S_new, K, 0.0, r, sigma)
        R      = S_new * H - tc_now - payoff
    return R, tc_now, traded


def bs_benchmark_cf(params, n_ep=5000):
    """BS continuous delta hedge under the CF formulation."""
    S0    = params["S0"]
    K     = params["K"]
    T     = params["T"]
    N     = params["N"]
    sigma = params["sigma"]
    r     = params["r"]
    kappa = params["kappa"]
    dt    = T / N

    cfs, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau   = S0, T
        H        = -bs_delta(S, K, tau, r, sigma)
        tc_init  = kappa * abs(S * H)
        total_cf = -(S * H) - tc_init
        total_tc = tc_init
        nt       = 1 if abs(H) > 1e-10 else 0

        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z)
            tau_new = max(T - (step + 1) * dt, 0.0)
            done    = (step == N - 1)
            H_new   = H if done else -bs_delta(S_new, K, tau_new, r, sigma)

            R, tc_now, traded = _cf_step(S, H, S_new, H_new, kappa, done,
                                         K, r, sigma)
            if traded:
                nt += 1
            total_cf += R
            total_tc += tc_now
            S, tau, H = S_new, tau_new, H_new

        cfs.append(total_cf)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(cfs), np.array(tcs), np.array(trades)


def bs_band_benchmark_cf(params, n_ep=5000,
                         band_type="fixed", band_width=0.10, c_band=1.0):
    """BS delta hedge with no-trade band under the CF formulation."""
    S0    = params["S0"]
    K     = params["K"]
    T     = params["T"]
    N     = params["N"]
    sigma = params["sigma"]
    r     = params["r"]
    kappa = params["kappa"]
    dt    = T / N

    _inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    def _ww_half_width(S_t, tau_t):
        if tau_t < 1e-10:
            return 0.0
        s_sqrt = sigma * math.sqrt(tau_t)
        d1     = (math.log(S_t / K) + (r + 0.5 * sigma ** 2) * tau_t) / s_sqrt
        phi_d1 = _inv_sqrt_2pi * math.exp(-0.5 * d1 * d1)
        gamma  = phi_d1 / (S_t * s_sqrt)
        return ((3.0 * kappa * (gamma * S_t) ** 2) / (2.0 * c_band)) ** (1.0 / 3.0)

    cfs, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau   = S0, T
        H        = -bs_delta(S, K, tau, r, sigma)
        tc_init  = kappa * abs(S * H)
        total_cf = -(S * H) - tc_init
        total_tc = tc_init
        nt       = 1 if abs(H) > 1e-10 else 0

        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z)
            tau_new = max(T - (step + 1) * dt, 0.0)
            done    = (step == N - 1)

            if not done:
                target = -bs_delta(S_new, K, tau_new, r, sigma)
                h = (_ww_half_width(S_new, tau_new)
                     if band_type == "ww" else band_width)
                H_new = target if abs(H - target) > h else H
            else:
                H_new = H

            R, tc_now, traded = _cf_step(S, H, S_new, H_new, kappa, done,
                                         K, r, sigma)
            if traded:
                nt += 1
            total_cf += R
            total_tc += tc_now
            S, tau, H = S_new, tau_new, H_new

        cfs.append(total_cf)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(cfs), np.array(tcs), np.array(trades)


# ═════════════════════════════════════════════════════════════════════════════
# PRINT TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_table(rows, V0):
    """
    rows : list of
        (label, apl_pnl, apl_tc, apl_tr, cf_pnl, cf_tc, cf_tr)
    TC shown as % of V0.
    """
    col = 13
    hdr = ("%-20s  %*s %*s %*s %*s   %*s %*s %*s %*s"
           % ("Strategy",
              col, "APL Mean", col, "APL Std", col, "APL TC%", col, "APL Tr",
              col, "CF Mean",  col, "CF Std",  col, "CF TC%",  col, "CF Tr"))
    sep = "-" * len(hdr)

    print("\n" + sep)
    print("  APL = Accounting P&L (thesis Eq. 8)  |  "
          "CF = Cash-Flow (thesis Eq. 10)")
    print("  TC shown as %% of initial option price V0 = %.4f" % V0)
    print(sep)
    print(hdr)
    print(sep)

    for label, ap, at, atr, cp, ct, ctr in rows:
        print("%-20s  %*.4f %*.4f %*.2f %*.1f   %*.4f %*.4f %*.2f %*.1f"
              % (label[:20],
                 col, ap.mean(), col, ap.std(),
                 col, 100.0 * at.mean() / V0, col, atr.mean(),
                 col, cp.mean(), col, cp.std(),
                 col, 100.0 * ct.mean() / V0, col, ctr.mean()))
    print(sep + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("APL vs CF evaluation — using agents saved by main.py")
    print("=" * 78)

    # Build moneyness edges (must match what main.py used)
    build_money_edges(PARAMS)

    V0 = bs_price(PARAMS["S0"], PARAMS["K"], PARAMS["T"],
                  PARAMS["r"], PARAMS["sigma"])
    print("\nInitial option price V0 = %.4f" % V0)

    # ── Load trained agents from disk (no retraining) ─────────────────────
    print("\n[Loading agents from saved_agents/]")
    agents = load_agents(CONFIGS, PARAMS)

    # ── BS benchmarks ─────────────────────────────────────────────────────
    print("\n[BS Delta — APL]")
    bs_apl, bs_apl_tc, bs_apl_tr, _ = bs_benchmark(PARAMS, EVAL_EPISODES)

    print("[BS Band  — APL  (type=%s  width=%.3f)]" % (BAND_TYPE, BAND_WIDTH))
    bsb_apl, bsb_apl_tc, bsb_apl_tr, _ = bs_band_benchmark(
        PARAMS, EVAL_EPISODES,
        band_type=BAND_TYPE, band_width=BAND_WIDTH, c_band=C_BAND)

    print("[BS Delta — CF]")
    bs_cf, bs_cf_tc, bs_cf_tr = bs_benchmark_cf(PARAMS, EVAL_EPISODES)

    print("[BS Band  — CF   (type=%s  width=%.3f)]" % (BAND_TYPE, BAND_WIDTH))
    bsb_cf, bsb_cf_tc, bsb_cf_tr = bs_band_benchmark_cf(
        PARAMS, EVAL_EPISODES,
        band_type=BAND_TYPE, band_width=BAND_WIDTH, c_band=C_BAND)

    # ── Evaluate loaded agents under both formulations ────────────────────
    print("\n[Evaluating agents — APL and CF …]")
    rows = [
        ("BS Delta",
         bs_apl, bs_apl_tc, bs_apl_tr, bs_cf, bs_cf_tc, bs_cf_tr),
        ("BS Band",
         bsb_apl, bsb_apl_tc, bsb_apl_tr, bsb_cf, bsb_cf_tc, bsb_cf_tr),
    ]

    for ag, cfg in zip(agents, CONFIGS):
        # APL — returns (pnls, tcs, trades, he_paths)
        apl_pnl, apl_tc, apl_tr, _ = evaluate(ag, PARAMS, n_ep=EVAL_EPISODES)

        # CF — same agent, different accounting
        cf_pnl, cf_tc, cf_tr = evaluate_cf(ag, PARAMS, n_ep=EVAL_EPISODES)

        rows.append((cfg["label"].strip(),
                     apl_pnl, apl_tc, apl_tr,
                     cf_pnl,  cf_tc,  cf_tr))

        print("  %-12s  APL mean=%7.4f  std=%7.4f  |  "
              "CF mean=%7.4f  std=%7.4f"
              % (cfg["label"].strip(),
                 apl_pnl.mean(), apl_pnl.std(),
                 cf_pnl.mean(),  cf_pnl.std()))

    print_table(rows, V0)


if __name__ == "__main__":
    main()
