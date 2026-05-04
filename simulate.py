"""
Hedge simulators and evaluation metrics.

All simulators follow the same accounting structure (tracking the option
writer's P&L):

    1. Start with cash = V0  (collected option premium)
    2. At each step: choose new position, buy/sell stock, pay transaction costs
    3. At maturity: liquidate remaining stock, pay option payoff to holder

The only difference between methods is HOW new_pos is chosen:
    - sim_bs:       rebalance to BS delta every step
    - sim_band:     rebalance only when |pos − delta| > half_width
    - sim_Q:        look up action in the Q-table
    - sim_double_Q: average two Q-tables, then look up action
"""

import numpy as np
from config import N, K, dt, kappa
from black_scholes import V0, bs_delta_vec
from environment import get_state, ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def mets(pnl, tc, nt, label=""):
    """
    Compute evaluation metrics from a P&L vector.

    Returns a dict with:
      mean_pnl    — average outcome across paths
      std_pnl     — hedging risk (P&L standard deviation)
      mean_tc     — average transaction costs paid
      mean_trades — average number of trades
      sharpe      — risk-adjusted return (mean / std)
      CVaR_5      — average P&L in the worst 5% of paths (tail risk)
    """
    return dict(
        label       = label,
        mean_pnl    = float(pnl.mean()),
        std_pnl     = float(pnl.std()),
        mean_tc     = float(tc.mean()),
        mean_trades = float(nt.mean()),
        sharpe      = float(pnl.mean() / (pnl.std() + 1e-10)),
        CVaR_5      = float(np.percentile(pnl, 5)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES DELTA HEDGE (benchmark)
# ══════════════════════════════════════════════════════════════════════════════

def sim_bs(paths):
    """
    Black-Scholes delta hedge — rebalance to exact delta every step.
    This is the standard benchmark. It hedges perfectly but trades a LOT,
    incurring high transaction costs.
    """
    n_paths = paths.shape[0]
    cash = np.full(n_paths, V0, dtype=np.float64)
    pos  = np.zeros(n_paths)
    tc   = np.zeros(n_paths)
    nt   = np.zeros(n_paths, dtype=int)

    for step in range(N):
        S   = paths[:, step]
        tau = (N - step) * dt
        new_pos = bs_delta_vec(S, tau) if step < N - 1 else np.zeros(n_paths)
        trade   = new_pos - pos
        cost    = kappa * np.abs(trade) * S
        cash   -= trade * S + cost
        tc     += cost
        nt     += (np.abs(trade) > 1e-8).astype(int)
        pos     = new_pos

    S_T   = paths[:, -1]
    cash += pos * S_T
    cash -= np.maximum(S_T - K, 0.0)
    return cash, tc, nt


# ══════════════════════════════════════════════════════════════════════════════
# NO-TRADE BAND (analytical baseline)
# ══════════════════════════════════════════════════════════════════════════════

def sim_band(half_width, paths):
    """
    No-trade band policy: only rebalance to delta when |pos − delta| > hw.
    This is a simple analytical improvement over BS — fewer trades, less cost,
    at the expense of slightly worse hedging when the position drifts.
    """
    n_paths = paths.shape[0]
    cash = np.full(n_paths, V0, dtype=np.float64)
    pos  = np.zeros(n_paths)
    tc   = np.zeros(n_paths)
    nt   = np.zeros(n_paths, dtype=int)

    for step in range(N):
        S   = paths[:, step]
        tau = (N - step) * dt
        if step < N - 1:
            delta   = bs_delta_vec(S, tau)
            new_pos = np.where(np.abs(pos - delta) > half_width, delta, pos)
        else:
            new_pos = np.zeros(n_paths)
        trade = new_pos - pos
        cost  = kappa * np.abs(trade) * S
        cash -= trade * S + cost
        tc   += cost
        nt   += (np.abs(trade) > 1e-8).astype(int)
        pos   = new_pos

    S_T   = paths[:, -1]
    cash += pos * S_T
    cash -= np.maximum(S_T - K, 0.0)
    return cash, tc, nt


# ══════════════════════════════════════════════════════════════════════════════
# Q-LEARNING POLICY
# ══════════════════════════════════════════════════════════════════════════════

def sim_Q(Q, paths):
    """
    Run a trained Q-table policy across many paths in parallel.
    At each step the agent picks the action with the highest Q-value.
    """
    n_paths = paths.shape[0]
    cash = np.full(n_paths, V0, dtype=np.float64)
    pos  = np.zeros(n_paths)
    tc   = np.zeros(n_paths)
    nt   = np.zeros(n_paths, dtype=int)

    for step in range(N):
        S = paths[:, step]
        if step == N - 1:
            new_pos = np.zeros(n_paths)           # forced close-out
        else:
            t_idx, m_idx, e_idx, _ = get_state(S, pos, step)
            a_idx   = np.argmax(Q[t_idx, m_idx, e_idx, :], axis=1)
            new_pos = np.clip(pos + ACTIONS[a_idx], 0.0, 1.0)

        trade = new_pos - pos
        cost  = kappa * np.abs(trade) * S
        cash -= trade * S + cost
        tc   += cost
        nt   += (np.abs(trade) > 1e-8).astype(int)
        pos   = new_pos

    S_T   = paths[:, -1]
    cash += pos * S_T
    cash -= np.maximum(S_T - K, 0.0)
    return cash, tc, nt


# ══════════════════════════════════════════════════════════════════════════════
# DOUBLE Q-LEARNING POLICY
# ══════════════════════════════════════════════════════════════════════════════

def sim_double_Q(Q_A, Q_B, paths):
    """Evaluate a Double-Q policy by averaging the two tables."""
    Q_avg = (Q_A + Q_B) / 2.0
    return sim_Q(Q_avg, paths)
