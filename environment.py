"""
Environment — state space, action space, discretisation, and path generation.

This is the core RL setup. It defines:
  - STATE:   (time_bucket, moneyness_bucket, position_error_bucket)
  - ACTIONS: 7 discrete position changes
  - How continuous market variables are mapped to discrete indices
  - GBM path simulation

Design rationale
----------------
The state space has three dimensions chosen for financial meaning:

1. Time buckets — finer near expiry where delta changes fastest (high gamma).
   Near expiry the option's delta swings wildly with stock price, so we need
   more resolution there.

2. Moneyness S/K — finer near ATM (S/K ≈ 1.0) where the option value
   curves the most (high gamma).

3. Position error (pos − BS_delta) — THE KEY INSIGHT. Instead of tracking
   the raw position, we track how far we are from the Black-Scholes target.
   This collapses the hedge decision to one dimension: small error → do
   nothing; large error → trade to reduce it. This is what lets the agent
   discover a no-trade band.

Total: 10 × 9 × 9 = 810 states × 7 actions = 5,670 Q-values.
Tiny enough for a table — no neural network needed.
"""

import numpy as np
from config import S0, K, T, N, dt, sigma, r
from black_scholes import bs_delta_vec


# ══════════════════════════════════════════════════════════════════════════════
# STATE DISCRETISATION
# ══════════════════════════════════════════════════════════════════════════════

# Time buckets — edges in units of "steps remaining until expiry"
# Finer near expiry (left) where gamma is high
T_STEPS = np.array([0, 5, 15, 30, 55, 90, 130, 175, 215, 240, 252])
N_T     = len(T_STEPS) - 1     # 10 buckets

# Moneyness S/K buckets — finer near ATM (1.0)
M_EDGES = np.array([0.70, 0.82, 0.91, 0.96, 0.99, 1.01, 1.04, 1.09, 1.18, 1.60])
N_M     = len(M_EDGES) - 1     # 9 buckets

# Position-error buckets — error = (current position − BS delta)
# Finer near zero where the no-trade band boundary matters most
E_EDGES = np.array([-1.0, -0.12, -0.07, -0.03, -0.01, 0.01, 0.03, 0.07, 0.12, 1.0])
N_E     = len(E_EDGES) - 1     # 9 buckets


# ══════════════════════════════════════════════════════════════════════════════
# ACTION SPACE
# ══════════════════════════════════════════════════════════════════════════════
# Discrete position changes. Includes:
#   ±0.20  — large snap-backs after big price moves
#   ±0.08  — medium corrections
#   ±0.025 — fine adjustments
#    0.0   — do nothing (CRUCIAL — enables the no-trade band)

ACTIONS = np.array([-0.20, -0.08, -0.025, 0.0, 0.025, 0.08, 0.20])
N_A     = len(ACTIONS)


# ══════════════════════════════════════════════════════════════════════════════
# STATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def get_state(S, pos, step):
    """
    Map continuous (stock_price, position, time_step) → discrete state indices.

    Parameters
    ----------
    S    : np.ndarray — stock prices (one per path)
    pos  : np.ndarray — current hedge positions (one per path)
    step : int        — current time step (0 to N-1)

    Returns
    -------
    t_idx : int        — time bucket index
    m_idx : np.ndarray — moneyness bucket indices
    e_idx : np.ndarray — position-error bucket indices
    delta : np.ndarray — BS delta values (used later in the reward)
    """
    tau       = (N - step) * dt
    delta     = bs_delta_vec(S, tau)
    steps_rem = N - step

    t_idx = int(np.clip(np.searchsorted(T_STEPS[1:-1], steps_rem, 'right'), 0, N_T - 1))
    m_idx = np.clip(np.searchsorted(M_EDGES[1:-1], S / K, 'right'), 0, N_M - 1)
    e_idx = np.clip(np.searchsorted(E_EDGES[1:-1], pos - delta, 'right'), 0, N_E - 1)

    return t_idx, m_idx, e_idx, delta


# ══════════════════════════════════════════════════════════════════════════════
# PATH GENERATION (Geometric Brownian Motion)
# ══════════════════════════════════════════════════════════════════════════════

def sim_paths(n_paths, seed=None):
    """
    Simulate n_paths independent GBM stock-price paths.

    Each path has N+1 points (days 0 through N).
    Returns shape (n_paths, N+1).
    """
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n_paths, N))
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    paths = np.empty((n_paths, N + 1))
    paths[:, 0] = S0
    for t in range(N):
        paths[:, t + 1] = paths[:, t] * np.exp(log_ret[:, t])
    return paths
