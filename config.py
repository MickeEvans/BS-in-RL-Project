"""
Configuration — all market parameters and hyperparameters live here.
Tweak these values and re-run main.py to experiment.
"""

# ── Market Parameters ────────────────────────────────────────────────────────
S0    = 100.0     # initial stock price
K     = 100.0     # strike price
T     = 1.0       # time to maturity (years)
N     = 252       # number of daily rebalancing steps
sigma = 0.20      # annualised volatility
r     = 0.0       # risk-free rate (set to 0 for simplicity)
kappa = 0.01      # proportional transaction cost (1% of notional traded)

# Derived
dt = T / N        # time step size


# ── Training Hyperparameters ─────────────────────────────────────────────────
# These control the Q-learning training loop. Override them in main.py
# or pass them directly to train() / train_double().

LAMBDA_SWEEP   = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]  # risk-aversion values to try
BAND_SWEEP     = [0.05, 0.10, 0.15, 0.20, 0.25]               # no-trade band half-widths

EPISODES_SWEEP = 5_000    # episodes per λ in the sweep
EPISODES_EXT   = 15_000    # extra fine-tuning episodes for the best agent
N_PARALLEL     = 64        # paths per training batch

EVAL_PATHS     = 5_000     # paths for in-training evaluation
OOS_PATHS      = 20_000    # paths for final out-of-sample test

SEED_TRAIN     = 42        # base random seed for training
SEED_EVAL      = 77_777    # seed for evaluation paths
SEED_OOS       = 99_999    # seed for out-of-sample paths
