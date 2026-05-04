# Risk-Sensitive Q-Learning for Option Hedging

Tabular Q-learning agent that hedges a short European call option under proportional transaction costs. The agent discovers a no-trade band policy from scratch and beats the Black-Scholes delta-hedge benchmark on a risk-adjusted basis.

## Key results (20,000 out-of-sample paths)

| Method | Mean P&L | Std P&L | Mean TC | CVaR 5% |
|---|---:|---:|---:|---:|
| BS-delta hedge | −6.08 | 2.27 | 6.07 | −9.90 |
| No-trade band (hw=0.20) | −2.08 | 2.27 | 2.06 | −6.06 |
| **Q-Learning (λ=0.3, extended)** | **−2.38** | **1.96** | **2.36** | **−6.05** |

Compared to Black-Scholes the Q-learning agent achieves **−61% transaction cost**, **−14% hedging risk**, and **+39% better tail risk**.

---

## Quick start

### 1. Install dependencies

```bash
pip install numpy scipy matplotlib
```

Requires Python 3.9+.

### 2. Run the experiment

```bash
python main.py
```

This runs the full pipeline end-to-end 3:

1. Evaluates the Black-Scholes delta-hedge benchmark
2. Sweeps analytical no-trade band widths
3. Trains Q-learning agents across 7 risk-aversion values λ
4. Fine-tunes the best agent for another 30,000 episodes
5. Trains a Double Q-Learning agent as ablation
6. Tests all methods on 20,000 fresh out-of-sample paths
7. Saves four result plots as PNG files

---

## Project structure

```
├── main.py            # Entry point — runs the full experiment
├── config.py          # All tuneable parameters (edit this file)
├── environment.py     # State space, action space, discretisation, path generation
├── agent.py           # Q-learning training (single & double), warm start
├── simulate.py        # Hedge simulators (BS, band, Q-policy) and metrics
├── black_scholes.py   # Black-Scholes delta and pricing formulas
└── plotting.py        # Result visualisation (4 figures)
```

### What each file does

**`config.py`** — The only file you need to edit to experiment. Contains all market parameters (S₀, K, σ, κ, …), training hyperparameters (episode counts, λ-sweep values), and random seeds.

**`environment.py`** — Defines the RL problem. The state has three dimensions:

- **Time buckets** — finer near expiry where delta changes fastest
- **Moneyness S/K** — finer near ATM where gamma is largest
- **Position error (pos − δ_BS)** — the key design choice that lets the agent learn a no-trade band

The action space is 7 discrete position changes: {−0.20, −0.08, −0.025, 0, +0.025, +0.08, +0.20}. Total state-action space: 10 × 9 × 9 × 7 = 5,670 cells.

**`agent.py`** — Training logic. Contains the reward function (transaction cost penalty + Whalley-Wilmott variance penalty), the Bellman Q-update, and the BS-heuristic warm start that speeds convergence by ~10×. Also includes the Double Q-learning variant.

**`simulate.py`** — Evaluates policies by simulating the option writer's P&L across thousands of GBM paths. All three simulators (BS-delta, no-trade band, Q-policy) share the same accounting: collect premium → trade at each step → pay costs → settle payoff at expiry.

**`black_scholes.py`** — Vectorised BS delta and scalar BS pricing. Used throughout by the environment, simulators, and agent.

**`plotting.py`** — Generates four figures: efficient frontier, P&L distributions, learned policy heatmap, and a bar-chart comparison.

---

## How to tweak parameters

Open `config.py` and change whatever you like:

```python
# Try a different stock / strike setup
S0    = 110.0
K     = 100.0

# Increase transaction costs
kappa = 0.02

# Train longer
EPISODES_SWEEP = 30_000
EPISODES_EXT   = 50_000

# Test different risk-aversion values
LAMBDA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
```

Then re-run `python main.py`.

For deeper changes to the state/action discretisation (bucket edges, action sizes), edit `environment.py` directly — see the `T_STEPS`, `M_EDGES`, `E_EDGES`, and `ACTIONS` arrays near the top.

---

## Using the code as a library

All pieces are importable:

```python
import numpy as np
from environment import N_T, N_M, N_E, N_A, sim_paths
from agent import warm_init, train
from simulate import sim_Q, mets

# Allocate Q-table and visit counters
Q = np.zeros((N_T, N_M, N_E, N_A))
v = np.zeros_like(Q, dtype=int)

# BS-heuristic warm start
warm_init(Q)

# Train 15,000 episodes with λ=0.3
train(Q, v, n_episodes=15_000, risk_lambda=0.3,
      eps_start=0.4, eps_end=0.03, seed=42)

# Evaluate on 10,000 out-of-sample paths
paths = sim_paths(10_000, seed=77777)
pnl, tc, n_trades = sim_Q(Q, paths)
m = mets(pnl, tc, n_trades, "my-agent")
print(f"Sharpe: {m['sharpe']:.3f}, TC: {m['mean_tc']:.3f}")
```

---

## Output plots

| Figure | Content |
|---|---|
| `frontier.png` | Efficient frontier — hedging risk vs transaction cost |
| `pnl_distributions.png` | P&L histogram + box plot for all methods |
| `policy_heatmap.png` | Learned actions across moneyness × position error (shows the discovered no-trade band) |
| `oos_comparison.png` | Bar charts comparing all methods on std, TC, and CVaR |

---

## Reproducibility

All random seeds are set explicitly. The evaluation set uses `seed=77777` and the out-of-sample set uses `seed=99999`. You should get identical numbers given the same NumPy version across platforms.

---

## References

- Hodges, S. D. & Neuberger, A. (1989). *Optimal replication of contingent claims under transactions costs.*
- Davis, M., Panas, V. G. & Zariphopoulou, T. (1993). *European option pricing with transaction costs.*
- Whalley, A. E. & Wilmott, P. (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs.*
- Hasselt, H. (2010). *Double Q-learning.*
- Cao, J., Chen, J., Hull, J. & Poulos, Z. (2021). *Deep hedging of derivatives using reinforcement learning.*