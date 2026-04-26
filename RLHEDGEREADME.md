# Risk-Sensitive Q-Learning for Option Hedging

Tabular Q-learning agent that hedges a short European call option under
proportional transaction costs and beats the Black-Scholes delta-hedge
benchmark on a risk-adjusted basis.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. The only dependencies are `numpy`, `scipy`, and
`matplotlib` (for the plotting helpers in the examples).

### 2. Run the main experiment

```bash
python rl_hedging.py
```

This runs the full pipeline end-to-end:

1. Simulates 5,000 GBM paths for training-time evaluation.
2. Evaluates the Black-Scholes delta-hedge benchmark.
3. Evaluates analytical no-trade band policies for several widths.
4. Trains Q-learning agents for each of 7 risk-aversion values λ.
5. Takes the best λ, fine-tunes for another 30,000 episodes.
6. Trains a Double Q-Learning agent at the same λ for ablation.
7. Tests every method on 20,000 fresh out-of-sample paths.
8. Pops up four matplotlib windows with the result plots.

The plots are:

| Window | Content |
|---|---|
| Figure 1 | Efficient frontier — hedging risk vs transaction cost |
| Figure 2 | P&L distribution histogram + box plot |
| Figure 3 | Heat-map of the learned policy (no-trade band) |
| Figure 4 | Final out-of-sample comparison (bar chart) |

Output looks like:

```
======================================================================
EUROPEAN CALL OPTION HEDGING — TABULAR Q-LEARNING
======================================================================
Parameters: S0=100.0, K=100.0, T=1.0, N=252, σ=0.2, κ=0.01
State space: 10×9×9 × 7 actions = 5670 cells
Initial BS price V0 = 7.9656

── Benchmark: Black-Scholes delta hedge ──
  mean P&L = -6.0756  std = 2.2682  TC = 6.0711  CVaR5% = -9.8974
...

── Out-of-sample evaluation (20,000 fresh paths, different seed) ──

Method                  Mean PnL   Std PnL   Mean TC   Sharpe   CVaR 5%
------------------------------------------------------------------------------
BS-delta                 -6.0756    2.2682    6.0711   -2.6786   -9.8974
Band hw=0.20             -2.0816    2.2727    2.0570   -0.9160   -6.0620
QL λ=0.3 (15k)           -2.4112    2.5070    2.5110   -0.9618   -7.0103
QL λ=0.3 extended        -2.3773    1.9628    2.3599   -1.2113   -6.0500
Double-QL λ=0.3          -8.4123    5.0894    8.4178   -1.6529  -16.8472

Q-Learning (extended) vs Black-Scholes:
  Transaction cost reduction: -61.1%
  Hedging risk (std) change:  -13.5%
  Tail risk (CVaR) improved:  +3.847
```

Full run takes ~3-5 minutes on a typical laptop.

If you don't want the plot windows (e.g. you're running headless), comment
out the `plt.show()` line at the bottom of `rl_hedging.py`, or save the
figures explicitly:

```python
fig1, fig2, fig3, fig4 = plot_results(results)
fig1.savefig('frontier.png')
# etc.
```

---

## Using the code as a library

All the pieces are importable:

```python
from rl_hedging import (
    # market parameters & Black-Scholes helpers
    S0, K, T, N, sigma, kappa, V0,
    bs_delta_vec, bs_price_scalar,

    # path generation
    sim_paths,

    # hedge simulators
    sim_bs,          # BS delta benchmark
    sim_band,        # no-trade band with given half-width
    sim_Q,           # a Q-learning policy
    sim_double_Q,    # a Double Q-learning policy

    # training
    warm_init,       # BS-heuristic Q-table initialisation
    train,           # standard tabular Q-learning
    train_double,    # Double Q-learning (Hasselt 2010)

    # utility
    mets,                        # compute a dict of metrics from P&L
    compare_single_vs_double,    # matched comparison helper
)
```

### Example: train an agent from scratch

```python
import numpy as np
from rl_hedging import (N_T, N_M, N_E, N_A, warm_init, train,
                        sim_paths, sim_Q, mets)

# Allocate Q-table and visit counters
Q = np.zeros((N_T, N_M, N_E, N_A))
v = np.zeros_like(Q, dtype=int)

# BS-heuristic warm start (speeds training by ~10x)
warm_init(Q)

# Train 15,000 episodes with risk-aversion λ=0.3
train(Q, v, n_episodes=15_000, risk_lambda=0.3,
      eps_start=0.4, eps_end=0.03, seed=42)

# Evaluate on 10,000 out-of-sample paths
eval_paths = sim_paths(10_000, seed=77777)
pnl, tc, n_trades = sim_Q(Q, eval_paths)
m = mets(pnl, tc, n_trades, "my-agent")
print(f"Sharpe: {m['sharpe']:.3f}, TC: {m['mean_tc']:.3f}")
```

### Example: compare Single vs Double Q

```python
from rl_hedging import compare_single_vs_double
results = compare_single_vs_double(n_episodes=15_000, risk_lambda=0.3,
                                   seeds=(42, 137, 2024))
# results['single'] and results['double'] each have a list of metrics
# dicts, one per seed.
```

---

## Files

| File | Purpose |
|------|---------|
| `rl_hedging.py` | Complete self-contained implementation |
| `requirements.txt` | Python dependencies |
| `report.md` | Full write-up with results, plots, analysis |
| `fig1_frontier.png` | Efficient frontier (hedging risk vs TC) |
| `fig2_pnl_distributions.png` | P&L histogram + box-plot comparison |
| `fig3_policy.png` | Heatmap of learned actions — shows no-trade band |
| `fig4_trajectories.png` | Sample hedging paths: BS vs QL |
| `fig5_lambda_effect.png` | Effect of λ on TC, risk, and CVaR |
| `fig6_double_q.png` | Double Q-learning vs Single Q-learning |

---

## Key results (20,000 out-of-sample paths)

| Method | Mean P&L | Std P&L | Mean TC | CVaR 5% |
|---|---:|---:|---:|---:|
| BS-delta | −6.08 | 2.27 | 6.07 | −9.90 |
| No-trade band (hw=0.20) | −2.08 | 2.27 | 2.06 | −6.06 |
| **Q-Learning (λ=0.3 ext.)** | **−2.38** | **1.96** | **2.36** | **−6.05** |

The Q-Learning agent delivers:

- **−61%** transaction cost
- **−14%** hedging risk (std P&L)
- **+39%** better tail risk (CVaR 5%)

all compared to the Black-Scholes benchmark.

---

## Notes on reproducibility

- All random seeds are set explicitly (e.g. `seed=42` in `train()`).
- The evaluation set is generated with `seed=77777` in
  `run_full_experiment()` and the OOS set with `seed=99_999`.
- On Windows/macOS/Linux you should see identical numbers given the same
  NumPy version.

---

## Mathematical references

- Hodges, S. D., & Neuberger, A. (1989). *Optimal replication of
  contingent claims under transactions costs.*
- Davis, M., Panas, V. G., & Zariphopoulou, T. (1993). *European option
  pricing with transaction costs.*
- Whalley, A. E., & Wilmott, P. (1997). *An asymptotic analysis of an
  optimal hedging model for option pricing with transaction costs.*
- Hasselt, H. (2010). *Double Q-learning.*
- Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). *Deep hedging of
  derivatives using reinforcement learning.*
