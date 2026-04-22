# Risk-Sensitive Q-Learning for Option Hedging

Tabular Q-learning that hedges a short European call under proportional transaction costs, and beats Black-Scholes delta rebalancing on the risk-adjusted objective `Y = E[cost] + 1.5·Std[cost]`.

Implements the Cao-Chen-Hull-Poulos (2021) dual-Q formulation plus a post-hoc no-trade bias at evaluation time. The learned policy is a discrete no-trade band around BS delta — exactly the theoretical structure from Davis-Panas-Zariphopoulou (1993).

## Headline Result

| Strategy | Mean $ | Std $ | CVaR-95% $ | **Y = Mean + 1.5·Std** | Trades/year |
|---|---|---|---|---|---|
| No hedge | 0.64 | 6.06 | 16.56 | 9.72 | 0 |
| BS daily | 5.54 | 2.07 | 9.93 | 8.64 | 245 |
| BS weekly | 2.81 | 1.59 | 6.70 | 5.19 | 49 |
| BS biweekly | 2.18 | 1.78 | 6.59 | 4.84 | 25 |
| **RL + no-trade bias** | **1.26** | **2.26** | **6.60** | **4.65** | **10** |

European call option, K=100, T=1 year, σ=0.2, κ=1% proportional transaction cost, 10,000 Monte Carlo eval paths.

## Setup

All files should live in one folder. In a terminal from that folder:

```
pip install -r requirements.txt
```

Dependencies are pure NumPy/SciPy/matplotlib — no PyTorch, no GPU required. Runs on any laptop.

## How to Run

**Start here:** open `final_run.ipynb` in VS Code (with the Jupyter extension installed) and run cells top-to-bottom. Takes ~90 seconds for the training step plus a few more for evaluation and plots. All figures render inline in the notebook and are also saved as PNG files in the folder.

The other scripts are optional deep-dives:

| File | What it does | Runtime |
|---|---|---|
| `final_run.ipynb` | **Main result.** Train + evaluate + produce all 5 thesis figures. | ~2 min |
| `sweep_nt_bonus.py` | Shows how the no-trade-bias parameter affects the result. Discovers the 0.0005 sweet spot. | ~2 min |
| `robust_sweep.py` | Trains 3 independent seeds, sweeps nt_bonus on each. Confirms robustness. | ~4 min |
| `final_compare.py` | Compares Single CCHP vs Double CCHP at full scale (3 seeds × 2 algorithms). Saves progress to `dq_final_results.pkl` so you can interrupt and resume. | ~10 min |

All scripts assume they're run from the folder they live in (so they can find `exp_framework.py` etc.). In VS Code, right-click the file and pick "Run Python File in Terminal", or open a terminal in the folder and `python sweep_nt_bonus.py`.

## File Map

```
hedging_rl/
├── README.md              ← this file
├── requirements.txt       ← pip dependencies
├── exp_framework.py       ← core library: CCHP dual-Q, BS utilities, env, eval
├── double_cchp.py         ← Double-Q variant of CCHP (4 Q-tables)
├── final_run.ipynb        ← main result notebook (START HERE)
├── sweep_nt_bonus.py      ← no-trade-bias parameter sweep
├── robust_sweep.py        ← 3-seed robustness check
└── final_compare.py       ← Single vs Double CCHP comparison
```

Running the scripts produces these output files in the same folder (safe to delete and regenerate):

- `final_training.png`, `final_histogram.png`, `final_frontier.png`, `final_trades.png`, `final_policy.png` — the thesis figures
- `final_Q_tables.npz` — trained Q-tables (compressed)
- `dq_final_results.pkl` — cached results from the Single-vs-Double comparison

## The Method in One Paragraph

The agent hedges a short call by choosing how much stock to hold at each daily timestep. State = `(time, moneyness, gap = position − BS_delta)`. Reward is the negative of the step's hedging cost (gains − transaction costs − option-value change). The CCHP innovation is to maintain **two** Q-functions: `Q1(s,a) = E[future cost]` and `Q2(s,a) = E[(future cost)²]`. The risk-adjusted policy minimizes `Q1 + c · sqrt(Q2 − Q1²)`. This is the only per-step bootstrapping scheme that correctly targets *cumulative* cost variance, which is what you want when hedging.

At evaluation time, a small bias `nt_bonus = 0.0005` is subtracted from the "do-nothing" action's Q-value. This breaks noisy ties in favor of holding, reducing over-trading from ~120 trades/year → ~10. This single intervention is what pushes the RL below the Black-Scholes benchmarks.

## Configuration (edit in the notebook)

```python
K, T, S0 = 100.0, 1.0, 100.0      # strike, maturity (years), spot
sigma, r = 0.2, 0.0                # vol, risk-free rate
dt, kappa = 1/252, 0.01            # daily steps, 1% transaction cost
c_risk = 1.5                       # risk aversion
nt_bonus = 0.0005                  # no-trade bias (try 0, 0.0005, 0.001)
n_batches = 800                    # training batches (reduce for faster runs)
```

To see how the result degrades without the no-trade bias, set `nt_bonus = 0.0` and rerun the evaluation cell — the RL will over-trade and lose to BS weekly.

## References

- Cao, Chen, Hull, Poulos (2021). *Deep Hedging of Derivatives Using Reinforcement Learning*. Journal of Financial Data Science, 3(1), 10-27. — The dual-Q formulation (their eq. 9-10).
- van Hasselt (2010). *Double Q-learning*. NeurIPS. — The overestimation-bias correction used in the optional Double CCHP variant.
- Davis, Panas, Zariphopoulou (1993). *European option pricing with transaction costs*. SIAM J. Control Optim, 31(2), 470-493. — The analytical no-trade band our agent empirically rediscovers.
- Hodges, Neuberger (1989). *Optimal replication of contingent claims under transaction costs*. Review of Futures Markets, 8, 222-239.
