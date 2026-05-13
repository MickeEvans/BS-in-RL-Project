"""
Q-Learning Hedging Agent  —  APL (P&L) reward per Bergling/Evans/Abboudi
========================================================================
Hedging a LONG European call by trading the underlying stock.
Action H = number of stock units held (H in [-1, 0] — short stock to
offset the positive call delta).

Per-step reward (Accounting Profit & Loss, Bergling/Evans/Abboudi Eq.
(8), adapted to our long-call / short-stock position by swapping the
roles of S and V relative to the article — the article holds the stock
and trades the option, here we hold the option and trade the stock):

    R_{i+1} = (V_{i+1} − V_i) + H_i·(S_{i+1} − S_i)
              − κ·S_{i+1}·|H_{i+1} − H_i|,        i = 0, …, n − 1.

In addition the episode return is charged
    initial hedge establishment cost  −κ·|S_0 · H_0|
    terminal liquidation cost         −κ·|S_n · H_n|
as per the article (Section 2.2.2, last paragraph).

NO c·R² Cao penalty — this matches the thesis APL reward exactly.

State: (time-to-maturity, log-moneyness)   — 2-D, no warm start.
Q-learning: tabular, ε-greedy, γ = 1, Robbins-Monro α-decay.

Diagnostic
----------
Trade counter `n_trades` = number of rebalancing events per episode:
  * +1 if the opening position H_0 is non-zero,
  * +1 per intermediate step where H_{i+1} ≠ H_i,
  * +1 for the terminal liquidation if H_n is non-zero.
For reference BS-Δ rebalances ~248 / 252 steps per episode under this
grid.
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math, time, warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

_SQRT2 = math.sqrt(2.0)
def Phi(x): return 0.5 * (1.0 + math.erf(x / _SQRT2))

# ── Market parameters (as specified) ─────────────────────────────────────────
S0, K, T, N, sigma, r, kappa = 100.0, 100.0, 1.0, 252, 0.20, 0.0, 0.01
dt = T / N

# ── Black-Scholes ────────────────────────────────────────────────────────────
def bs_price(S, K, tau, r, sigma):
    if tau < 1e-10: return max(S - K, 0.0)
    s_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / s_sqrt
    d2 = d1 - s_sqrt
    return S * Phi(d1) - K * math.exp(-r * tau) * Phi(d2)

def bs_delta_fn(S, K, tau, r, sigma):
    if tau < 1e-10: return 1.0 if S > K else 0.0
    s_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / s_sqrt
    return Phi(d1)

# ── State / action discretisation ────────────────────────────────────────────
# State grid: 5×15 (75 cells) — Pareto-dominant from the 2-D sweep
# (rl_hedging_grid_sweep.py, 2026-05-12).  N_MONEY matters far more than
# N_TIME — the BS gradient ∂δ/∂m = φ(d₁)/(σ√τ) is primarily a function of m;
# the agent doesn't need fine time resolution.  Coarse N_TIME=5 gives ~6000
# samples per cell, killing per-cell variance without smoothing.
# Alternative grids (all c=4, N_ACT=5, no smoothing, 30k ep):
#   10×5   → TC=4.68 (−24% vs BS), std=2.82 (+41%) -- absolute min-TC corner
#   5×15   → TC=5.57 (−10%),       std=2.22 (+11%) -- BALANCED (default)
#   10×15  → TC=5.62 (−9%),        std=2.53 (+27%) -- prior default
#   5×10   → TC=5.43 (−12%),       std=2.56 (+28%)
#   25×30  → TC=6.83 (+11% WORSE)                  -- fails without smoothing
N_TIME, N_MONEY = 5, 15
# N_ACT=5 (H ∈ {-1.0, -0.75, -0.5, -0.25, 0.0}) Pareto-dominates N_ACT=11 at
# c=4.  Step 0.25 keeps the ATM BS-delta H=-0.5 ON the grid (grid alignment
# matters sharply — even-N_ACT grids that miss -0.5 perform much worse).
N_ACT           = 5
M_LO, M_HI      = -0.5, 0.5

def encode(tau, S):
    t_idx = min(int((tau / T) * N_TIME), N_TIME - 1)
    m = max(M_LO, min(M_HI, math.log(S / K)))
    m_idx = min(int((m - M_LO) / (M_HI - M_LO) * N_MONEY), N_MONEY - 1)
    return t_idx, m_idx

def a2h(a):                                  # action index → holding H ∈ [-1, 0]
    return -(N_ACT - 1 - a) / (N_ACT - 1)

# ── Q-Learning Agent ─────────────────────────────────────────────────────────
class QHedger:
    def __init__(self, name="QL", c=0.0):
        # c is the Cao R² curvature regulariser used ONLY in the TD target
        # (R_pen = R − c·R²). c=0 → pure thesis-APL reward.  c>0 makes the
        # variance-minimising hedge (BS delta) optimal in expectation.
        self.name = name
        self.c    = c
        self.Q    = np.zeros((N_TIME, N_MONEY, N_ACT))     # NO warm start

    def act(self, tau, S, eps):
        ti, mi = encode(tau, S)
        if np.random.random() < eps:
            return np.random.randint(N_ACT)
        return int(np.argmax(self.Q[ti, mi]))

    def update(self, ti, mi, a, reward, ti_n, mi_n, alpha, gamma, done):
        target = reward if done else reward + gamma * self.Q[ti_n, mi_n].max()
        self.Q[ti, mi, a] += alpha * (target - self.Q[ti, mi, a])

# ── Episode (APL / thesis reward, with trade counter) ────────────────────────
def run_episode(agent, eps, alpha, gamma, training=True):
    S, tau    = S0, T
    V         = bs_price(S, K, tau, r, sigma)
    a         = agent.act(tau, S, eps if training else 0.0)
    H         = a2h(a)
    tc_init   = kappa * abs(S * H)           # opening trade from 0 → H
    total_pnl = -tc_init
    total_tc  = tc_init
    n_trades  = 1 if abs(H) > 1e-10 else 0   # opening counts as a trade

    for step in range(N):
        z       = np.random.randn()
        S_new   = S * math.exp((r - 0.5 * sigma * sigma) * dt
                               + sigma * math.sqrt(dt) * z)
        tau_new = max(T - (step + 1) * dt, 0.0)
        V_new   = bs_price(S_new, K, tau_new, r, sigma)
        done    = (step == N - 1)

        if not done:
            a_new  = agent.act(tau_new, S_new, eps if training else 0.0)
            H_new  = a2h(a_new)
            tc_now = kappa * abs(S_new * (H_new - H))
            if abs(H_new - H) > 1e-10:
                n_trades += 1
        else:
            a_new, H_new, tc_now = a, H, 0.0

        # APL per-step reward (thesis Eq. (8), with S↔V swap so that we are
        # long the option and short stock).  No c·R² penalty.
        R = (V_new - V) + H * (S_new - S) - tc_now

        if done:                              # terminal liquidation
            term_tc   = kappa * abs(S_new * H)
            R        -= term_tc
            total_tc += term_tc
            if abs(H) > 1e-10:
                n_trades += 1                 # liquidation counts

        if training:
            ti, mi     = encode(tau,     S)
            ti_n, mi_n = encode(tau_new, S_new)
            # Cao curvature regulariser: training target uses R − c·R².
            # The accumulated PnL stream still uses the raw APL reward.
            R_pen = R - agent.c * R * R
            agent.update(ti, mi, a, R_pen, ti_n, mi_n, alpha, gamma, done)

        total_pnl += R
        total_tc  += tc_now

        S, tau, V, a, H = S_new, tau_new, V_new, a_new, H_new

    return total_pnl, total_tc, n_trades

# ── Training loop ────────────────────────────────────────────────────────────
def train(agent, n_ep=30000, gamma=1.0, verbose=True):
    eps_s, eps_e     = 1.00, 0.05
    alpha_s, alpha_e = 0.10, 0.005
    eps_d   = (eps_e   / eps_s)   ** (1.0 / n_ep)
    alpha_d = (alpha_e / alpha_s) ** (1.0 / n_ep)
    eps, alpha = eps_s, alpha_s
    log, t0 = [], time.time()
    for ep in range(n_ep):
        eps   *= eps_d
        alpha *= alpha_d
        pnl, tc, _ = run_episode(agent, eps, alpha, gamma, training=True)
        log.append((pnl, tc))
        if verbose and (ep + 1) % (n_ep // 6) == 0:
            recent = log[-(n_ep // 12):]
            print("  [%s] ep %5d eps=%.3f alpha=%.4f | PnL=%.4f TC=%.4f"
                  % (agent.name, ep + 1, eps, alpha,
                     np.mean([x[0] for x in recent]),
                     np.mean([x[1] for x in recent])))
    print("  Training done in %.1fs" % (time.time() - t0))
    return log

def evaluate(agent, n_ep=5000):
    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        pnl, tc, nt = run_episode(agent, 0.0, 0.0, 1.0, training=False)
        pnls.append(pnl); tcs.append(tc); trades.append(nt)
    return np.array(pnls), np.array(tcs), np.array(trades)

# ── Black-Scholes benchmark (LONG call hedged with H = -δ) ──────────────────
def bs_benchmark(n_ep=5000):
    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau = S0, T
        V      = bs_price(S, K, tau, r, sigma)
        H      = -bs_delta_fn(S, K, tau, r, sigma)
        tc_init = kappa * abs(S * H)
        total_pnl, total_tc = -tc_init, tc_init
        nt = 1 if abs(H) > 1e-10 else 0
        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*z)
            tau_new = max(T - (step+1)*dt, 0.0)
            V_new   = bs_price(S_new, K, tau_new, r, sigma)
            done    = (step == N-1)
            if not done:
                H_new  = -bs_delta_fn(S_new, K, tau_new, r, sigma)
                tc_now = kappa * abs(S_new * (H_new - H))
                if abs(H_new - H) > 1e-10:
                    nt += 1
            else:
                H_new, tc_now = H, 0.0
            R = (V_new - V) + H * (S_new - S) - tc_now
            if done:
                term_tc = kappa * abs(S_new * H)
                R       -= term_tc
                total_tc += term_tc
                if abs(H) > 1e-10:
                    nt += 1
            total_pnl += R
            total_tc  += tc_now
            S, V, tau, H = S_new, V_new, tau_new, H_new
        pnls.append(total_pnl); tcs.append(total_tc); trades.append(nt)
    return np.array(pnls), np.array(tcs), np.array(trades)

# ── BS quantised (continuous BS rounded to action grid) ─────────────────────
def bs_quantised_benchmark(n_ep=5000):
    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau = S0, T
        V      = bs_price(S, K, tau, r, sigma)
        # snap H to grid
        H_target = -bs_delta_fn(S, K, tau, r, sigma)
        a_idx    = min(max(round(-H_target * (N_ACT - 1)), 0), N_ACT - 1)
        H        = a2h(N_ACT - 1 - a_idx)
        tc_init  = kappa * abs(S * H)
        total_pnl, total_tc = -tc_init, tc_init
        nt = 1 if abs(H) > 1e-10 else 0
        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*z)
            tau_new = max(T - (step+1)*dt, 0.0)
            V_new   = bs_price(S_new, K, tau_new, r, sigma)
            done    = (step == N-1)
            if not done:
                H_t   = -bs_delta_fn(S_new, K, tau_new, r, sigma)
                a_idx = min(max(round(-H_t * (N_ACT - 1)), 0), N_ACT - 1)
                H_new = a2h(N_ACT - 1 - a_idx)
                tc_now = kappa * abs(S_new * (H_new - H))
                if abs(H_new - H) > 1e-10:
                    nt += 1
            else:
                H_new, tc_now = H, 0.0
            R = (V_new - V) + H * (S_new - S) - tc_now
            if done:
                term_tc = kappa * abs(S_new * H)
                R       -= term_tc
                total_tc += term_tc
                if abs(H) > 1e-10:
                    nt += 1
            total_pnl += R
            total_tc  += tc_now
            S, V, tau, H = S_new, V_new, tau_new, H_new
        pnls.append(total_pnl); tcs.append(total_tc); trades.append(nt)
    return np.array(pnls), np.array(tcs), np.array(trades)

# ── Policy extraction ───────────────────────────────────────────────────────
def extract_policy(agent):
    pol = np.zeros((N_TIME, N_MONEY))
    for ti in range(N_TIME):
        for mi in range(N_MONEY):
            pol[ti, mi] = a2h(int(np.argmax(agent.Q[ti, mi])))
    return pol

def bs_policy_grid():
    pol = np.zeros((N_TIME, N_MONEY))
    for ti in range(N_TIME):
        tau = max((ti + 0.5) / N_TIME * T, 1e-6)
        for mi in range(N_MONEY):
            m = M_LO + (mi + 0.5) / N_MONEY * (M_HI - M_LO)
            pol[ti, mi] = -bs_delta_fn(K * math.exp(m), K, tau, r, sigma)
    return pol

# =============================================================================
# MAIN
# =============================================================================
print("=" * 78)
print("Q-Learning Hedging  —  APL (thesis) reward, NO warm start")
print("=" * 78)

print("\n[BS benchmark]")
bs_pnl, bs_tc, bs_tr = bs_benchmark(5000)
print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
      % (bs_tc.mean(), bs_pnl.std(), bs_pnl.mean(), bs_tr.mean()))

print("\n[BS quantised — BS-delta rounded to 0.1 grid]")
bsq_pnl, bsq_tc, bsq_tr = bs_quantised_benchmark(5000)
print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
      % (bsq_tc.mean(), bsq_pnl.std(), bsq_pnl.mean(), bsq_tr.mean()))

# PnL stream is always thesis-APL.  c is a training-side regulariser only:
# c=0 reproduces the article (and the "few-but-fat trades" pathology);
# c>0 adds variance-penalising curvature in the action axis whose optimum
# is the BS-delta hedge.
configs = [
    dict(c=0.0, name="QL_APL_c0",   label="APL  c=0   (thesis pure)"),
    dict(c=0.7, name="QL_APL_c07",  label="APL  c=0.7 (min-TC)"),
    dict(c=3.0, name="QL_APL_c30",  label="APL  c=3.0 (min-risk)"),
    dict(c=4.0, name="QL_APL_c40",  label="APL  c=4.0 (Pareto-best)"),
]

agents, results = [], {}
for cfg in configs:
    print("\n[Train %s]" % cfg["label"])
    ag = QHedger(name=cfg["name"], c=cfg["c"])
    train(ag, n_ep=30000, gamma=1.0)
    pnl, tc, tr = evaluate(ag, n_ep=5000)
    results[cfg["name"]] = dict(pnl=pnl, tc=tc, trades=tr, label=cfg["label"])
    agents.append(ag)
    print("  TC=%.4f  std(PnL)=%.4f  mean(PnL)=%.4f  trades/ep=%.2f"
          "   (BS TC=%.4f, trades=%.1f)"
          % (tc.mean(), pnl.std(), pnl.mean(), tr.mean(),
             bs_tc.mean(), bs_tr.mean()))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("%-28s  %10s  %10s  %10s  %10s"
      % ("Strategy", "Mean TC", "Std PnL", "Mean PnL", "Trades/ep"))
print("-" * 78)
print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
      % ("BS Delta", bs_tc.mean(), bs_pnl.std(), bs_pnl.mean(), bs_tr.mean()))
print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
      % ("BS quantised (0.1 grid)", bsq_tc.mean(), bsq_pnl.std(),
         bsq_pnl.mean(), bsq_tr.mean()))
for name, res in results.items():
    print("%-28s  %10.4f  %10.4f  %10.4f  %10.2f"
          % (res["label"][:28], res["tc"].mean(), res["pnl"].std(),
             res["pnl"].mean(), res["trades"].mean()))
print("=" * 78)

# ── Plotting ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Q-Learning Hedging  —  APL (thesis) reward, no warm start",
             fontsize=14, fontweight="bold")

palette = ["tomato", "darkorange", "green", "purple"]

ax = axes[0, 0]
ax.hist(bs_pnl, bins=70, alpha=0.55, label="BS Delta", color="steelblue", density=True)
for (name, res), c in zip(results.items(), palette):
    ax.hist(res["pnl"], bins=70, alpha=0.45, label=res["label"], color=c, density=True)
ax.set_title("PnL distribution"); ax.set_xlabel("PnL")
ax.set_ylabel("Density"); ax.legend(fontsize=7)

ax = axes[0, 1]
ax.hist(bs_tc, bins=70, alpha=0.55, label="BS Delta", color="steelblue", density=True)
for (name, res), c in zip(results.items(), palette):
    ax.hist(res["tc"], bins=70, alpha=0.45, label=res["label"], color=c, density=True)
ax.set_title("Transaction-cost distribution"); ax.set_xlabel("TC")
ax.set_ylabel("Density"); ax.legend(fontsize=7)

ax = axes[0, 2]
ax.scatter(bs_tc.mean(), bs_pnl.std(), s=200, marker="*", color="steelblue",
           zorder=5, label="BS Delta")
ax.scatter(bsq_tc.mean(), bsq_pnl.std(), s=120, marker="P", color="grey",
           zorder=5, label="BS quantised")
for (name, res), c in zip(results.items(), palette):
    ax.scatter(res["tc"].mean(), res["pnl"].std(), s=90, marker="o",
               color=c, zorder=5, label=res["label"])
ax.set_title("TC  vs  Risk (Std PnL)")
ax.set_xlabel("Mean transaction cost"); ax.set_ylabel("Std(PnL)")
ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Pick best agent by lowest TC for the policy plot.
best_name = min(results, key=lambda k: results[k]["tc"].mean())
best_idx  = next(i for i, c in enumerate(configs)
                 if c["name"] == best_name)
best_agent = agents[best_idx]

ax = axes[1, 0]
im = ax.imshow(bs_policy_grid(), origin="lower", aspect="auto",
               vmin=-1, vmax=0, cmap="RdYlGn",
               extent=[M_LO, M_HI, 0, 1])
ax.set_title("BS Hedge  H = -delta")
ax.set_xlabel("Log-Moneyness log(S/K)")
ax.set_ylabel("Time-to-Maturity tau/T")
plt.colorbar(im, ax=ax, label="H")

ax = axes[1, 1]
im = ax.imshow(extract_policy(best_agent), origin="lower", aspect="auto",
               vmin=-1, vmax=0, cmap="RdYlGn",
               extent=[M_LO, M_HI, 0, 1])
ax.set_title("QL Policy  (%s)" % best_agent.name)
ax.set_xlabel("Log-Moneyness log(S/K)")
ax.set_ylabel("Time-to-Maturity tau/T")
plt.colorbar(im, ax=ax, label="H")

ax = axes[1, 2]
diff = extract_policy(best_agent) - bs_policy_grid()
vmax = max(np.abs(diff).max(), 1e-6)
im   = ax.imshow(diff, origin="lower", aspect="auto",
                 vmin=-vmax, vmax=vmax, cmap="bwr",
                 extent=[M_LO, M_HI, 0, 1])
ax.set_title("QL  -  BS Hedge")
ax.set_xlabel("Log-Moneyness log(S/K)")
ax.set_ylabel("Time-to-Maturity tau/T")
plt.colorbar(im, ax=ax, label="H diff")

plt.tight_layout()
print("\nClose the plot window when you're done viewing "
      "(use the toolbar's save icon to keep it).")
plt.show()

print("\nDone.")