"""
environment.py — Market simulation and hedging environment
===========================================================
Contains:
  - Black-Scholes pricing and delta  (thesis Eq. 4, 6)
  - GBM stock-price simulation       (thesis Def. 2.1)
  - State discretisation              (encode: τ, log-moneyness → grid cell)
  - Action discretisation             (a2h: action index → holding H)
  - Episode runner                    (APL reward, thesis Eq. 8 with S↔V swap)
  - BS and BS-quantised benchmarks

Reward (APL, Bergling/Evans/Abboudi Eq. 8, adapted for long-call/short-stock):
    R_{i+1} = (V_{i+1} − V_i) + H_i·(S_{i+1} − S_i) − κ·S_{i+1}·|H_{i+1} − H_i|
Plus initial cost −κ|S_0·H_0| and terminal liquidation −κ|S_n·H_n|.

State space:
    2-D grid (N_TIME × N_MONEY) over (time-to-maturity τ/T, log-moneyness m).
    m = clip(log(S/K), M_LO, M_HI).  See `encode()`.

Action space:
    N_ACT evenly spaced holdings H ∈ [-1, 0].  See `a2h()`.
    H = -1 is fully short stock, H = 0 is unhedged.
"""

import math
import numpy as np

# ─── Standard normal CDF ────────────────────────────────────────────────────
_SQRT2 = math.sqrt(2.0)


def _phi(x):
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ─── Black-Scholes pricing and delta (thesis Eq. 4–6) ───────────────────────

def bs_price(S, K, tau, r, sigma):
    """BS European call price.  Returns intrinsic value when τ ≈ 0."""
    if tau < 1e-10:
        return max(S - K, 0.0)
    s_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / s_sqrt
    d2 = d1 - s_sqrt
    return S * _phi(d1) - K * math.exp(-r * tau) * _phi(d2)


def bs_delta(S, K, tau, r, sigma):
    """BS call delta Φ(d₁).  Returns 1 (ITM) or 0 (OTM) when τ ≈ 0."""
    if tau < 1e-10:
        return 1.0 if S > K else 0.0
    s_sqrt = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / s_sqrt
    return _phi(d1)


# ─── State / action discretisation ──────────────────────────────────────────

def encode(tau, S, params):
    """
    Map continuous (τ, S) to grid indices (t_idx, m_idx).

    State space is a uniform N_TIME × N_MONEY grid over
        τ/T  ∈ [0, 1]
        m    ∈ [M_LO, M_HI]   where m = log(S/K)
    """
    T       = params["T"]
    K       = params["K"]
    N_TIME  = params["N_TIME"]
    N_MONEY = params["N_MONEY"]
    M_LO    = params["M_LO"]
    M_HI    = params["M_HI"]

    t_idx = min(int((tau / T) * N_TIME), N_TIME - 1)
    m     = max(M_LO, min(M_HI, math.log(S / K)))
    m_idx = min(int((m - M_LO) / (M_HI - M_LO) * N_MONEY), N_MONEY - 1)
    return t_idx, m_idx


def a2h(a, N_ACT):
    """
    Action index → stock holding H ∈ [-1, 0].

    a = 0         → H = -1   (fully short)
    a = N_ACT - 1 → H =  0   (unhedged)
    """
    return -(N_ACT - 1 - a) / (N_ACT - 1)


# ─── Episode runner ─────────────────────────────────────────────────────────

def run_episode(agent, params, eps, alpha, gamma, training=True):
    """
    Simulate one option lifetime (N daily steps) under the agent's policy.

    Returns (total_pnl, total_tc, n_trades).

    Reward: APL (thesis Eq. 8 with S↔V swap).
    Training penalty: R_pen = R − c·R²  (Cao variance regulariser, applied
        only to the TD target; accumulated PnL uses raw R).
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
    V      = bs_price(S, K, tau, r, sigma)

    # ── Opening trade ────────────────────────────────────────────────────
    a       = agent.act(tau, S, params, eps if training else 0.0)
    H       = a2h(a, N_ACT)
    tc_init = kappa * abs(S * H)
    total_pnl = -tc_init
    total_tc  = tc_init
    n_trades  = 1 if abs(H) > 1e-10 else 0

    # ── Daily steps ──────────────────────────────────────────────────────
    for step in range(N):
        z       = np.random.randn()
        S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                               + sigma * math.sqrt(dt) * z)
        tau_new = max(T - (step + 1) * dt, 0.0)
        V_new   = bs_price(S_new, K, tau_new, r, sigma)
        done    = (step == N - 1)

        if not done:
            a_new  = agent.act(tau_new, S_new, params, eps if training else 0.0)
            H_new  = a2h(a_new, N_ACT)
            tc_now = kappa * abs(S_new * (H_new - H))
            if abs(H_new - H) > 1e-10:
                n_trades += 1
        else:
            a_new, H_new, tc_now = a, H, 0.0

        # APL per-step reward (thesis Eq. 8, S↔V swap)
        R = (V_new - V) + H * (S_new - S) - tc_now

        if done:
            term_tc   = kappa * abs(S_new * H)
            R        -= term_tc
            total_tc += term_tc
            if abs(H) > 1e-10:
                n_trades += 1

        # ── Q-update (training only) ────────────────────────────────────
        if training:
            ti, mi     = encode(tau,     S,     params)
            ti_n, mi_n = encode(tau_new, S_new, params)
            R_pen      = R - agent.c * R * R          # Cao c·R² shaping
            agent.update(ti, mi, a, R_pen, ti_n, mi_n, alpha, gamma, done)

        total_pnl += R
        total_tc  += tc_now
        S, tau, V, a, H = S_new, tau_new, V_new, a_new, H_new

    return total_pnl, total_tc, n_trades


# ─── Benchmarks ─────────────────────────────────────────────────────────────

def bs_benchmark(params, n_ep=5000):
    """
    Black-Scholes delta hedge: H = −δ_BS at every step (continuous delta).
    This is the theoretically optimal frictionless strategy from thesis §2.1.2.
    """
    S0    = params["S0"]
    K     = params["K"]
    T     = params["T"]
    N     = params["N"]
    sigma = params["sigma"]
    r     = params["r"]
    kappa = params["kappa"]
    dt    = T / N

    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau = S0, T
        V      = bs_price(S, K, tau, r, sigma)
        H      = -bs_delta(S, K, tau, r, sigma)
        tc_init = kappa * abs(S * H)
        total_pnl, total_tc = -tc_init, tc_init
        nt = 1 if abs(H) > 1e-10 else 0

        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z)
            tau_new = max(T - (step + 1) * dt, 0.0)
            V_new   = bs_price(S_new, K, tau_new, r, sigma)
            done    = (step == N - 1)

            if not done:
                H_new  = -bs_delta(S_new, K, tau_new, r, sigma)
                tc_now = kappa * abs(S_new * (H_new - H))
                if abs(H_new - H) > 1e-10:
                    nt += 1
            else:
                H_new, tc_now = H, 0.0

            R = (V_new - V) + H * (S_new - S) - tc_now
            if done:
                term_tc    = kappa * abs(S_new * H)
                R         -= term_tc
                total_tc  += term_tc
                if abs(H) > 1e-10:
                    nt += 1

            total_pnl += R
            total_tc  += tc_now
            S, V, tau, H = S_new, V_new, tau_new, H_new

        pnls.append(total_pnl)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(pnls), np.array(tcs), np.array(trades)


def bs_quantised_benchmark(params, n_ep=5000):
    """
    BS delta rounded to the discrete action grid — isolates the effect of
    action-space discretisation from the effect of learning.
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

    pnls, tcs, trades = [], [], []
    for _ in range(n_ep):
        S, tau = S0, T
        V      = bs_price(S, K, tau, r, sigma)
        H_target = -bs_delta(S, K, tau, r, sigma)
        a_idx    = min(max(round(-H_target * (N_ACT - 1)), 0), N_ACT - 1)
        H        = a2h(N_ACT - 1 - a_idx, N_ACT)
        tc_init  = kappa * abs(S * H)
        total_pnl, total_tc = -tc_init, tc_init
        nt = 1 if abs(H) > 1e-10 else 0

        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z)
            tau_new = max(T - (step + 1) * dt, 0.0)
            V_new   = bs_price(S_new, K, tau_new, r, sigma)
            done    = (step == N - 1)

            if not done:
                H_t   = -bs_delta(S_new, K, tau_new, r, sigma)
                a_idx = min(max(round(-H_t * (N_ACT - 1)), 0), N_ACT - 1)
                H_new = a2h(N_ACT - 1 - a_idx, N_ACT)
                tc_now = kappa * abs(S_new * (H_new - H))
                if abs(H_new - H) > 1e-10:
                    nt += 1
            else:
                H_new, tc_now = H, 0.0

            R = (V_new - V) + H * (S_new - S) - tc_now
            if done:
                term_tc    = kappa * abs(S_new * H)
                R         -= term_tc
                total_tc  += term_tc
                if abs(H) > 1e-10:
                    nt += 1

            total_pnl += R
            total_tc  += tc_now
            S, V, tau, H = S_new, V_new, tau_new, H_new

        pnls.append(total_pnl)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(pnls), np.array(tcs), np.array(trades)


# ─── Policy visualisation helpers ───────────────────────────────────────────

def extract_policy(agent, params):
    """Extract the greedy policy as an N_TIME × N_MONEY matrix of H values."""
    N_TIME  = params["N_TIME"]
    N_MONEY = params["N_MONEY"]
    N_ACT   = params["N_ACT"]
    pol = np.zeros((N_TIME, N_MONEY))
    for ti in range(N_TIME):
        for mi in range(N_MONEY):
            pol[ti, mi] = a2h(int(np.argmax(agent.Q[ti, mi])), N_ACT)
    return pol


def bs_policy_grid(params):
    """BS delta on the same state grid — for visual comparison with RL."""
    N_TIME  = params["N_TIME"]
    N_MONEY = params["N_MONEY"]
    T       = params["T"]
    K       = params["K"]
    r       = params["r"]
    sigma   = params["sigma"]
    M_LO    = params["M_LO"]
    M_HI    = params["M_HI"]

    pol = np.zeros((N_TIME, N_MONEY))
    for ti in range(N_TIME):
        tau = max((ti + 0.5) / N_TIME * T, 1e-6)
        for mi in range(N_MONEY):
            m = M_LO + (mi + 0.5) / N_MONEY * (M_HI - M_LO)
            pol[ti, mi] = -bs_delta(K * math.exp(m), K, tau, r, sigma)
    return pol