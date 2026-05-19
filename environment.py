"""
environment.py — Market simulation and hedging environment
===========================================================
Contains:
  - Black-Scholes pricing and delta  (thesis Eq. 4, 6)
  - GBM stock-price simulation       (thesis Def. 2.1)
  - State discretisation              (encode: τ, log-moneyness → grid cell)
  - Action discretisation             (a2h: action index → holding H)
  - Episode runner                    (APL reward, thesis Eq. 8 with S↔V swap)
  - BS delta and BS no-trade-band benchmarks

Reward (APL, Bergling/Evans/Abboudi Eq. 8, adapted for long-call/short-stock):
    R_{i+1} = (V_{i+1} − V_i) + H_i·(S_{i+1} − S_i) − κ·S_{i+1}·|H_{i+1} − H_i|
Plus initial cost −κ|S_0·H_0| and terminal liquidation −κ|S_n·H_n|.

Hedging error (tracked per time step):
    HE_i = Portfolio_i − e^{-r·τ_i} · V_i
where Portfolio_i is the cumulative cash account (running PnL minus the
option's initial value, so it starts at V_0 in value terms and tracks how
well the hedge replicates the option's discounted price).

State space:
    2-D grid (N_TIME × N_MONEY) over (time-to-maturity τ/T, moneyness m).
    m = log(S/K) [log mode] or S/K [linear mode].  See `encode()`.
    Bin spacing: uniform (BIN_ALPHA=1.0) or ATM-compressed (BIN_ALPHA<1.0).
    Call build_money_edges(params) once before training to activate.

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


def build_money_edges(params):
    """
    Precompute moneyness bin edges and store them in params["M_EDGES"].

    Call this ONCE in main() after defining PARAMS, before training.
    The edges array has shape (N_MONEY + 1,) and is used by encode()
    and bs_policy_grid() for both uniform and geometric spacing.

    Spacing is controlled by params["BIN_ALPHA"]:

        BIN_ALPHA = 1.0   →  uniform edges (identical to the old behaviour)
        BIN_ALPHA < 1.0   →  power-law compression toward ATM.
                              Recommended range: 0.35 – 0.45.

    How the compression works
    -------------------------
    1. Map [M_LO, M_HI] linearly to the signed unit interval [-1, +1],
       so that ATM (m=0 for log, m=1 for linear) sits at 0.
    2. Apply  u → sign(u)·|u|^alpha  which compresses values toward 0
       (ATM) when alpha < 1.
    3. Map back to [M_LO, M_HI].

    The result: bins near ATM are narrow (high resolution) and bins in
    the tails are wide (low resolution), matching the visit-frequency
    and gamma profiles of a GBM path started ATM.
    """
    N_MONEY    = params["N_MONEY"]
    M_LO       = params["M_LO"]
    M_HI       = params["M_HI"]
    alpha      = params.get("BIN_ALPHA", 1.0)
    money_bins = params.get("MONEY_BINS", "log")

    # ATM in the chosen moneyness space
    atm = 0.0 if money_bins == "log" else 1.0

    if abs(alpha - 1.0) < 1e-9:
        # Uniform — straightforward linspace
        edges = np.linspace(M_LO, M_HI, N_MONEY + 1)
    else:
        # Step 1: uniform in [-1, +1] centred on ATM
        lo_signed = (M_LO - atm) / max(abs(M_LO - atm), abs(M_HI - atm))
        hi_signed = (M_HI - atm) / max(abs(M_LO - atm), abs(M_HI - atm))
        u = np.linspace(lo_signed, hi_signed, N_MONEY + 1)

        # Step 2: signed power compression toward 0
        u_c = np.sign(u) * np.abs(u) ** alpha

        # Step 3: map back to [M_LO, M_HI]
        half = max(abs(M_LO - atm), abs(M_HI - atm))
        edges = atm + u_c * half

        # Safety clip (rounding can push edge slightly outside bounds)
        edges[0]  = M_LO
        edges[-1] = M_HI

    params["M_EDGES"] = edges
    return edges


# ─── State / action discretisation ──────────────────────────────────────────

def encode(tau, S, params):
    """
    Map continuous (τ, S) to grid indices (t_idx, m_idx).

    Moneyness convention — set via params["MONEY_BINS"]:

        "log"    (default) — m = log(S/K),  bounds e.g. M_LO=-0.5 / M_HI=0.5
        "linear"           — m = S/K,        bounds e.g. M_LO=0.5  / M_HI=1.5

    Bin-spacing scheme — set via params["BIN_ALPHA"]:

        1.0  (default) — uniform spacing (original behaviour).
        < 1.0          — geometric (power-law) compression toward ATM (m=0
                         for log, m=1 for linear).  Recommended: 0.35–0.45.
                         Narrower bins near ATM where gamma is large and
                         hedging decisions are most sensitive; wider bins in
                         the rarely-visited tails.

    When BIN_ALPHA != 1.0 the bin edges are precomputed once by
    build_money_edges() and stored in params["M_EDGES"].  encode() then
    does a fast np.searchsorted lookup instead of the linear formula.
    """
    T          = params["T"]
    K          = params["K"]
    N_TIME     = params["N_TIME"]
    N_MONEY    = params["N_MONEY"]
    money_bins = params.get("MONEY_BINS", "log")

    t_idx = min(int((tau / T) * N_TIME), N_TIME - 1)

    # ── Raw moneyness value ──────────────────────────────────────────────
    if money_bins == "linear":
        m = S / K
    else:                           # "log"
        m = math.log(S / K)

    # ── Bin lookup ───────────────────────────────────────────────────────
    edges = params.get("M_EDGES")   # precomputed by build_money_edges()
    if edges is not None:
        # searchsorted returns index in [1, N_MONEY]; subtract 1 → [0, N_MONEY-1]
        m_idx = int(np.searchsorted(edges[1:-1], m))
        m_idx = min(max(m_idx, 0), N_MONEY - 1)
    else:
        # Uniform fallback (BIN_ALPHA = 1.0 or edges not built yet)
        M_LO = params["M_LO"]
        M_HI = params["M_HI"]
        m    = max(M_LO, min(M_HI, m))
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

    Returns
    -------
    total_pnl : float
        Total accumulated APL PnL over the episode.
    total_tc  : float
        Total transaction costs paid.
    n_trades  : int
        Number of trades executed.
    he_path   : np.ndarray, shape (N+1,)
        Per-step hedging error  HE_i = Portfolio_i − e^{-r·τ_i}·V_i.
        Portfolio_i = V_0 + (cumulative APL PnL up to step i).
        HE_0 reflects the initial transaction cost; HE_N is the terminal
        replication error.

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
    V0     = bs_price(S, K, tau, r, sigma)
    V      = V0

    # ── Opening trade ────────────────────────────────────────────────────
    a       = agent.act(tau, S, params, eps if training else 0.0)
    H       = a2h(a, N_ACT)
    tc_init = kappa * abs(S * H)
    total_pnl = -tc_init
    total_tc  = tc_init
    n_trades  = 1 if abs(H) > 1e-10 else 0

    # ── Hedging-error tracking ───────────────────────────────────────────
    # Portfolio value at step i = V_0 + cumulative APL PnL.
    # HE_i = Portfolio_i − e^{-r·τ_i} · V_i.
    he_path = np.zeros(N + 1)
    portfolio = V0 + total_pnl
    he_path[0] = portfolio - math.exp(-r * tau) * V

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

        # Update hedging error
        portfolio = V0 + total_pnl
        he_path[step + 1] = portfolio - math.exp(-r * tau_new) * V_new

        S, tau, V, a, H = S_new, tau_new, V_new, a_new, H_new

    return total_pnl, total_tc, n_trades, he_path


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
    he_paths = np.zeros((n_ep, N + 1))

    for ep in range(n_ep):
        S, tau = S0, T
        V0     = bs_price(S, K, tau, r, sigma)
        V      = V0
        H      = -bs_delta(S, K, tau, r, sigma)
        tc_init = kappa * abs(S * H)
        total_pnl, total_tc = -tc_init, tc_init
        nt = 1 if abs(H) > 1e-10 else 0

        portfolio = V0 + total_pnl
        he_paths[ep, 0] = portfolio - math.exp(-r * tau) * V

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

            portfolio = V0 + total_pnl
            he_paths[ep, step + 1] = portfolio - math.exp(-r * tau_new) * V_new

            S, V, tau, H = S_new, V_new, tau_new, H_new

        pnls.append(total_pnl)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(pnls), np.array(tcs), np.array(trades), he_paths


def bs_band_benchmark(params, n_ep=5000, band_type="fixed",
                      band_width=0.10, c_band=1.0):
    """
    Black-Scholes delta hedge with a NO-TRADE BAND.

    At every step the target is −δ_BS, but the position is only rebalanced
    when |H − target| > h (the band half-width).  When triggered the holding
    snaps back to the target (centre of the band).

    Parameters
    ----------
    band_type : {"fixed", "ww"}
        "fixed" — constant half-width = band_width.  Sweep band_width to
                  trace the empirical cost-vs-risk frontier.
        "ww"    — Whalley–Wilmott analytical half-width (Math. Finance 1997,
                  asymptotic small-κ optimum under exponential utility):
                      h*(t,S) = ( 3·κ·(Γ·S)² / (2·c_band) )^{1/3}
                  where Γ is the closed-form BS gamma.  band_width ignored.
    band_width : float
        Half-width used when band_type="fixed".
    c_band : float
        Risk-aversion coefficient used inside the WW formula (band_type="ww").
        Has no effect for band_type="fixed".

    Returns
    -------
    pnls, tcs, trades : np.ndarray, shape (n_ep,)
    he_paths          : np.ndarray, shape (n_ep, N+1)
        Per-step hedging-error trajectories, same definition as run_episode.
    """
    S0    = params["S0"]
    K     = params["K"]
    T     = params["T"]
    N     = params["N"]
    sigma = params["sigma"]
    r     = params["r"]
    kappa = params["kappa"]
    dt    = T / N

    # ── Whalley–Wilmott half-width helper ────────────────────────────────
    _inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    def _ww_half_width(S_t, tau_t):
        """Closed-form WW band half-width at (S_t, tau_t)."""
        if tau_t < 1e-10:
            return 0.0
        s_sqrt = sigma * math.sqrt(tau_t)
        d1     = (math.log(S_t / K) + (r + 0.5 * sigma ** 2) * tau_t) / s_sqrt
        phi_d1 = _inv_sqrt_2pi * math.exp(-0.5 * d1 * d1)
        gamma  = phi_d1 / (S_t * s_sqrt)
        return ((3.0 * kappa * (gamma * S_t) ** 2) / (2.0 * c_band)) ** (1.0 / 3.0)

    pnls, tcs, trades = [], [], []
    he_paths = np.zeros((n_ep, N + 1))

    for ep in range(n_ep):
        S, tau = S0, T
        V0     = bs_price(S, K, tau, r, sigma)
        V      = V0

        # Opening trade: no band at issue — snap straight to −δ_BS.
        H       = -bs_delta(S, K, tau, r, sigma)
        tc_init = kappa * abs(S * H)
        total_pnl = -tc_init
        total_tc  = tc_init
        nt = 1 if abs(H) > 1e-10 else 0

        portfolio = V0 + total_pnl
        he_paths[ep, 0] = portfolio - math.exp(-r * tau) * V

        for step in range(N):
            z       = np.random.randn()
            S_new   = S * math.exp((r - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z)
            tau_new = max(T - (step + 1) * dt, 0.0)
            V_new   = bs_price(S_new, K, tau_new, r, sigma)
            done    = (step == N - 1)

            if not done:
                target = -bs_delta(S_new, K, tau_new, r, sigma)
                if band_type == "ww":
                    h = _ww_half_width(S_new, tau_new)
                else:                           # "fixed"
                    h = band_width

                if abs(H - target) > h:         # outside band → rebalance
                    H_new  = target
                    tc_now = kappa * abs(S_new * (H_new - H))
                    nt    += 1
                else:                           # inside band → hold
                    H_new, tc_now = H, 0.0
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

            portfolio = V0 + total_pnl
            he_paths[ep, step + 1] = portfolio - math.exp(-r * tau_new) * V_new

            S, V, tau, H = S_new, V_new, tau_new, H_new

        pnls.append(total_pnl)
        tcs.append(total_tc)
        trades.append(nt)

    return np.array(pnls), np.array(tcs), np.array(trades), he_paths



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
    """
    BS delta on the same state grid — for visual comparison with RL.

    Uses the cell-centre of each bin (midpoint between adjacent edges)
    and respects both MONEY_BINS convention and BIN_ALPHA spacing.
    """
    N_TIME     = params["N_TIME"]
    N_MONEY    = params["N_MONEY"]
    T          = params["T"]
    K          = params["K"]
    r          = params["r"]
    sigma      = params["sigma"]
    money_bins = params.get("MONEY_BINS", "log")

    # Use precomputed edges if available, otherwise fall back to uniform
    edges = params.get("M_EDGES")
    if edges is None:
        M_LO = params["M_LO"]
        M_HI = params["M_HI"]
        edges = np.linspace(M_LO, M_HI, N_MONEY + 1)

    pol = np.zeros((N_TIME, N_MONEY))
    for ti in range(N_TIME):
        tau = max((ti + 0.5) / N_TIME * T, 1e-6)
        for mi in range(N_MONEY):
            m = 0.5 * (edges[mi] + edges[mi + 1])  # cell centre
            if money_bins == "linear":
                S_cell = m * K            # m = S/K  →  S = m·K
            else:
                S_cell = K * math.exp(m)  # m = log(S/K)  →  S = K·e^m
            pol[ti, mi] = -bs_delta(S_cell, K, tau, r, sigma)
    return pol

# ─── Policy query at arbitrary (τ, S) — for moneyness-slice plot ────────────

def rl_hedge_ratio(agent, S, K, tau, params):
    """
    Query the RL policy at a continuous (τ, S) point and return the implied
    hedge ratio in [0, 1] (i.e. −H, since H ∈ [−1, 0] and the BLS-delta
    plot is in hedge-ratio convention).
    """
    ti, mi = encode(tau, S, params)
    H = a2h(int(np.argmax(agent.Q[ti, mi])), params["N_ACT"])
    return -H
