"""
Hedging environment for a short European call.

State:
    s = (tau, m)
    where  tau = time-to-maturity (in years)
           m   = moneyness S / K

Action:
    a = desired hedge ratio H in [0, 1]   (number of shares held long
        against one short call). Discretized into a finite grid for tabular Q.

Reward modes:
    'apl' - Accounting P&L formulation, thesis Section 2.2.2, eq. (8).
            Per-period:
                R_{i+1} = H_i*(S_{i+1} - S_i) - (V_{i+1} - V_i) - kappa*|S_i*dH|
            where V_i = C^BS(S_i, K, tau_i, r, sigma) is the BS call price.
            Plus initial setup cost -kappa*|S_0*H_0| and terminal liquidation
            cost -kappa*|S_N*H_N|. This is the THESIS TRAINING REWARD.
            BS is used inside the reward; the agent learns to minimize a
            risk-adjusted P&L against a BS-priced book.

    'cf'  - Cash-flow formulation, thesis Section 2.2.3, eq. (10).
            Per-period reward is only the realized cash flow from trading.
            Pure BS-free signal but very sparse (almost all signal is at
            the terminal payoff). Used for EVALUATION, per the thesis.

    'pnl' - Stock-only P&L (no BS in reward). Same as 'apl' but with the
            -(V_{i+1} - V_i) term dropped. The hedger sees only their own
            stock holding's MtM change, ignoring the option's MtM.
            Kept for ablation comparisons.

Trades happen on the stock (the hedger holds H shares against a short call).
Transaction cost = kappa * |S_i * dH| (stock-trading convention).

Stock dynamics use the risk-neutral drift r (matches thesis Definition 2.1).

The agent receives the option premium p at t=0 (recorded for accounting); we
report terminal hedging cost = -(premium + sum_of_step_rewards) so that a
perfectly replicating frictionless BS hedge has cost ~0.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Black-Scholes
# -----------------------------------------------------------------------------
from scipy.stats import norm
import math

# Vectorized cdf via scipy (for batch use). For inner loops we use math.erf.
_norm_cdf = norm.cdf

def _ncdf_scalar(x):
    """Fast scalar standard-normal CDF using math.erf (~10x faster than scipy)."""
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))

def bs_price(S, K, r, T, sigma):
    """Vectorized BS call price; handles T=0 gracefully."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    price = S * _norm_cdf(d1) - K * np.exp(-r * T_safe) * _norm_cdf(d2)
    return np.where(T <= 1e-12, np.maximum(S - K, 0.0), price)


def bs_price_scalar(S, K, r, T, sigma):
    """Fast scalar BS call price for inner training loops."""
    if T <= 1e-12:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _ncdf_scalar(d1) - K * math.exp(-r * T) * _ncdf_scalar(d2)

def bs_delta(S, K, r, T, sigma):
    """Vectorized BS call delta; handles T=0."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    T_safe = np.maximum(T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_safe) / (sigma * np.sqrt(T_safe))
    return np.where(T <= 1e-12, np.where(S > K, 1.0, 0.0), _norm_cdf(d1))


def bs_delta_scalar(S, K, r, T, sigma):
    """Fast scalar BS call delta for inner loops."""
    if T <= 1e-12:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf_scalar(d1)


# -----------------------------------------------------------------------------
# Hedging environment
# -----------------------------------------------------------------------------
class HedgingEnv:
    """
    Short-call hedging episode.

    Parameters
    ----------
    S0, K       : spot, strike
    T           : maturity (years)
    N           : number of trading dates between 0 and T
    mu, sigma   : real-world drift and volatility for GBM
    r           : risk-free rate
    kappa       : proportional transaction cost
    """
    def __init__(self, S0=100.0, K=100.0, T=1.0, N=252,
                 sigma=0.20, r=0.0, kappa=0.01, seed=None,
                 reward_mode="apl", cost_basis="stock"):
        """
        Parameters
        ----------
        S0, K        : spot, strike
        T            : maturity (years)
        N            : number of trading dates between 0 and T
        sigma        : GBM volatility
        r            : risk-free rate (also the drift of S under the
                       risk-neutral measure, matching thesis Definition 2.1)
        kappa        : proportional transaction cost
        reward_mode  : 'apl' (thesis eq. 8, training reward, uses BS prices),
                       'cf'  (thesis eq. 10, evaluation reward, no BS),
                       'pnl' (stock-only P&L, no BS, ablation only)
        cost_basis   : 'stock'  -> TC = kappa * |S * dH|  (default; matches
                                  the Cao et al. convention and the cost
                                  audit decision in this project)
                       'option' -> TC = kappa * |V * dH|  (literal thesis
                                  eq. (7), with V = current BS call price)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.N = N
        self.dt = T / N
        # Stock simulated under risk-neutral measure: drift = r
        self.sigma = sigma
        self.r = r
        self.kappa = kappa
        self.reward_mode = reward_mode
        assert cost_basis in ("stock", "option")
        self.cost_basis = cost_basis
        self.rng = np.random.default_rng(seed)

        # Initial BS option premium (used to score the realized hedge cost).
        self.option_premium = float(bs_price(S0, K, r, T, sigma))

        self.reset()

    # -------------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.S = self.S0
        self.H = 0.0          # hedge holding (shares of stock, long)
        # We track hedger cash flows ONLY (not the option premium).
        # Cost := premium - terminal_cash (= what the hedge ate of the premium).
        self.cash = 0.0
        # Cache for APL: BS value at current time/price; reuse across steps.
        # Initialize to BS(S0, T) which equals option_premium.
        self._V_cached = self.option_premium
        return self._obs()

    def _obs(self):
        tau = self.T - self.t * self.dt
        m = self.S / self.K
        return np.array([tau, m], dtype=np.float64)

    # -------------------------------------------------------------------------
    def step(self, new_H):
        """
        new_H is the desired hedge ratio for the period [t_i, t_{i+1}).
        At time t_i we trade from H_i to new_H, paying transaction cost on the
        change. Then the stock evolves to t_{i+1}.

        Returns (next_obs, reward, done, info).
        """
        S_now = self.S
        H_old = self.H
        dH = new_H - H_old
        tau_now  = self.T - self.t * self.dt
        # BS option value at the current step: needed if reward_mode='apl'
        # OR cost_basis='option'. Served from cache.
        need_V = (self.reward_mode == "apl") or (self.cost_basis == "option")
        if need_V:
            V_now = self._V_cached

        # ------- trade at the current price ------------------------------
        trade_cashflow = -S_now * dH                     # buy dH shares
        if self.cost_basis == "stock":
            trade_cost = -self.kappa * abs(S_now * dH)
        else:  # 'option'
            trade_cost = -self.kappa * abs(V_now * dH)
        self.cash += trade_cashflow + trade_cost
        self.H = new_H

        # ------- evolve stock with GBM (RISK-NEUTRAL drift r) ------------
        z = self.rng.standard_normal()
        S_next = S_now * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt
                                + self.sigma * np.sqrt(self.dt) * z)

        # cash earns risk-free rate over [t_i, t_{i+1})
        self.cash *= np.exp(self.r * self.dt)

        self.S = S_next
        self.t += 1
        tau_next = self.T - self.t * self.dt
        done = self.t >= self.N

        # BS option value at the next step (needed if APL reward OR option cost).
        # Cache it so the next step doesn't recompute.
        if need_V:
            if done:
                V_next = max(S_next - self.K, 0.0)          # phi(S_T)
            else:
                V_next = bs_price_scalar(S_next, self.K, self.r, tau_next, self.sigma)
            self._V_cached = V_next

        # ------- per-period reward depending on mode ---------------------
        if self.reward_mode == "cf":
            # Pure cash-flow reward: only the realized cash this period
            reward = trade_cashflow + trade_cost
        elif self.reward_mode == "pnl":
            # Stock-only P&L (ignores option MtM; ablation)
            reward = self.H * (S_next - S_now) + trade_cost
        elif self.reward_mode == "apl":
            # Accounting P&L, thesis eq. (8).
            # Stock holding's MtM gain minus change in BS option value minus
            # transaction cost paid this period. Initial setup cost
            # -kappa*|S_0*H_0| is naturally included via trade_cost when t=0.
            reward = self.H * (S_next - S_now) - (V_next - V_now) + trade_cost
        else:
            raise ValueError(f"Unknown reward_mode {self.reward_mode}")

        if done:
            # Liquidate hedge at the terminal price.
            S_T = self.S
            payoff = max(S_T - self.K, 0.0)
            liquidation_cf  = +S_T * self.H
            # Terminal liquidation cost: option-basis uses V_N = payoff
            if self.cost_basis == "stock":
                liquidation_tc = -self.kappa * abs(S_T * self.H)
            else:  # 'option'
                liquidation_tc = -self.kappa * abs(payoff * self.H)
            self.cash += liquidation_cf + liquidation_tc - payoff

            if self.reward_mode == "cf":
                # CF: payoff and liquidation cash flow appear at the end
                reward += liquidation_cf + liquidation_tc - payoff
            elif self.reward_mode == "pnl":
                # pnl: pay the payoff and the terminal trade cost
                reward += liquidation_tc - payoff
            elif self.reward_mode == "apl":
                # apl: option MtM term already covers the payoff (since
                # V_next = phi(S_T) at done). Only the terminal liquidation
                # transaction cost remains.
                reward += liquidation_tc

            self.H = 0.0

        return self._obs(), reward, done, {"S": self.S, "cash": self.cash}

    # -------------------------------------------------------------------------
    @property
    def hedge_cost(self):
        """
        Realized hedging cost of the current/finished episode.

        At t=0 the hedger receives the option premium p (recorded for
        accounting only; not a step reward). All step rewards are realized
        cash flows of the hedging strategy itself. Hence

            terminal_PnL = p + sum(rewards) = p + (terminal_cash - 0)

        and the hedge cost is

            cost = -terminal_PnL = -(p + terminal_cash).

        Negative cost means the hedger ended in profit. For a perfect
        frictionless BS replicating strategy this is approximately zero.
        """
        return -(self.option_premium + self.cash)
