"""
Black-Scholes helpers — delta hedge ratio and option pricing.

These are the textbook formulas. bs_delta_vec is vectorised so you can
pass an array of 5,000 stock prices and get 5,000 deltas at once.
"""

import numpy as np
from scipy.special import ndtr

from config import K, sigma, S0, T


def bs_delta_vec(S, tau):
    """
    Vectorised Black-Scholes delta for a European call.

    Parameters
    ----------
    S   : float or np.ndarray — current stock price(s)
    tau : float — time remaining to maturity (in years)

    Returns
    -------
    delta : same shape as S — hedge ratio(s) in [0, 1]
    """
    safe_tau = np.maximum(tau, 1e-8)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * safe_tau) / (sigma * np.sqrt(safe_tau))
    return np.where(tau <= 0, (S > K).astype(float), ndtr(d1))


def bs_price_scalar(S, tau):
    """
    Black-Scholes call price (scalar inputs).

    Parameters
    ----------
    S   : float — current stock price
    tau : float — time remaining to maturity

    Returns
    -------
    price : float — fair call option price
    """
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return float(S * ndtr(d1) - K * ndtr(d2))


# Initial option premium collected by the writer
V0 = bs_price_scalar(S0, T)
