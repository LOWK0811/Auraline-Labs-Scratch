# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: BLACK-SCHOLES PRICER
# ======================================================================
class BlackScholes:
    """
    Black-Scholes option pricing model.

    Prices European call and put options and computes
    the Greeks — sensitivities of the option price to
    each input parameter.

    Greeks:
    - Delta: how much the option price changes per $1 move in stock
    - Gamma: how much Delta itself changes per $1 move in stock
    - Theta: how much the option loses per day (time decay)
    - Vega:  how much the option price changes per 1% vol move
    - Rho:   how much the option price changes per 1% rate move
    """

    def __init__(self, S, K, T, r, sigma):
        """
        Parameters
        ----------
        S     : float — current stock price
        K     : float — strike price
        T     : float — time to expiry in years (e.g. 0.25 = 3 months)
        r     : float — risk-free rate (e.g. 0.05 = 5%)
        sigma : float — annualized volatility (e.g. 0.25 = 25%)
        """
        self.S     = S
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self._compute_d1_d2()

    def _compute_d1_d2(self):
        """Computes d1 and d2 — the core of the Black-Scholes formula."""
        if self.T <= 0 or self.sigma <= 0:
            self.d1 = float("inf")
            self.d2 = float("inf")
            return
        self.d1 = (
            np.log(self.S / self.K) +
            (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)


    # ======================================================================
    # SECTION 3: OPTION PRICES
    # ======================================================================
    def call_price(self):
        """
        Price of a European call option.
        Interpretation: the fair premium to pay for the right
        to buy the stock at strike K at expiry.
        """
        if self.T <= 0:
            return max(self.S - self.K, 0)
        return (self.S * norm.cdf(self.d1) -
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self):
        """
        Price of a European put option.
        Derived from call price via put-call parity:
        C - P = S - K·e^(-rT)
        This relationship must hold to prevent arbitrage.
        """
        if self.T <= 0:
            return max(self.K - self.S, 0)
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) -
                self.S * norm.cdf(-self.d1))


    # ======================================================================
    # SECTION 4: THE GREEKS
    # ======================================================================
    def delta(self, option_type="call"):
        """
        Delta: rate of change of option price with respect to stock price.
        Call delta ranges from 0 to 1.
        Put delta ranges from -1 to 0.
        An at-the-money option has delta ≈ 0.5 (call) or -0.5 (put).
        """
        if option_type == "call":
            return norm.cdf(self.d1)
        return norm.cdf(self.d1) - 1

    def gamma(self):
        """
        Gamma: rate of change of delta with respect to stock price.
        Same for calls and puts. Highest when near expiry and at-the-money.
        High gamma = delta changes rapidly = more hedging required.
        """
        return (norm.pdf(self.d1) /
                (self.S * self.sigma * np.sqrt(self.T)))

    def theta(self, option_type="call"):
        """
        Theta: time decay — how much the option loses per day.
        Almost always negative (options lose value as time passes).
        Expressed per calendar day (divided by 365).
        """
        term1 = (-(self.S * norm.pdf(self.d1) * self.sigma) /
                  (2 * np.sqrt(self.T)))
        if option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * \
                     norm.cdf(self.d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * \
                    norm.cdf(-self.d2)
        return (term1 + term2) / 365  # per calendar day

    def vega(self):
        """
        Vega: sensitivity to volatility.
        How much the option price changes per 1% increase in volatility.
        Same for calls and puts.
        """
        return (self.S * norm.pdf(self.d1) *
                np.sqrt(self.T)) / 100  # per 1% vol move

    def rho(self, option_type="call"):
        """
        Rho: sensitivity to the risk-free rate.
        How much the option price changes per 1% increase in rates.
        """
        if option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) *
                    norm.cdf(self.d2)) / 100
        return (-self.K * self.T * np.exp(-self.r * self.T) *
                norm.cdf(-self.d2)) / 100

    def all_greeks(self, option_type="call"):
        """Returns all Greeks as a dictionary."""
        return {
            "Delta": round(self.delta(option_type), 4),
            "Gamma": round(self.gamma(), 4),
            "Theta": round(self.theta(option_type), 4),
            "Vega":  round(self.vega(), 4),
            "Rho":   round(self.rho(option_type), 4),
        }


    # ======================================================================
    # SECTION 5: IMPLIED VOLATILITY
    # ======================================================================
    def implied_volatility(self, market_price, option_type="call"):
        """
        Implied volatility: the volatility that makes the Black-Scholes
        price match the observed market price.

        This inverts the Black-Scholes formula numerically using
        Brent's method — a reliable root-finding algorithm.

        IV is what options traders actually trade. When IV is high,
        the market is pricing in large future moves. When IV is low,
        the market expects calm. IV differs from historical vol
        because it reflects forward-looking expectations.
        """
        def objective(sigma):
            bs = BlackScholes(self.S, self.K, self.T, self.r, sigma)
            if option_type == "call":
                return bs.call_price() - market_price
            return bs.put_price() - market_price

        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=1e-6)
            return round(iv, 4)
        except ValueError:
            return None


# ======================================================================
# SECTION 6: OPTIONS CHAIN BUILDER
# ======================================================================
def build_options_chain(S, T, r, sigma, n_strikes=11):
    """
    Builds a complete options chain — a table of calls and puts
    across a range of strike prices centered on the current
    stock price. This is what you see on a real options trading
    platform like Robinhood, IBKR, or Bloomberg.

    Parameters
    ----------
    S        : float — current stock price
    T        : float — time to expiry in years
    r        : float — risk-free rate
    sigma    : float — annualized volatility
    n_strikes: int   — number of strike prices to generate
    """
    strikes = np.linspace(S * 0.7, S * 1.3, n_strikes)
    rows    = []

    for K in strikes:
        bs   = BlackScholes(S, K, T, r, sigma)
        call = bs.call_price()
        put  = bs.put_price()
        moneyness = "ITM" if S > K else ("ATM" if abs(S-K) < 2 else "OTM")

        rows.append({
            "Strike":        round(K, 2),
            "Moneyness":     moneyness,
            "Call Price":    round(call, 2),
            "Call Delta":    round(bs.delta("call"), 3),
            "Call Theta":    round(bs.theta("call"), 3),
            "Put Price":     round(put, 2),
            "Put Delta":     round(bs.delta("put"), 3),
            "Put Theta":     round(bs.theta("put"), 3),
            "Gamma":         round(bs.gamma(), 4),
            "Vega":          round(bs.vega(), 4),
        })

    return pd.DataFrame(rows)