# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: MONTE CARLO ENGINE
# ======================================================================
class MonteCarloSimulator:
    """
    Generates thousands of synthetic portfolio paths by bootstrapping
    from historical daily returns, then computes the full distribution
    of outcomes.

    Method: Block Bootstrap
    Instead of sampling individual days randomly (which would destroy
    autocorrelation in returns), we sample blocks of consecutive days.
    This preserves short-term momentum and volatility clustering that
    real markets exhibit.
    """

    def __init__(self, portfolio_values, n_simulations=5000,
                 block_size=20):
        """
        Parameters
        ----------
        portfolio_values : list — historical daily portfolio values
        n_simulations    : int  — number of paths to generate
        block_size       : int  — days per bootstrap block
                                  (20 = roughly one trading month)
        """
        self.portfolio_values = pd.Series(portfolio_values)
        self.daily_returns    = self.portfolio_values.pct_change()\
                                    .dropna()
        self.n_simulations    = n_simulations
        self.block_size       = block_size
        self.n_days           = len(self.daily_returns)
        logger.info(f"Monte Carlo initialized: "
                   f"{n_simulations} simulations × "
                   f"{self.n_days} days")


    # ======================================================================
    # SECTION 3: GENERATE SYNTHETIC PATHS
    # ======================================================================
    def generate_paths(self, horizon_days=None):
        """
        Generates n_simulations synthetic return paths using
        block bootstrap resampling of historical daily returns.

        Parameters
        ----------
        horizon_days : int — simulation length (default: same as history)

        Returns
        -------
        np.ndarray of shape (n_simulations, horizon_days)
        Each row is one synthetic path of portfolio values,
        starting at the same value as the historical portfolio.
        """
        if horizon_days is None:
            horizon_days = self.n_days

        starting_value = self.portfolio_values.iloc[0]
        returns_array  = self.daily_returns.values
        n_returns      = len(returns_array)
        paths          = np.zeros((self.n_simulations, horizon_days))

        for sim in range(self.n_simulations):
            synthetic_returns = []

            # Sample blocks of consecutive days until we have enough
            while len(synthetic_returns) < horizon_days:
                start = np.random.randint(
                    0, n_returns - self.block_size)
                block = returns_array[start:start + self.block_size]
                synthetic_returns.extend(block)

            synthetic_returns = np.array(
                synthetic_returns[:horizon_days])

            # Compound the returns into a price path
            path = starting_value * np.cumprod(1 + synthetic_returns)
            paths[sim] = path

        logger.info(f"Generated {self.n_simulations} paths "
                   f"× {horizon_days} days")
        return paths


    # ======================================================================
    # SECTION 4: COMPUTE OUTCOME DISTRIBUTION
    # ======================================================================
    def analyze(self, paths):
        """
        Computes the distribution of final outcomes and key metrics
        across all simulated paths.

        Returns a dict of statistics and percentile values.
        """
        starting = self.portfolio_values.iloc[0]
        final_values = paths[:, -1]
        final_returns = (final_values / starting) - 1

        # Max drawdown for each path
        max_drawdowns = []
        for path in paths:
            series = pd.Series(path)
            peak   = series.cummax()
            dd     = ((series - peak) / peak).min()
            max_drawdowns.append(dd * 100)

        max_drawdowns = np.array(max_drawdowns)

        # Sharpe ratio for each path
        sharpes = []
        for path in paths:
            returns = pd.Series(path).pct_change().dropna()
            excess  = returns - 0.05/252
            sh = (excess.mean()/excess.std())*np.sqrt(252) \
                 if excess.std() > 0 else 0.0
            sharpes.append(sh)

        sharpes = np.array(sharpes)

        results = {
            # Return distribution
            "mean_return":     float(np.mean(final_returns) * 100),
            "median_return":   float(np.median(final_returns) * 100),
            "std_return":      float(np.std(final_returns) * 100),
            "p5_return":       float(np.percentile(final_returns, 5) * 100),
            "p25_return":      float(np.percentile(final_returns, 25) * 100),
            "p75_return":      float(np.percentile(final_returns, 75) * 100),
            "p95_return":      float(np.percentile(final_returns, 95) * 100),
            # Risk metrics
            "prob_loss":       float((final_returns < 0).mean() * 100),
            "prob_loss_10pct": float((final_returns < -0.10).mean() * 100),
            "mean_max_dd":     float(np.mean(max_drawdowns)),
            "worst_max_dd":    float(np.min(max_drawdowns)),
            "p5_max_dd":       float(np.percentile(max_drawdowns, 5)),
            # Sharpe distribution
            "mean_sharpe":     float(np.mean(sharpes)),
            "p25_sharpe":      float(np.percentile(sharpes, 25)),
            "p75_sharpe":      float(np.percentile(sharpes, 75)),
            # Raw arrays for plotting
            "_final_returns":  final_returns,
            "_max_drawdowns":  max_drawdowns,
            "_paths":          paths,
        }
        return results


    # ======================================================================
    # SECTION 5: PRINT REPORT
    # ======================================================================
    def print_report(self, results, strategy_name,
                     historical_return=None):
        """Prints a formatted Monte Carlo analysis report."""
        print(f"\n{'='*60}")
        print(f"  MONTE CARLO ANALYSIS — {strategy_name}")
        print(f"  {self.n_simulations:,} simulations · "
              f"{self.n_days} trading days")
        print(f"{'='*60}")

        if historical_return is not None:
            pct = float(np.mean(results["_final_returns"] < historical_return / 100) * 100)
            print(f"\n  Historical return: {historical_return:+.2f}%")
            print(f"  Percentile rank:   {pct:.1f}th percentile")
            print(f"  (better than {pct:.1f}% of simulated paths)")

        print(f"\n  RETURN DISTRIBUTION")
        print(f"  {'Metric':<30} {'Value':>10}")
        print(f"  {'-'*42}")
        print(f"  {'Mean return':<30} "
              f"{results['mean_return']:>+9.2f}%")
        print(f"  {'Median return':<30} "
              f"{results['median_return']:>+9.2f}%")
        print(f"  {'Std deviation':<30} "
              f"{results['std_return']:>9.2f}%")
        print(f"  {'5th percentile (bad case)':<30} "
              f"{results['p5_return']:>+9.2f}%")
        print(f"  {'25th percentile':<30} "
              f"{results['p25_return']:>+9.2f}%")
        print(f"  {'75th percentile':<30} "
              f"{results['p75_return']:>+9.2f}%")
        print(f"  {'95th percentile (good case)':<30} "
              f"{results['p95_return']:>+9.2f}%")

        print(f"\n  RISK METRICS")
        print(f"  {'Metric':<30} {'Value':>10}")
        print(f"  {'-'*42}")
        print(f"  {'Probability of any loss':<30} "
              f"{results['prob_loss']:>9.1f}%")
        print(f"  {'Probability of >10% loss':<30} "
              f"{results['prob_loss_10pct']:>9.1f}%")
        print(f"  {'Mean max drawdown':<30} "
              f"{results['mean_max_dd']:>9.2f}%")
        print(f"  {'Worst max drawdown':<30} "
              f"{results['worst_max_dd']:>9.2f}%")
        print(f"  {'5th percentile max DD':<30} "
              f"{results['p5_max_dd']:>9.2f}%")

        print(f"\n  SHARPE DISTRIBUTION")
        print(f"  {'Mean Sharpe':<30} "
              f"{results['mean_sharpe']:>9.3f}")
        print(f"  {'25th–75th percentile':<30} "
              f"  {results['p25_sharpe']:.3f} – "
              f"{results['p75_sharpe']:.3f}")
        print(f"{'='*60}")