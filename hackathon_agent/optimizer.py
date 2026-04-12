"""
Hackathon@IITD2026 — Portfolio Optimizer
======================================
Mean-Variance Optimisation (MVO) using cvxpy.

Maximises:  w^T * mu - gamma * w^T * Sigma * w
Subject to:
  - sum(w) == 1              (fully invested)
  - w_i >= 0                 (long-only)
  - w_i >= MIN_WEIGHT if held (minimum position size)
  - count(w_i > 0) <= MAX_HOLDINGS  (cardinality constraint)
  - sum(|w_i - w_prev_i|) <= TURNOVER_BUDGET  (turnover constraint)

Falls back to equal-weight if cvxpy is unavailable or optimization fails.

Extend this file to improve your Sharpe:
  - Tune gamma (risk aversion)
  - Add sector concentration limits
  - Add individual position max weights
  - Use a better covariance estimator (Ledoit-Wolf shrinkage, etc.)
"""

import logging
import math
from typing import Optional

import numpy as np

log = logging.getLogger("optimizer")

# Try to import cvxpy — fall back gracefully if not available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    log.warning("cvxpy not available — falling back to equal-weight optimizer")
    CVXPY_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
GAMMA            = 1.0    # risk aversion parameter (higher = more conservative)
MIN_HISTORY      = 5      # minimum ticks of price history required
REGULARISATION   = 1e-4   # ridge regularisation on covariance diagonal
MAX_SINGLE_WEIGHT = 0.15  # optional: max 15% in any single position


class Optimizer:
    def __init__(
        self,
        max_holdings: int = 30,
        min_weight: float = 0.005,
        gamma: float = GAMMA,
        max_single_weight: float = MAX_SINGLE_WEIGHT,
        max_return_window: Optional[int] = 30,
    ):
        self.max_holdings       = max_holdings
        self.min_weight         = min_weight
        self.gamma              = gamma
        self.max_single_weight  = max_single_weight
        self.max_return_window  = max_return_window

    def optimise(
        self,
        tickers: list[str],
        expected_returns: dict[str, float],
        price_history: dict[str, list[float]],
        current_weights: dict[str, float],
        turnover_budget: float = 0.30,
    ) -> dict[str, float]:
        """
        Run the optimizer and return target portfolio weights {ticker: weight}.

        Parameters
        ----------
        tickers          : list of all available tickers this tick
        expected_returns : {ticker: expected log return}
        price_history    : {ticker: [price_t-N, ..., price_t]}
        current_weights  : {ticker: current weight in portfolio}
        turnover_budget  : remaining fraction of portfolio that can be traded

        Returns
        -------
        {ticker: target_weight}  — weights sum to 1, all >= 0
        """
        # Filter to tickers with sufficient price history
        eligible = [
            t for t in tickers
            if len(price_history.get(t, [])) >= MIN_HISTORY
        ]

        if len(eligible) < 2:
            log.warning("Not enough tickers with price history — returning equal weight")
            return self._equal_weight(eligible or tickers[:self.max_holdings])

        mu    = self._build_mu(eligible, expected_returns)
        Sigma = self._build_covariance(eligible, price_history)
        w_prev = np.array([current_weights.get(t, 0.0) for t in eligible])

        if CVXPY_AVAILABLE:
            weights = self._cvxpy_optimise(mu, Sigma, w_prev, turnover_budget, len(eligible))
        else:
            weights = self._greedy_optimise(mu, Sigma, eligible)

        # Map back to tickers and apply cardinality trim
        result = dict(zip(eligible, weights))
        result = self._apply_cardinality(result)
        result = self._normalise(result)
        return result

    # ── Build expected return vector ───────────────────────────────────────────
    def _build_mu(self, tickers: list[str], expected_returns: dict[str, float]) -> np.ndarray:
        return np.array([expected_returns.get(t, 0.0) for t in tickers])

    # ── Build covariance matrix from log returns ───────────────────────────────
    def _build_covariance(
        self, tickers: list[str], price_history: dict[str, list[float]]
    ) -> np.ndarray:
        n = len(tickers)
        returns_matrix = []

        for t in tickers:
            prices = price_history[t]
            log_rets = [
                math.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices))
                if prices[i - 1] > 0
            ]
            returns_matrix.append(log_rets)

        # Align lengths (use the shortest series)
        min_len = min(len(r) for r in returns_matrix)
        if min_len < 2:
            return np.eye(n) * 1e-4

        R = np.array([r[-min_len:] for r in returns_matrix])  # shape: (n_tickers, T)

        if self.max_return_window is not None and R.shape[1] > self.max_return_window:
            R = R[:, -int(self.max_return_window) :]

        # Sample covariance
        Sigma = np.cov(R)

        # Ledoit-Wolf-style diagonal regularisation (shrinkage toward identity)
        avg_var = np.trace(Sigma) / n
        Sigma   = Sigma + REGULARISATION * avg_var * np.eye(n)

        return Sigma

    # ── cvxpy solver ──────────────────────────────────────────────────────────
    def _cvxpy_optimise(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        w_prev: np.ndarray,
        turnover_budget: float,
        n: int,
    ) -> np.ndarray:
        w = cp.Variable(n, nonneg=True)

        objective = cp.Maximize(mu @ w - self.gamma * cp.quad_form(w, Sigma))

        constraints = [
            cp.sum(w) == 1,                              # fully invested
            w <= self.max_single_weight,                 # max single position
            cp.sum(cp.abs(w - w_prev)) <= turnover_budget,  # turnover limit
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as exc:
            log.warning(f"OSQP failed ({exc}), trying SCS")
            try:
                prob.solve(solver=cp.SCS, verbose=False)
            except Exception as exc2:
                log.warning(f"SCS also failed ({exc2}) — falling back to equal weight")
                return self._equal_weight_array(n)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            log.warning(f"Optimizer status: {prob.status} — falling back to equal weight")
            return self._equal_weight_array(n)

        weights = np.clip(w.value, 0, None)

        # Zero out tiny weights (below min_weight)
        weights[weights < self.min_weight * 0.5] = 0.0

        total = weights.sum()
        if total < 1e-8:
            return self._equal_weight_array(n)

        return weights / total

    # ── Greedy fallback (no cvxpy) ─────────────────────────────────────────────
    def _greedy_optimise(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        tickers: list[str],
    ) -> np.ndarray:
        """
        Simple greedy: rank by Sharpe-like score (mu / sigma), take top K.
        Not optimal but respects cardinality and is fast.
        """
        n = len(tickers)
        sigmas = np.sqrt(np.diag(Sigma))
        sigmas = np.where(sigmas < 1e-8, 1e-8, sigmas)
        scores = mu / sigmas

        k = min(self.max_holdings, n)
        top_k = np.argsort(scores)[-k:]

        weights = np.zeros(n)
        positive_scores = np.maximum(scores[top_k], 0)
        total = positive_scores.sum()

        if total < 1e-8:
            weights[top_k] = 1.0 / k
        else:
            weights[top_k] = positive_scores / total

        return weights

    # ── Cardinality enforcement ────────────────────────────────────────────────
    def _apply_cardinality(self, weights: dict[str, float]) -> dict[str, float]:
        """Keep only the top MAX_HOLDINGS positions by weight."""
        if len(weights) <= self.max_holdings:
            return weights
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        kept = dict(sorted_items[: self.max_holdings])
        return kept

    # ── Normalise weights to sum to 1 ─────────────────────────────────────────
    def _normalise(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total < 1e-8:
            tickers = list(weights.keys())
            return {t: 1.0 / len(tickers) for t in tickers}
        return {t: w / total for t, w in weights.items()}

    # ── Equal weight helpers ───────────────────────────────────────────────────
    def _equal_weight(self, tickers: list[str]) -> dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    def _equal_weight_array(self, n: int) -> np.ndarray:
        return np.full(n, 1.0 / n)
