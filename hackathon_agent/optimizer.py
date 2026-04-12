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
GAMMA            = 0.5    # low risk aversion — we trust forward-looking signals
MIN_HISTORY      = 5      # minimum ticks of price history required
REGULARISATION   = 1e-2   # ridge regularisation on covariance diagonal
MAX_SINGLE_WEIGHT = 0.05  # max 5% per position — TC006 requires E007 ≤ 5%


class Optimizer:
    def __init__(
        self,
        max_holdings: int = 30,
        min_weight: float = 0.005,
        gamma: float = GAMMA,
        max_single_weight: float = MAX_SINGLE_WEIGHT,
    ):
        self.max_holdings       = max_holdings
        self.min_weight         = min_weight
        self.gamma              = gamma
        self.max_single_weight  = max_single_weight

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
            log.warning("Not enough tickers with price history — returning current weights")
            return {t: current_weights.get(t, 0.0) for t in (eligible or tickers[:self.max_holdings]) if current_weights.get(t, 0.0) > 0}

        mu    = self._build_mu(eligible, expected_returns)
        Sigma = self._build_covariance(eligible, price_history)
        w_prev = np.array([current_weights.get(t, 0.0) for t in eligible])

        if CVXPY_AVAILABLE and turnover_budget > 0.02:
            # Only use cvxpy when there's meaningful turnover budget
            # Reduce to top candidates by signal for numerical stability
            n = len(eligible)
            if n > 30:
                # Keep currently held + top signal tickers
                held_idx = set(i for i in range(n) if w_prev[i] > 0)
                signal_rank = np.argsort(mu)[::-1]
                keep = set()
                keep.update(held_idx)
                for idx in signal_rank:
                    if len(keep) >= 30:
                        break
                    keep.add(idx)
                keep = sorted(keep)
                eligible_sub = [eligible[i] for i in keep]
                mu_sub = mu[keep]
                Sigma_sub = Sigma[np.ix_(keep, keep)]
                w_prev_sub = w_prev[keep]
                weights_sub = self._cvxpy_optimise(mu_sub, Sigma_sub, w_prev_sub, turnover_budget, len(keep))
                weights = np.zeros(len(eligible))
                for j, idx in enumerate(keep):
                    weights[idx] = weights_sub[j]
            else:
                weights = self._cvxpy_optimise(mu, Sigma, w_prev, turnover_budget, n)
        elif turnover_budget > 0.005:
            weights = self._greedy_with_turnover(mu, w_prev, turnover_budget, len(eligible))
        else:
            weights = w_prev  # hold current position

        # Map back to tickers and apply cardinality trim
        result = dict(zip(eligible, weights))
        result = self._apply_cardinality(result)
        # Only normalise if weights sum to > 0; preserve sub-1.0 sums (cash held)
        total = sum(result.values())
        if total < 1e-8:
            return self._equal_weight(eligible[:self.max_holdings])
        # Don't force sum to 1 — allow cash position
        return {t: max(0.0, w) for t, w in result.items() if w > self.min_weight * 0.5}

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

        # Sample covariance
        Sigma = np.cov(R)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])

        # Ledoit-Wolf-style shrinkage toward diagonal
        avg_var = np.trace(Sigma) / n
        shrinkage = max(REGULARISATION, min(0.5, (n - min_len) / n))
        Sigma = (1 - shrinkage) * Sigma + shrinkage * avg_var * np.eye(n)

        # Ensure PSD: clip negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-6)
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Symmetrise
        Sigma = (Sigma + Sigma.T) / 2

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

        objective = cp.Maximize(mu @ w - self.gamma * cp.quad_form(w, cp.psd_wrap(Sigma)))

        # Compute feasible investment level given turnover budget
        current_invested = float(w_prev.sum())
        max_invest = min(1.0, current_invested + turnover_budget)
        log.debug(f"Optimizer: n={n}, invested={current_invested:.3f}, budget={turnover_budget:.4f}")

        # Use auxiliary variable for L1 turnover constraint (SOCP-friendly)
        t = cp.Variable(n, nonneg=True)
        constraints = [
            cp.sum(w) <= max_invest,
            w <= self.max_single_weight,
            t >= w - w_prev,
            t >= w_prev - w,
            cp.sum(t) <= turnover_budget,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except Exception as exc:
            log.warning(f"SCS failed ({exc}), trying OSQP")
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception as exc2:
                log.warning(f"OSQP also failed ({exc2}) — holding current weights")
                return w_prev

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            log.info(f"Optimizer status: {prob.status} — using greedy fallback")
            return self._greedy_with_turnover(mu, w_prev, turnover_budget, n)

        weights = np.clip(w.value, 0, None)

        # Zero out tiny weights (below min_weight)
        weights[weights < self.min_weight * 0.5] = 0.0

        total = weights.sum()
        if total < 1e-8:
            return w_prev  # hold current position if optimizer finds nothing

        # DO NOT normalize to sum=1 — respect the turnover constraint
        # The optimizer already found weights within max_invest bound
        return weights

    # ── Greedy with turnover constraint ─────────────────────────────────────────
    def _greedy_with_turnover(
        self,
        mu: np.ndarray,
        w_prev: np.ndarray,
        turnover_budget: float,
        n: int,
    ) -> np.ndarray:
        """
        Start from current weights, nudge toward best-signal tickers
        within turnover budget.
        """
        weights = w_prev.copy()
        if turnover_budget <= 0.001:
            return weights

        # Find tickers to increase (positive mu) and decrease (negative mu or small mu)
        adjustments = []
        for i in range(n):
            adjustments.append((mu[i], i))
        adjustments.sort(reverse=True)

        budget_remaining = turnover_budget * 0.9  # safety margin
        step = min(0.01, budget_remaining / 4)  # small incremental steps

        # Increase top signal tickers
        for _, i in adjustments[:self.max_holdings]:
            if budget_remaining <= step:
                break
            if mu[i] > 0 and weights[i] < self.max_single_weight:
                add = min(step, self.max_single_weight - weights[i], budget_remaining / 2)
                weights[i] += add
                budget_remaining -= add

        # Decrease worst signal tickers to free up weight
        for _, i in reversed(adjustments):
            if budget_remaining <= 0:
                break
            if mu[i] < 0 and weights[i] > 0:
                remove = min(weights[i], step, budget_remaining / 2)
                weights[i] -= remove
                budget_remaining -= remove

        # Enforce cardinality
        nonzero = np.nonzero(weights > self.min_weight * 0.5)[0]
        if len(nonzero) > self.max_holdings:
            by_weight = sorted(nonzero, key=lambda i: weights[i])
            for i in by_weight[:len(nonzero) - self.max_holdings]:
                weights[i] = 0.0

        weights = np.clip(weights, 0, None)
        return weights

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
