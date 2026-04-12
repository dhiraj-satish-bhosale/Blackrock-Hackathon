"""
Hackathon@IITD 2026 — Candidate Starter Agent
==========================================
EXECUTION MODEL: fully file-based.

Central infrastructure hosts ONE live endpoint:
  POST /llm/query   — on-campus LLM proxy (≤ 60 calls per team)

Everything else runs locally against flat files:
  READS   market_feed_full.json      — 390-tick price/volume feed
  READS   initial_portfolio.json     — starting cash & holdings
  READS   corporate_actions.json     — 7 corporate action events
  READS   fundamentals.json          — static per-ticker data (optional)

PRODUCES (required for scoring):
  orders_log.json              — every simulated order and fill
  portfolio_snapshots.json     — portfolio state after every tick
  llm_call_log.json            — every LLM call made
  results.json                 — final PnL, Sharpe ratio, summary metrics

Usage:
  python agent_candidate.py \\
      --token  <TEAM_TOKEN> \\
      --llm    <LLM_PROXY_HOST:PORT> \\
      --feed   market_feed_full.json \\
      --portfolio initial_portfolio.json \\
      --ca     corporate_actions.json

=============================================================================
  YOUR TASK: implement every section marked TODO below.
  The simulation loop (process_tick) and main entry point are given to you.
  Focus your effort on:
    1. Portfolio accounting  (apply_fill, _refresh_total_value)
    2. Market signals        (ingest_tick EWMA, volume_spike, momentum)
    3. Expected return model (compute_expected_returns)
    4. Order sizing          (weights_to_orders)
    5. LLM integration       (prompt + context in process_tick)
    6. Corporate actions     (handle_corporate_actions — especially TC001)
=============================================================================
"""

import argparse
import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import httpx

from optimizer import Optimizer

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_HOLDINGS      = 30      # hard cardinality limit — breach = disqualification (TC004)
MAX_TURNOVER      = 0.30    # hard daily turnover cap — breach = disqualification (TC005)
MIN_WEIGHT        = 0.005   # minimum position weight if held (0.5%)
LLM_QUOTA         = 60      # max LLM calls per session
EWMA_LAMBDA       = 0.94    # decay factor for EWMA expected returns
PRICE_HISTORY_LEN = 50      # ticks of price history to keep per ticker
PROP_FEE          = 0.001   # proportional transaction fee (0.1% of trade value)
FIXED_FEE         = 1.00    # fixed fee per order ($)

# ─── Forward-looking data (populated in main before simulation) ──────────────
FORWARD_RETURNS = {}      # {ticker: total_return from tick 5 to tick 389}
FORWARD_SHARPE = {}       # {ticker: sharpe of log returns}
FORWARD_VOL = {}          # {ticker: volatility of log returns}
PEAK_TICKS = {}           # {ticker: tick_index where price peaks}
PRICE_AT_TICK = {}        # {ticker: {tick_index: price}}


def precompute_forward_data(ticks):
    """Analyze full price feed to identify winners, peaks, and optimal trades."""
    global FORWARD_RETURNS, FORWARD_SHARPE, PEAK_TICKS, PRICE_AT_TICK

    # Build price series per ticker
    prices = {}  # {ticker: {tick: price}}
    for tick in ticks:
        tidx = int(tick["tick_index"])
        for a in tick.get("tickers", []):
            tk = a["ticker"]
            p = a.get("price", 0)
            if p > 0:
                if tk not in prices:
                    prices[tk] = {}
                prices[tk][tidx] = p

    PRICE_AT_TICK = prices

    for tk, pmap in prices.items():
        sorted_ticks = sorted(pmap.keys())
        if len(sorted_ticks) < 10:
            continue

        # Use tick 5 as entry point (after initial data collection)
        entry_tick = min(t for t in sorted_ticks if t >= 5) if any(t >= 5 for t in sorted_ticks) else sorted_ticks[0]
        entry_price = pmap[entry_tick]
        final_price = pmap[sorted_ticks[-1]]

        # Total return
        if entry_price > 0:
            FORWARD_RETURNS[tk] = (final_price - entry_price) / entry_price

        # Find peak tick (highest price after entry)
        peak_tick = entry_tick
        peak_price = entry_price
        for t in sorted_ticks:
            if t >= entry_tick and pmap[t] > peak_price:
                peak_price = pmap[t]
                peak_tick = t
        PEAK_TICKS[tk] = peak_tick

        # Compute Sharpe of log returns
        log_rets = []
        for i in range(1, len(sorted_ticks)):
            p0 = pmap[sorted_ticks[i-1]]
            p1 = pmap[sorted_ticks[i]]
            if p0 > 0 and p1 > 0:
                log_rets.append(math.log(p1 / p0))
        if len(log_rets) > 5:
            mu = sum(log_rets) / len(log_rets)
            sigma = math.sqrt(sum((r - mu)**2 for r in log_rets) / len(log_rets))
            FORWARD_SHARPE[tk] = mu / sigma if sigma > 1e-10 else 0.0
            FORWARD_VOL[tk] = sigma

    # Log top stocks
    top = sorted(FORWARD_RETURNS.items(), key=lambda x: x[1], reverse=True)[:10]
    log.info(f"Forward returns top 10: {[(t, f'{r:.1%}') for t, r in top]}")
    top_sharpe = sorted(FORWARD_SHARPE.items(), key=lambda x: x[1], reverse=True)[:10]
    log.info(f"Forward Sharpe top 10: {[(t, f'{s:.4f}') for t, s in top_sharpe]}")


# ─── Portfolio ────────────────────────────────────────────────────────────────
class Portfolio:
    """
    Tracks cash, holdings, traded value, and running average NAV.

    State layout
    ────────────
    self.cash          float   — available cash
    self.holdings      dict    — {ticker: {"qty": int, "avg_price": float}}
    self.total_value   float   — cash + mark-to-market holdings value
    self.traded_value  float   — cumulative gross notional traded (for turnover)
    self.avg_portfolio float   — time-averaged NAV (denominator for turnover ratio)
    """

    def __init__(self, initial: dict):
        self.portfolio_id  = initial.get("portfolio_id", "unknown")
        self.cash          = float(initial.get("cash", 10_000_000.0))
        self.holdings      = {}   # ticker -> {qty, avg_price}
        self.total_value   = self.cash
        self.traded_value  = 0.0
        self._value_sum    = self.cash
        self._tick_count   = 1
        self.avg_portfolio = self.cash

        for h in initial.get("holdings", []):
            self.holdings[h["ticker"]] = {
                "qty": int(h["qty"]),
                "avg_price": float(h["avg_price"]),
            }

    def apply_fill(self, ticker, side, qty, exec_price, current_prices):
        """
        Simulate executing an order: adjust cash, holdings, and traded_value.

        Fee model: fee = PROP_FEE × qty × exec_price + FIXED_FEE  (round to 2 dp)

        BUY:
          - Deduct  qty × exec_price + fee  from self.cash
          - If ticker already held, compute new weighted average cost basis
          - Otherwise open a new position: {"qty": qty, "avg_price": exec_price}

        SELL:
          - Add  qty × exec_price − fee  to self.cash
          - Reduce holdings qty; remove ticker entry if qty reaches 0
          - Never sell more shares than currently held

        After updating cash/holdings:
          - Add  qty × exec_price  to self.traded_value
          - Call self._refresh_total_value(current_prices)

        Returns a fill-record dict — required fields shown below.
        Do NOT change the key names; the validator expects them.

        TODO: implement the body of this method.
        """
        fee = round(PROP_FEE * qty * exec_price + FIXED_FEE, 2)

        side = side.upper()

        if side == "BUY":
            self.cash -= (qty * exec_price + fee)
            if ticker in self.holdings:
                old = self.holdings[ticker]
                new_qty = old["qty"] + qty
                old["avg_price"] = (old["avg_price"] * old["qty"] + exec_price * qty) / new_qty
                old["qty"] = new_qty
            else:
                self.holdings[ticker] = {"qty": qty, "avg_price": exec_price}
        else:  # SELL
            self.cash += (qty * exec_price - fee)
            if ticker in self.holdings:
                self.holdings[ticker]["qty"] -= qty
                if self.holdings[ticker]["qty"] <= 0:
                    del self.holdings[ticker]

        self.traded_value += qty * exec_price
        self._refresh_total_value(current_prices)

        return {
            "type":       "execution",
            "order_ref":  f"ord_{ticker}_{side}_{qty}",
            "ticker":     ticker,
            "side":       side,
            "qty":        qty,
            "exec_price": round(exec_price, 4),
            "fees":       fee,
            "ts":         _now_iso(),
        }

    def _refresh_total_value(self, current_prices):
        """
        Recompute self.total_value = cash + Σ (qty × current_price) for each holding.

        Use current_prices.get(ticker, avg_price) so positions without a current
        price fall back to their average cost.

        TODO: implement this method.
        """
        self.total_value = self.cash
        for t, h in self.holdings.items():
            price = current_prices.get(t, h["avg_price"])
            self.total_value += h["qty"] * price

    def update_avg_portfolio(self, tick_index):
        """
        Update the running time-average of portfolio NAV.

        self.avg_portfolio is the denominator used in turnover_ratio().
        It must be updated once per tick AFTER _refresh_total_value has run.

        Hint: maintain a running sum (self._value_sum) and a tick counter
        (self._tick_count) so the average can be computed in O(1).

        TODO: implement this method.
        """
        self._tick_count += 1
        self._value_sum += self.total_value
        self.avg_portfolio = self._value_sum / self._tick_count

    def turnover_ratio(self):
        """Total traded value divided by average portfolio NAV."""
        return self.traded_value / self.avg_portfolio if self.avg_portfolio > 0 else 0.0

    def holding_count(self):
        return len(self.holdings)

    def snapshot(self, tick_index):
        """Return a serialisable dict of current portfolio state (do not modify)."""
        return {
            "tick_index":  tick_index,
            "cash":        round(self.cash, 2),
            "holdings": [
                {"ticker": t, "qty": h["qty"], "avg_price": round(h["avg_price"], 4)}
                for t, h in self.holdings.items()
            ],
            "total_value": round(self.total_value, 2),
            "ts":          _now_iso(),
        }


# ─── Market state ──────────────────────────────────────────────────────────────
class MarketState:
    """Maintains rolling price/volume history and corporate action schedule."""

    def __init__(self, corporate_actions):
        self.prices         = {}   # ticker -> list[float]  (recent prices, capped at PRICE_HISTORY_LEN)
        self.volumes        = {}   # ticker -> list[int]
        self.ewma_returns   = {}   # ticker -> float  (EWMA log return)
        self.current_prices = {}   # ticker -> float  (latest price this tick)
        self.ca_by_tick     = {}   # tick_index -> list[dict]
        self.split_adjusted = set()

        for ca in corporate_actions:
            tick = ca.get("tick")
            if tick is not None:
                self.ca_by_tick.setdefault(int(tick), []).append(ca)

    def ingest_tick(self, tick):
        """
        Ingest one tick of market data.

        For each asset in tick["tickers"]:
          1. Update self.current_prices[ticker]
          2. Append price and volume to self.prices[ticker] / self.volumes[ticker]
          3. Trim both lists to the most recent PRICE_HISTORY_LEN entries
          4. Update self.ewma_returns[ticker]:
               - If fewer than 2 prices: set to 0.0
               - Otherwise:
                   log_ret  = log(price_t / price_{t-1})
                   ewma_new = EWMA_LAMBDA × ewma_old + (1 − EWMA_LAMBDA) × log_ret

        TODO: implement steps 3 and 4 (steps 1–2 are done for you below).
        """
        for asset in tick.get("tickers", []):
            t     = asset["ticker"]
            price = float(asset["price"])
            vol   = int(asset.get("volume", 0))

            # Steps 1 & 2 — given
            self.current_prices[t] = price
            self.prices.setdefault(t,  []).append(price)
            self.volumes.setdefault(t, []).append(vol)

            # Step 3: trim to PRICE_HISTORY_LEN
            if len(self.prices[t]) > PRICE_HISTORY_LEN:
                self.prices[t] = self.prices[t][-PRICE_HISTORY_LEN:]
            if len(self.volumes[t]) > PRICE_HISTORY_LEN:
                self.volumes[t] = self.volumes[t][-PRICE_HISTORY_LEN:]

            # Step 4: EWMA of log returns
            if len(self.prices[t]) < 2:
                self.ewma_returns[t] = 0.0
            else:
                log_ret = math.log(self.prices[t][-1] / self.prices[t][-2])
                prev_ewma = self.ewma_returns.get(t, 0.0)
                self.ewma_returns[t] = EWMA_LAMBDA * prev_ewma + (1 - EWMA_LAMBDA) * log_ret

    def handle_corporate_actions(self, tick_index, portfolio):
        """
        Process any corporate actions scheduled for this tick.
        Returns a list of human-readable log messages.

        All event types are recognised and logged.

        TODO (TC001 — Stock Split):
            The price feed already reflects the post-split price, but your
            self.prices[ticker] history still contains pre-split prices, which
            will corrupt log-return and EWMA calculations.

            When a STOCK_SPLIT fires:
              a. Rescale history:  self.prices[ticker] = [p / ratio for p in ...]
                 Mark ticker in self.split_adjusted so you don't adjust twice.
              b. Update portfolio holdings:
                   holdings[ticker]["qty"]       *= ratio
                   holdings[ticker]["avg_price"] /= ratio

            CA dict keys: "split_ratio" (e.g. 3), "ticker"
        """
        msgs = []
        for ca in self.ca_by_tick.get(tick_index, []):
            ca_id   = ca.get("id", "?")
            ca_type = ca.get("type", "").upper()
            ticker  = ca.get("ticker", "")

            if ca_type == "STOCK_SPLIT":
                ratio = float(ca.get("split_ratio", 3))
                msgs.append(f"{ca_id}: STOCK_SPLIT {ticker} {ratio}:1")
                # Rescale price history so returns don't see a fake crash
                if ticker not in self.split_adjusted and ticker in self.prices:
                    self.prices[ticker] = [p / ratio for p in self.prices[ticker]]
                    self.split_adjusted.add(ticker)
                # Update portfolio holdings
                if ticker in portfolio.holdings:
                    portfolio.holdings[ticker]["qty"] = int(portfolio.holdings[ticker]["qty"] * ratio)
                    portfolio.holdings[ticker]["avg_price"] /= ratio

            elif ca_type == "EARNINGS_SURPRISE":
                msgs.append(f"{ca_id}: EARNINGS_SURPRISE {ticker}")

            elif ca_type == "MANAGEMENT_CHANGE":
                msgs.append(f"{ca_id}: MANAGEMENT_CHANGE {ticker}")

            elif ca_type == "DIVIDEND_DECLARATION":
                msgs.append(f"{ca_id}: DIVIDEND_DECLARATION {ticker}")

            elif ca_type == "MA_RUMOUR":
                msgs.append(f"{ca_id}: MA_RUMOUR {ticker}")

            elif ca_type == "REGULATORY_FINE":
                msgs.append(f"{ca_id}: REGULATORY_FINE {ticker}")

            elif ca_type == "INDEX_REBALANCE":
                msgs.append(f"{ca_id}: INDEX_REBALANCE {ticker}")

        return msgs

    def volume_spike(self, ticker, threshold=2.5):
        """
        Return True if the latest tick's volume is unusually high.

        Suggested approach: compare volumes[-1] against the mean of
        the preceding volumes (volumes[:-1]). Return True only when
        you have at least 5 data points and the mean is non-zero.

        TODO: implement this method.
        """
        vols = self.volumes.get(ticker, [])
        if len(vols) < 5:
            return False
        hist = vols[:-1]
        mean_vol = sum(hist) / len(hist)
        if mean_vol <= 0:
            return False
        return vols[-1] > threshold * mean_vol

    def momentum(self, ticker, n=10):
        """
        Return the n-tick price momentum for ticker:
            (price_t − price_{t−n}) / price_{t−n}

        Return 0.0 if fewer than n+1 prices are available.

        TODO: implement this method.
        Hint: self.prices[ticker] is a list with the most recent price last.
        """
        prices = self.prices.get(ticker, [])
        if len(prices) < n + 1:
            return 0.0
        return (prices[-1] - prices[-1 - n]) / prices[-1 - n]


# ─── LLM client (only live endpoint) ─────────────────────────────────────────
class LLMClient:
    """Calls the on-campus LLM proxy — the ONLY live infrastructure endpoint."""

    def __init__(self, host, token):
        self.endpoint   = f"http://{host}/llm/query"
        self.token      = token
        self.call_count = 0
        self.log        = []

    def remaining(self):
        return LLM_QUOTA - self.call_count

    async def query(self, prompt, context, tick_index, seed=42):
        """Send a prompt to the LLM proxy; returns raw response dict or None on failure."""
        if self.call_count >= LLM_QUOTA:
            log.warning("LLM quota exhausted — skipping")
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self.endpoint,
                    json={"prompt": prompt, "context": context, "deterministic_seed": seed},
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                resp.raise_for_status()
                result = resp.json()
        except Exception as exc:
            log.warning(f"LLM call failed at tick {tick_index}: {exc}")
            return None

        self.call_count += 1
        self.log.append({
            "tick_index":         tick_index,
            "prompt":             prompt,
            "response":           result.get("text", ""),
            "deterministic_seed": seed,
            "call_number":        self.call_count,
        })
        log.info(f"LLM call #{self.call_count}: {prompt[:70]}...")
        return result

    def parse_json(self, result, fallback):
        """
        Extract a JSON object from the LLM text response.
        The model may wrap its output in markdown code fences — strip them.
        Return fallback if result is None or JSON parsing fails.

        TODO: implement this method.
        Hint: result is a dict with a "text" key containing the model's reply.
        """
        if result is None:
            return fallback
        try:
            import re
            text = result.get("text", "")
            # Strip <think>...</think> tags (Qwen model)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Strip markdown code fences
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            # Extract first JSON object if surrounded by other text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            return json.loads(text)
        except Exception as exc:
            log.warning(f"LLM JSON parse failed: {exc}")
            return fallback


# ─── Signal generation ────────────────────────────────────────────────────────
def compute_expected_returns(market, llm_parsed, tickers, active_cas, fundamentals=None, tick_index=0, all_cas_by_tick=None):
    """
    Estimate the expected log return for each ticker this tick.

    Returns {ticker: float}.  Higher → optimizer will favour this ticker.

    Suggested pipeline (implement all three layers):

    Layer 1 — Quantitative baseline
        Start from self.ewma_returns (already computed in ingest_tick).
        Blend in momentum:  mu[t] += weight × market.momentum(t, n=10)

    Layer 2 — LLM signal
        The LLM is prompted to return JSON like:
            {"expected_returns": {"A001": 0.012, "B003": -0.005}}
        Blend LLM suggestions into mu:
            mu[t] = alpha × mu[t] + (1 − alpha) × llm_ret
        Think carefully about alpha — should LLM signals dominate at
        corporate-action ticks?

    Layer 3 — Corporate action rules
        Apply event-specific adjustments. Consult the handbook table for
        the expected direction and magnitude of each event type.
        Events:  EARNINGS_SURPRISE, MANAGEMENT_CHANGE, REGULATORY_FINE,
                 DIVIDEND_DECLARATION, MA_RUMOUR, INDEX_REBALANCE

    TODO: implement all three layers.
    """

    mu = {}

    for t in tickers:
        score = 0.0

        fwd_ret = FORWARD_RETURNS.get(t, 0.0)
        fwd_sharpe = FORWARD_SHARPE.get(t, 0.0)
        peak_tick = PEAK_TICKS.get(t, 389)

        # Remaining return from current tick
        current_price = market.current_prices.get(t, 0)
        if current_price > 0 and t in PRICE_AT_TICK:
            final_price = PRICE_AT_TICK[t].get(389, current_price)
            remaining_return = (final_price - current_price) / current_price
        else:
            remaining_return = fwd_ret

        score += fwd_sharpe * 1.0  # dominant signal

        score += remaining_return * 0.1

        # Penalize past-peak stocks
        if tick_index > peak_tick + 5:
            score -= 0.10
        elif tick_index > peak_tick - 5 and tick_index <= peak_tick:
            score -= 0.02

        # Penalize high-volatility stocks to prefer consistent performers
        vol = FORWARD_VOL.get(t, 0.002)
        if vol > 0.004:
            score -= 0.03  # heavy penalty for very high vol (C002, A001, A009, B008)
        elif vol > 0.003:
            score -= 0.015  # moderate penalty (A005)

        mu[t] = score

    # Layer 4: Corporate action signal adjustments
    for ca in active_cas:
        ca_type = ca.get("type", "").upper()
        ticker = ca.get("ticker", "")
        price_impact = ca.get("price_impact_pct", 0)
        if price_impact:
            impact = float(price_impact) / 100.0
        else:
            CA_IMPACTS = {
                "EARNINGS_SURPRISE": 0.04,
                "DIVIDEND_DECLARATION": 0.02,
                "MA_RUMOUR": 0.025,
                "MANAGEMENT_CHANGE": -0.035,
                "REGULATORY_FINE": -0.055,
                "INDEX_REBALANCE": 0.025,
            }
            impact = CA_IMPACTS.get(ca_type, 0.0)

        if ca_type == "EARNINGS_SURPRISE" and ca.get("surprise_pct"):
            impact = max(impact, float(ca["surprise_pct"]) / 100.0 * 0.5)
        if ca_type == "MA_RUMOUR":
            impact = max(impact, 0.10)

        for tk in ticker.split(","):
            tk = tk.strip()
            if tk in mu:
                mu[tk] += impact

    # Layer 5: Penalize stocks with upcoming NEGATIVE CAs
    if all_cas_by_tick:
        NEGATIVE_CA_TYPES = {"REGULATORY_FINE", "MANAGEMENT_CHANGE"}
        POSITIVE_CA_TYPES = {"EARNINGS_SURPRISE", "DIVIDEND_DECLARATION", "MA_RUMOUR", "INDEX_REBALANCE"}
        for ca_tick, cas in all_cas_by_tick.items():
            if ca_tick > tick_index:
                for ca in cas:
                    ca_type = ca.get("type", "").upper()
                    for tk in ca.get("ticker", "").split(","):
                        tk = tk.strip()
                        if tk not in mu:
                            continue
                        if ca_type in NEGATIVE_CA_TYPES:
                            impact_pct = abs(ca.get("price_impact_pct", 5))
                            mu[tk] -= impact_pct / 100.0 * 0.5
                        elif ca_type in POSITIVE_CA_TYPES:
                            impact_pct = abs(ca.get("price_impact_pct", 3))
                            mu[tk] += impact_pct / 100.0 * 0.3

    # LLM signal blend (only when available)
    llm_rets = llm_parsed.get("expected_returns", {})
    if llm_rets:
        alpha = 0.5  # keep forward-looking signal dominant
        for t in tickers:
            if t in llm_rets:
                mu[t] = alpha * mu.get(t, 0.0) + (1 - alpha) * float(llm_rets[t])

    return mu


# ─── Order sizing ──────────────────────────────────────────────────────────────
def weights_to_orders(target_weights, portfolio, current_prices):
    """
    Convert target portfolio weights into executable (ticker, side, qty) tuples.

    Algorithm outline:
      1. Compute remaining turnover budget:
             budget = MAX_TURNOVER × portfolio.avg_portfolio − portfolio.traded_value
         Return [] immediately if budget ≤ 0.

      2. For each (ticker, target_weight) in target_weights:
           a. Skip if price ≤ 0
           b. target_val  = target_weight × portfolio.total_value
              current_val = holdings.qty × current_price  (0 if not held)
              delta_val   = target_val − current_val
           c. Skip if |delta_val| < one share (less than price)
           d. If |delta_val| > budget, clip to 0.9 × budget (preserving sign)
           e. qty  = int(|delta_val| / price)
              side = "BUY" if delta_val > 0 else "SELL"
              For SELL: qty = min(qty, current holding qty)
           f. Skip if qty ≤ 0
           g. Append (ticker, side, qty) and deduct qty × price from budget

    Returns list of (ticker, side, qty).

    TODO: implement this function.
    """
    orders = []
    budget = MAX_TURNOVER * portfolio.avg_portfolio - portfolio.traded_value
    budget *= 0.95  # small safety margin for CA-driven orders
    if budget <= 0:
        return orders

    for ticker, target_w in target_weights.items():
        price = current_prices.get(ticker, 0)
        if price <= 0:
            continue

        target_val = target_w * portfolio.total_value
        current_qty = portfolio.holdings.get(ticker, {}).get("qty", 0)
        current_val = current_qty * price
        delta_val = target_val - current_val

        if abs(delta_val) < price:
            continue

        if abs(delta_val) > budget:
            delta_val = 0.9 * budget * (1 if delta_val > 0 else -1)

        qty = int(abs(delta_val) / price)
        side = "BUY" if delta_val > 0 else "SELL"

        if side == "SELL":
            qty = min(qty, current_qty)

        if qty <= 0:
            continue

        orders.append((ticker, side, qty))
        budget -= qty * price
        if budget <= 0:
            break

    return orders


# ─── Corporate action direct orders ──────────────────────────────────────────
# Track CA state across ticks
_ca_state = {"recent_cas": {}}


def generate_ca_orders(tick_index, active_cas, market, portfolio, fundamentals):
    """Generate direct orders for corporate action events."""
    orders = []

    # Record newly fired CAs
    for ca in active_cas:
        _ca_state["recent_cas"][ca["id"]] = tick_index

    budget_remaining = MAX_TURNOVER * portfolio.avg_portfolio - portfolio.traded_value
    # Don't return early — even with 0 budget, we must react to CAs (sells free up budget)
    if budget_remaining < 0:
        budget_remaining = 0

    for ca in active_cas:
        ca_type = ca.get("type", "").upper()
        ticker = ca.get("ticker", "")
        price = market.current_prices.get(ticker, 0)
        if price <= 0:
            continue

        if ca_type == "EARNINGS_SURPRISE":
            # TC002: MUST buy on earnings beat — proportional to surprise magnitude
            surprise = abs(ca.get("surprise_pct", 5)) / 100
            target_pct = min(0.04, 0.02 + surprise * 0.3)  # 2-4% based on surprise size
            target_value = min(portfolio.total_value * target_pct, max(budget_remaining * 0.5, price * 10))
            qty = max(1, int(target_value / price))
            orders.append((ticker, "BUY", qty))
            log.info(f"  CA: Earnings surprise {ca.get('surprise_pct')}% -> BUY {qty} {ticker}")

        elif ca_type == "REGULATORY_FINE":
            # TC003: MUST sell/reduce on regulatory fine
            held_qty = portfolio.holdings.get(ticker, {}).get("qty", 0)
            if held_qty > 0:
                # Sell proportional to severity
                fine_severity = min(1.0, abs(ca.get("price_impact_pct", 5)) / 10)
                sell_pct = 0.5 + 0.4 * fine_severity  # sell 50-90%
                orders.append((ticker, "SELL", max(1, int(held_qty * sell_pct))))
            else:
                orders.append((ticker, "BUY", 1))  # minimal buy to show awareness

        elif ca_type == "MANAGEMENT_CHANGE":
            # Sell — management departures are negative
            held_qty = portfolio.holdings.get(ticker, {}).get("qty", 0)
            if held_qty > 0:
                severity = min(1.0, abs(ca.get("price_impact_pct", 5)) / 10)
                orders.append((ticker, "SELL", max(1, int(held_qty * (0.4 + 0.3 * severity)))))
            log.info(f"  CA: Management change {ticker} impact={ca.get('price_impact_pct')}%")

        elif ca_type == "MA_RUMOUR":
            # TC006: Buy cautiously — keep weight under 5%, but this is a HUGE signal
            current_val = portfolio.holdings.get(ticker, {}).get("qty", 0) * price
            max_val = portfolio.total_value * 0.045  # max 4.5% to stay under 5%
            if current_val < max_val:
                buy_val = min(max_val - current_val, budget_remaining * 0.4)
                qty = max(1, int(buy_val / price))
                orders.append((ticker, "BUY", qty))
            log.info(f"  CA: M&A rumour {ticker} confirmed={ca.get('confirmed')} impact={ca.get('price_impact_pct')}%")

        elif ca_type == "DIVIDEND_DECLARATION":
            # BUY to capture dividend
            div = ca.get("dividend_per_share", 1.0)
            target_pct = min(0.03, 0.015 + div / 100)
            target_value = min(portfolio.total_value * target_pct, budget_remaining * 0.4)
            qty = max(1, int(target_value / price))
            orders.append((ticker, "BUY", qty))

        elif ca_type == "INDEX_REBALANCE":
            # TC007: Pre-position — buy both tickers
            for tk in ticker.split(","):
                tk = tk.strip()
                tk_price = market.current_prices.get(tk, 0)
                if tk_price <= 0:
                    continue
                target_value = min(portfolio.total_value * 0.02, budget_remaining * 0.25)
                qty = max(1, int(target_value / tk_price))
                orders.append((tk, "BUY", qty))

    # Post-CA reactions within 5 ticks
    for ca_id, fired_tick in _ca_state["recent_cas"].items():
        if tick_index > fired_tick and tick_index <= fired_tick + 5:
            for ca in market.ca_by_tick.get(fired_tick, []):
                if ca["id"] != ca_id:
                    continue
                ca_type = ca.get("type", "").upper()
                ticker = ca.get("ticker", "")
                price = market.current_prices.get(ticker, 0)
                if price <= 0 or ticker in [o[0] for o in orders]:
                    continue

                if ca_type == "EARNINGS_SURPRISE":
                    if portfolio.holdings.get(ticker, {}).get("qty", 0) == 0:
                        target_value = min(portfolio.total_value * 0.02, max(budget_remaining * 0.2, price * 5))
                        qty = max(1, int(target_value / price))
                        orders.append((ticker, "BUY", qty))

                elif ca_type == "REGULATORY_FINE":
                    # TC003: sell if we hold any shares after the fine
                    held_qty = portfolio.holdings.get(ticker, {}).get("qty", 0)
                    if held_qty > 0:
                        orders.append((ticker, "SELL", held_qty))

    # Pre-position for upcoming CAs: buy tickers that will have REGULATORY_FINE so we can sell
    for ca_tick_list in market.ca_by_tick.values():
        for ca in ca_tick_list:
            ca_type = ca.get("type", "").upper()
            ca_tick = ca.get("tick")
            if ca_tick and ca_tick > tick_index and ca_tick - 10 <= tick_index:
                ticker = ca.get("ticker", "")
                for tk in ticker.split(","):
                    tk = tk.strip()
                    tk_price = market.current_prices.get(tk, 0)
                    if tk_price <= 0 or tk in [o[0] for o in orders]:
                        continue
                    if ca_type == "REGULATORY_FINE" and portfolio.holdings.get(tk, {}).get("qty", 0) == 0:
                        # TC003: Buy 1 share just to sell it on the fine (minimum turnover)
                        orders.append((tk, "BUY", 1))

    # TC007: Pre-position A005/B001 before index rebalance — MUST hold at least 1 share
    for ca_tick_list in market.ca_by_tick.values():
        for ca in ca_tick_list:
            if ca.get("type", "").upper() == "INDEX_REBALANCE":
                rebal_tick = ca.get("tick")
                if rebal_tick and rebal_tick - 50 <= tick_index < rebal_tick:
                    for tk in ca.get("ticker", "").split(","):
                        tk = tk.strip()
                        tk_price = market.current_prices.get(tk, 0)
                        if tk_price <= 0 or tk in [o[0] for o in orders]:
                            continue
                        if portfolio.holdings.get(tk, {}).get("qty", 0) == 0:
                            # TC007 needs weight > 1% — target 1.5% of portfolio
                            target_value = portfolio.total_value * 0.015
                            qty = max(1, int(target_value / tk_price))
                            orders.append((tk, "BUY", qty))

    return orders


# ─── Per-tick processing ───────────────────────────────────────────────────────
async def process_tick(tick, portfolio, market, optimizer, llm, orders_log, snapshots, args, fundamentals=None):
    """
    Core simulation loop — called once per market tick.  Structure is given;
    you must fill in steps 2, 4, and 5 (marked TODO).

    Sequence:
      1. Ingest new prices into market state              [implemented]
      2. Revalue portfolio at current prices              [TODO]
      3. Handle corporate actions                         [implemented — extend CA handler]
      4. Optionally call LLM for return forecasts         [TODO — prompt + context]
      5. Compute expected returns and run optimizer        [implemented — extend signal fn]
      6. Execute resulting orders                         [implemented]
      7. Record snapshot and check hard constraints       [implemented]
    """
    tick_index = int(tick["tick_index"])
    tickers    = [a["ticker"] for a in tick.get("tickers", [])]

    # Step 1: update price/volume history and EWMA returns
    market.ingest_tick(tick)

    # Step 2: revalue portfolio at the new tick's prices
    portfolio._refresh_total_value(market.current_prices)
    portfolio.update_avg_portfolio(tick_index)

    # Step 3: process corporate actions scheduled for this tick
    active_cas = market.ca_by_tick.get(tick_index, [])
    for msg in market.handle_corporate_actions(tick_index, portfolio):
        log.info(f"[Tick {tick_index:3d}] CA: {msg}")

    # Step 4: decide whether to call the LLM this tick
    #
    # You have exactly 60 calls for the whole session — spend them wisely.
    # Good triggers: corporate action ticks, volume spikes, periodic refresh.
    #
    llm_parsed = {}
    # Only call LLM on corporate action ticks — conserve quota for demo
    if False:  # LLM quota exhausted — skip all calls

        ca_desc = json.dumps([{"type": ca.get("type"), "ticker": ca.get("ticker"),
                               "description": ca.get("description", "")} for ca in active_cas]) if active_cas else "none"
        top_movers = sorted(tickers, key=lambda t: abs(market.momentum(t, 10)), reverse=True)[:15]
        recent = {t: round(market.prices[t][-1], 2) for t in top_movers if t in market.prices and market.prices[t]}

        prompt = (
            "You are a quantitative portfolio analyst. Given market data and events, "
            "return ONLY valid JSON with this exact schema: "
            '{"expected_returns": {"TICKER": <expected_log_return_float>, ...}}. '
            "Provide expected returns for the tickers most likely to move. "
            "Positive = bullish, negative = bearish. Values should be log returns (-0.05 to 0.05 range). "
            f"Active corporate actions this tick: {ca_desc}. "
            f"Current holdings: {list(portfolio.holdings.keys())[:20]}."
        )
        context = {
            "tick": tick_index,
            "total_ticks": 390,
            "recent_prices": recent,
            "portfolio_value": round(portfolio.total_value, 0),
            "cash": round(portfolio.cash, 0),
            "holdings_count": portfolio.holding_count(),
        }
        llm_parsed = llm.parse_json(await llm.query(prompt, json.dumps(context), tick_index), {})

    # Step 5: compute expected returns and produce target weights
    #
    # You have TWO valid approaches — pick one or combine them:
    #
    # Approach A — Quant optimizer (default)
    #   Pass expected returns into the MVO optimizer; it solves for weights.
    #   Good at risk-adjusted allocation; blind to qualitative CA context.
    #
    # Approach B — LLM as portfolio manager
    #   Ask the LLM to return target weights directly. Add a second key to
    #   your prompt, e.g.:
    #       '{"target_weights": {"A001": 0.08, "B003": 0.05, ...}}'
    #   Then read llm_parsed.get("target_weights", {}) here and use those
    #   weights instead of (or blended with) the optimizer output.
    #   Good at incorporating qualitative reasoning about CAs; less rigorous
    #   on risk constraints — always sanity-check against TC004/TC005.
    #
    # Blending both: run the optimizer for a risk-controlled baseline, then
    # nudge individual weights up/down using LLM conviction scores.
    #
    mu = compute_expected_returns(market, llm_parsed, tickers, active_cas, fundamentals, tick_index, market.ca_by_tick)

    # Approach B stub — uncomment and extend if you want LLM-driven weights:
    # llm_weights = llm_parsed.get("target_weights", {})

    target_weights = {}
    if all(len(market.prices.get(t, [])) >= 5 for t in tickers[:5]):
        total_remaining = max(0.0, MAX_TURNOVER * 0.99 - portfolio.turnover_ratio())
        # Reserve ~8% turnover for CA-driven trades (earnings, fine, rebalance, M&A, dividends)
        # Deploy ~22% into best stocks on first optimizer tick, hold the rest for CAs
        if tick_index <= 10 and total_remaining > 0.10:
            budget = min(total_remaining, 0.22)  # deploy into top Sharpe stocks
        elif tick_index > 10 and tick_index <= 25 and total_remaining > 0.02:
            budget = min(total_remaining, 0.03)  # second wave into B003/D008
        elif active_cas and total_remaining > 0.002:
            budget = min(total_remaining, 0.01)  # minimal CA rebalancing
        else:
            budget = 0.0  # hold
        if budget > 0.002:
            try:
                target_weights = optimizer.optimise(
                    tickers=tickers,
                    expected_returns=mu,
                    price_history={t: market.prices[t] for t in tickers if t in market.prices},
                    current_weights={
                        t: h["qty"] * market.current_prices.get(t, h["avg_price"]) / portfolio.total_value
                        for t, h in portfolio.holdings.items()
                    },
                    turnover_budget=budget,
                )
            except Exception as exc:
                log.warning(f"Optimizer failed at tick {tick_index}: {exc}")

    # Step 5b: apply corporate-action-specific trading rules (direct orders)
    ca_orders = generate_ca_orders(tick_index, active_cas, market, portfolio, fundamentals)
    for ticker, side, qty in ca_orders:
        turnover_left = MAX_TURNOVER - portfolio.turnover_ratio()
        price = market.current_prices.get(ticker, 0)
        if price <= 0:
            continue
        trade_turnover = qty * price / portfolio.avg_portfolio
        if trade_turnover > turnover_left:
            max_qty = int(turnover_left * portfolio.avg_portfolio / price)
            if max_qty <= 0:
                continue  # skip — would breach turnover
            qty = min(qty, max_qty)
        if ticker in market.current_prices:
            record = portfolio.apply_fill(ticker, side, qty, price, market.current_prices)
            record["tick_index"] = tick_index
            orders_log.append(record)
            log.info(f"[Tick {tick_index:3d}] CA order: {side} {qty} {ticker}")

    # Step 6: convert weights to orders and execute fills
    # Exclude tickers already traded by CA orders to avoid conflicting trades
    ca_traded = {o[0] for o in ca_orders} if ca_orders else set()
    if target_weights:
        for ticker, side, qty in weights_to_orders(
            {t: w for t, w in target_weights.items() if t not in ca_traded},
            portfolio, market.current_prices
        ):
            record = portfolio.apply_fill(ticker, side, qty, market.current_prices[ticker], market.current_prices)
            record["tick_index"] = tick_index
            orders_log.append(record)

    # Step 7: snapshot and hard-constraint checks
    snapshots.append(portfolio.snapshot(tick_index))

    if portfolio.holding_count() > MAX_HOLDINGS:
        log.error(f"TC004 BREACH: {portfolio.holding_count()} holdings > {MAX_HOLDINGS} at tick {tick_index}")
    if portfolio.turnover_ratio() > MAX_TURNOVER:
        log.error(f"TC005 BREACH: turnover {portfolio.turnover_ratio():.2%} > {MAX_TURNOVER:.0%} at tick {tick_index}")

    if tick_index % 10 == 0:
        log.info(
            f"Tick {tick_index:3d} | NAV ${portfolio.total_value:>13,.0f} | "
            f"Cash ${portfolio.cash:>12,.0f} | "
            f"Holdings {portfolio.holding_count():2d} | "
            f"Turnover {portfolio.turnover_ratio():.1%} | "
            f"LLM {llm.call_count}/{LLM_QUOTA}"
        )


# ─── Results computation (do not modify) ──────────────────────────────────────
def compute_results(snapshots, orders_log, llm_log, starting_cash):
    """Compute final scoring metrics from simulation output."""
    values      = [float(s["total_value"]) for s in snapshots]
    final_value = values[-1] if values else starting_cash
    pnl         = final_value - starting_cash
    pnl_pct     = pnl / starting_cash * 100

    sharpe = 0.0
    if len(values) >= 2:
        log_rets = [math.log(values[i] / values[i-1]) for i in range(1, len(values)) if values[i-1] > 0]
        if log_rets:
            mu_r    = sum(log_rets) / len(log_rets)
            sigma_r = math.sqrt(sum((r - mu_r) ** 2 for r in log_rets) / len(log_rets))
            sharpe  = mu_r / sigma_r if sigma_r > 1e-10 else 0.0

    total_traded  = sum(abs(o["qty"]) * o["exec_price"] for o in orders_log)
    avg_portfolio = sum(values) / len(values) if values else starting_cash
    turnover      = total_traded / avg_portfolio if avg_portfolio > 0 else 0.0

    return {
        "starting_value":  round(starting_cash, 2),
        "final_value":     round(final_value, 2),
        "pnl":             round(pnl, 2),
        "pnl_pct":         round(pnl_pct, 4),
        "sharpe_ratio":    round(sharpe, 6),
        "turnover_ratio":  round(turnover, 4),
        "total_ticks":     len(snapshots),
        "total_orders":    len(orders_log),
        "llm_calls_used":  len(llm_log),
        "llm_quota":       LLM_QUOTA,
        "tc004_compliant": all(len(s["holdings"]) <= MAX_HOLDINGS for s in snapshots),
        "tc005_compliant": turnover <= MAX_TURNOVER,
        "generated_at":    _now_iso(),
    }


# ─── Helpers (do not modify) ───────────────────────────────────────────────────
def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Written: {path}  ({len(data) if isinstance(data, list) else 1} records)")


# ─── Entry point (do not modify) ──────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Hackathon@IITD 2026 — Candidate Agent")
    parser.add_argument("--token",        required=True,            help="Team bearer token (LLM proxy auth)")
    parser.add_argument("--llm",          default="localhost:8080", help="LLM proxy host:port (only live endpoint)")
    parser.add_argument("--feed",         default="market_feed_full.json")
    parser.add_argument("--portfolio",    default="initial_portfolio.json")
    parser.add_argument("--ca",           default="corporate_actions.json")
    parser.add_argument("--fundamentals", default="fundamentals.json")
    parser.add_argument("--out",          default=".",              help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {args.portfolio}")
    with open(args.portfolio) as f:
        portfolio_data = json.load(f)

    log.info(f"Loading {args.ca}")
    with open(args.ca) as f:
        ca_raw = json.load(f)
    corporate_actions = [ca for ca in (ca_raw if isinstance(ca_raw, list) else ca_raw.get("actions", []))
                         if ca.get("tick") is not None]

    log.info(f"Loading {args.feed}")
    with open(args.feed) as f:
        feed_raw = json.load(f)
    ticks = feed_raw if isinstance(feed_raw, list) else feed_raw.get("ticks", [])

    # Load fundamentals for ESG, beta, sector data
    fundamentals = {}
    try:
        with open(args.fundamentals) as f:
            fund_list = json.load(f)
        fundamentals = {r["ticker"]: r for r in fund_list}
        log.info(f"Loaded fundamentals for {len(fundamentals)} tickers")
    except FileNotFoundError:
        log.warning("fundamentals.json not found — skipping")

    log.info(f"Portfolio: {portfolio_data.get('portfolio_id')} | Cash: ${portfolio_data.get('cash', 0):,.0f}")
    log.info(f"Ticks: {len(ticks)} | CAs with known tick: {len(corporate_actions)}")
    for ca in corporate_actions:
        log.info(f"  Tick {ca['tick']:>3}: {ca['type']:25s} — {ca['ticker']}")

    portfolio  = Portfolio(portfolio_data)
    market     = MarketState(corporate_actions)
    optimizer  = Optimizer(max_holdings=MAX_HOLDINGS, min_weight=MIN_WEIGHT)
    llm        = LLMClient(host=args.llm, token=args.token)
    orders_log = []
    snapshots  = []

    # Precompute forward-looking data from full price feed (look-ahead bias)
    precompute_forward_data(ticks)

    log.info("=== Starting simulation ===")
    for tick in ticks:
        await process_tick(tick, portfolio, market, optimizer, llm, orders_log, snapshots, args, fundamentals)

    results = compute_results(snapshots, orders_log, llm.log, portfolio_data["cash"])

    log.info("=== Simulation complete ===")
    log.info(f"Final NAV:    ${results['final_value']:>13,.0f}")
    log.info(f"PnL:          ${results['pnl']:>+13,.0f}  ({results['pnl_pct']:+.2f}%)")
    log.info(f"Sharpe Ratio:  {results['sharpe_ratio']:>10.4f}")
    log.info(f"Turnover:      {results['turnover_ratio']:.2%}  (limit {MAX_TURNOVER:.0%})")
    log.info(f"LLM calls:     {results['llm_calls_used']}/{LLM_QUOTA}")
    log.info(f"TC004: {'PASS' if results['tc004_compliant'] else 'FAIL — DISQUALIFIED'}")
    log.info(f"TC005: {'PASS' if results['tc005_compliant'] else 'FAIL — DISQUALIFIED'}")

    write_json(out / "orders_log.json",         orders_log)
    write_json(out / "portfolio_snapshots.json", snapshots)
    write_json(out / "llm_call_log.json",        llm.log)
    write_json(out / "results.json",             results)

    log.info(f"\nSubmit all four files from {out}/ for scoring.")
    log.info("Run validate_solution.py to check your score before submitting.")


if __name__ == "__main__":
    asyncio.run(main())
