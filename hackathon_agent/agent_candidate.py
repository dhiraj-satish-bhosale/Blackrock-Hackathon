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
import copy
import json
import logging
import math
import re
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
MIN_PRICE_HISTORY_FOR_MVO = 5
TURNOVER_HEADROOM_FOR_LLM   = 0.25  # block LLM calls if results-style turnover already here
TURNOVER_EXEC_BUFFER        = 0.97  # stay below 30% vs compute_results / float noise
LLM_BLEND_LLMW              = 0.6   # final_mu = LLM_BLEND_LLMW * llm + (1-LLM_BLEND_LLMW) * ewma
COVAR_RETURN_WINDOW          = 30   # passed to Optimizer (see optimizer.py)


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
        self._value_sum    = 0.0
        self._tick_count   = 0
        self.avg_portfolio = max(float(self.cash), 1e-6)  # overwritten after MTM refresh below

        for h in initial.get("holdings", []):
            self.holdings[h["ticker"]] = {
                "qty": int(h["qty"]),
                "avg_price": float(h["avg_price"]),
            }

        self._refresh_total_value({})
        self.avg_portfolio = max(self.total_value, 1e-6)

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
        side = side.upper()

        if side == "SELL":
            held = int(self.holdings.get(ticker, {}).get("qty", 0))
            qty = min(int(qty), held)
            if qty <= 0:
                self._refresh_total_value(current_prices)
                return {
                    "type":       "execution",
                    "order_ref":  f"ord_{ticker}_{side}_0",
                    "ticker":     ticker,
                    "side":       side,
                    "qty":        0,
                    "exec_price": round(exec_price, 4),
                    "fees":       0.0,
                    "ts":         _now_iso(),
                }
            notional = float(qty) * float(exec_price)
            fee = round(PROP_FEE * notional + FIXED_FEE, 2)
            proceeds = qty * float(exec_price) - fee
            self.cash += proceeds
            new_qty = held - qty
            if new_qty <= 0:
                self.holdings.pop(ticker, None)
            else:
                self.holdings[ticker]["qty"] = new_qty
        else:  # BUY
            notional = float(qty) * float(exec_price)
            fee = round(PROP_FEE * notional + FIXED_FEE, 2)
            gross = qty * float(exec_price) + fee
            if gross > self.cash:
                max_notional = self.cash - FIXED_FEE
                if max_notional <= 0 or exec_price <= 0:
                    qty = 0
                else:
                    denom = float(exec_price) * (1.0 + PROP_FEE)
                    qty = int(max_notional / denom)
                if qty <= 0:
                    self._refresh_total_value(current_prices)
                    return {
                        "type":       "execution",
                        "order_ref":  f"ord_{ticker}_{side}_0",
                        "ticker":     ticker,
                        "side":       side,
                        "qty":        0,
                        "exec_price": round(exec_price, 4),
                        "fees":       0.0,
                        "ts":         _now_iso(),
                    }
                notional = float(qty) * float(exec_price)
                fee = round(PROP_FEE * notional + FIXED_FEE, 2)
            if ticker in self.holdings:
                old = self.holdings[ticker]
                old_qty, old_avg = int(old["qty"]), float(old["avg_price"])
                new_qty = old_qty + qty
                new_avg = (old_qty * old_avg + qty * float(exec_price)) / new_qty if new_qty else float(exec_price)
                self.holdings[ticker] = {"qty": new_qty, "avg_price": new_avg}
            else:
                self.holdings[ticker] = {"qty": int(qty), "avg_price": float(exec_price)}
            self.cash -= qty * float(exec_price) + fee

        self.traded_value += float(qty) * float(exec_price)
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
        mtm = self.cash
        for t, h in self.holdings.items():
            px = float(current_prices.get(t, h["avg_price"]))
            mtm += int(h["qty"]) * px
        self.total_value = mtm

    def update_avg_portfolio(self, nav_sample):
        """
        Update the running time-average of portfolio NAV (aligned with snapshot total_value).

        self.avg_portfolio is the denominator used in turnover_ratio().
        Pass the same total_value that will be written to portfolio_snapshots this tick.
        """
        self._tick_count += 1
        self._value_sum += float(nav_sample)
        self.avg_portfolio = self._value_sum / self._tick_count if self._tick_count else float(nav_sample)

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

            if len(self.prices[t]) > PRICE_HISTORY_LEN:
                self.prices[t] = self.prices[t][-PRICE_HISTORY_LEN:]
            if len(self.volumes[t]) > PRICE_HISTORY_LEN:
                self.volumes[t] = self.volumes[t][-PRICE_HISTORY_LEN:]

            series = self.prices[t]
            if len(series) < 2:
                self.ewma_returns[t] = 0.0
            else:
                p0, p1 = float(series[-2]), float(series[-1])
                log_ret = math.log(p1 / p0) if p0 > 0 and p1 > 0 else 0.0
                prev = self.ewma_returns.get(t, 0.0)
                self.ewma_returns[t] = EWMA_LAMBDA * prev + (1.0 - EWMA_LAMBDA) * log_ret

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
                if ratio <= 0 or not ticker:
                    continue
                if ticker not in self.split_adjusted:
                    if ticker in self.prices and self.prices[ticker]:
                        self.prices[ticker] = [p / ratio for p in self.prices[ticker]]
                    self.split_adjusted.add(ticker)
                    self.ewma_returns[ticker] = 0.0
                if ticker in portfolio.holdings:
                    h = portfolio.holdings[ticker]
                    h["qty"] = int(round(int(h["qty"]) * ratio))
                    h["avg_price"] = float(h["avg_price"]) / ratio

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
        last = vols[-1]
        prev = vols[:-1]
        mean_prev = sum(prev) / len(prev)
        if mean_prev <= 0:
            return False
        return last > threshold * mean_prev

    def momentum(self, ticker, n=10):
        """
        Return the n-tick price momentum for ticker:
            (price_t − price_{t−n}) / price_{t−n}

        Return 0.0 if fewer than n+1 prices are available.

        TODO: implement this method.
        Hint: self.prices[ticker] is a list with the most recent price last.
        """
        series = self.prices.get(ticker, [])
        if len(series) < n + 1:
            return 0.0
        p_old, p_new = float(series[-(n + 1)]), float(series[-1])
        if p_old <= 0:
            return 0.0
        return (p_new - p_old) / p_old


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
            async with httpx.AsyncClient(timeout=15.0) as client:
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
            text = result.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            stripped = text.strip()
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
            return json.loads(stripped)
        except Exception as exc:
            log.warning(f"LLM JSON parse failed: {exc}")
            return fallback


# ─── Signal generation ────────────────────────────────────────────────────────
def compute_expected_returns(market, llm_parsed, tickers, active_cas):
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
        ew = float(market.ewma_returns.get(t, 0.0))
        mom = float(market.momentum(t, n=10))
        mu[t] = ew + 0.25 * mom

    llm_map = {}
    if isinstance(llm_parsed, dict):
        llm_map = llm_parsed.get("expected_returns") or {}
    if not isinstance(llm_map, dict):
        llm_map = {}
    for t in tickers:
        if t in llm_map and llm_map[t] is not None:
            try:
                llm_ret = float(llm_map[t])
            except (TypeError, ValueError):
                continue
            baseline = float(mu.get(t, 0.0))
            mu[t] = (1.0 - LLM_BLEND_LLMW) * baseline + LLM_BLEND_LLMW * llm_ret

    for ca in active_cas:
        ctype = str(ca.get("type", "")).upper()
        raw = ca.get("ticker", "") or ""
        affected = [x.strip() for x in str(raw).split(",") if x.strip()]
        bump = 0.0
        if ctype == "EARNINGS_SURPRISE":
            bump = 0.002
        elif ctype == "REGULATORY_FINE":
            bump = -0.002
        elif ctype == "MA_RUMOUR":
            bump = 0.001
        elif ctype == "DIVIDEND_DECLARATION":
            bump = 0.0005
        elif ctype == "MANAGEMENT_CHANGE":
            bump = -0.0005
        elif ctype == "INDEX_REBALANCE":
            bump = 0.0005
        else:
            continue
        for tick in affected:
            if tick in mu:
                mu[tick] = float(mu[tick]) + bump

    return mu


# ─── Turnover / constraints (align with results.json) ────────────────────────
def results_style_turnover(orders_log, snapshots):
    """Same turnover definition as compute_results (for guards and LLM headroom)."""
    total_traded = sum(abs(int(o["qty"])) * float(o["exec_price"]) for o in orders_log)
    values = [float(s["total_value"]) for s in snapshots]
    avg_portfolio = sum(values) / len(values) if values else 0.0
    if avg_portfolio <= 0:
        return 0.0
    return total_traded / avg_portfolio


def order_batch_passes_guard(portfolio, orders_log, snapshots, batch, current_prices):
    """Reject entire batch if TC004/TC005 would be violated under results.json turnover."""
    if not batch:
        return True
    try:
        sim = copy.deepcopy(portfolio)
    except Exception as exc:
        log.warning(f"Deepcopy portfolio failed — rejecting batch: {exc}")
        return False
    orders_traded = sum(abs(int(o["qty"])) * float(o["exec_price"]) for o in orders_log)
    for ticker, side, qty in batch:
        if ticker not in current_prices or float(current_prices[ticker]) <= 0:
            continue
        sim.apply_fill(ticker, side, int(qty), float(current_prices[ticker]), current_prices)
    total_traded = orders_traded + (float(sim.traded_value) - float(portfolio.traded_value))
    values = [float(s["total_value"]) for s in snapshots]
    nav_end = float(sim.total_value)
    projected_avg = (sum(values) + nav_end) / (len(values) + 1) if values else nav_end
    if projected_avg <= 0:
        return False
    if total_traded / projected_avg > MAX_TURNOVER * TURNOVER_EXEC_BUFFER:
        return False
    if sim.holding_count() > MAX_HOLDINGS:
        return False
    return True


def should_call_llm(tick_index, portfolio, market, tickers, active_cas, llm, orders_log, snapshots):
    """Budget gate: warmup + event + periodic triggers, turnover headroom, quota."""
    if llm.remaining() <= 0:
        return False
    if results_style_turnover(orders_log, snapshots) >= TURNOVER_HEADROOM_FOR_LLM:
        return False
    warmup = tick_index <= 10 and tick_index in (0, 2, 5, 8, 10)
    periodic = tick_index > 10 and tick_index % 25 == 0
    vol_spike = any(market.volume_spike(t) for t in tickers)
    event_ca = bool(active_cas)
    event_vol = vol_spike and tick_index > 10 and tick_index % 8 == 0
    event = event_ca or event_vol
    if warmup:
        return True
    if periodic:
        return True
    if event and tick_index >= 1:
        return True
    return False


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
    budget = (
        MAX_TURNOVER * TURNOVER_EXEC_BUFFER * float(portfolio.avg_portfolio)
        - float(portfolio.traded_value)
    )
    if budget <= 0:
        return []

    orders = []
    for ticker, w in sorted(target_weights.items(), key=lambda x: x[0]):
        price = float(current_prices.get(ticker, 0.0))
        if price <= 0:
            continue
        target_val = float(w) * float(portfolio.total_value)
        held = portfolio.holdings.get(ticker, {"qty": 0, "avg_price": 0.0})
        cur_qty = int(held.get("qty", 0))
        current_val = cur_qty * price
        delta_val = target_val - current_val
        if abs(delta_val) < price:
            continue
        clip = delta_val
        if abs(clip) > budget:
            clip = math.copysign(0.9 * budget, clip)
        qty = int(abs(clip) / price)
        if qty <= 0:
            continue
        side = "BUY" if clip > 0 else "SELL"
        if side == "SELL":
            qty = min(qty, cur_qty)
        if qty <= 0:
            continue
        notional = qty * price
        if notional > budget + 1e-9:
            qty = int(budget / price)
            if qty <= 0:
                continue
            notional = qty * price
        orders.append((ticker, side, qty))
        budget -= notional
        if budget <= 0:
            break
    return orders


# ─── Per-tick processing ───────────────────────────────────────────────────────
async def process_tick(tick, portfolio, market, optimizer, llm, orders_log, snapshots, args):
    """
    Per-tick pipeline: corporate actions (split-adjust) before ingest/signals,
    MTM, LLM budget gate, MVO on eligible universe, constraint guard, fills,
    snapshot, incremental JSON flush.
    """
    tick_index = int(tick["tick_index"])
    tickers    = [a["ticker"] for a in tick.get("tickers", [])]
    active_cas = market.ca_by_tick.get(tick_index, [])

    for msg in market.handle_corporate_actions(tick_index, portfolio):
        log.info(f"[Tick {tick_index:3d}] CA: {msg}")

    market.ingest_tick(tick)
    portfolio._refresh_total_value(market.current_prices)

    llm_parsed = {}
    if should_call_llm(tick_index, portfolio, market, tickers, active_cas, llm, orders_log, snapshots):
        prompt = (
            "You are a quant analyst. Return ONLY valid JSON with this exact shape: "
            '{"expected_returns": {"TICKER": <float>, ...}} '
            "where each value is a short-horizon expected log return (rough order 1e-3 to 1e-2). "
            f"Tick {tick_index}. Summarize how corporate actions should shift relative rankings."
        )
        context = {
            "tick": tick_index,
            "active_corporate_actions": active_cas,
            "recent_prices": {t: market.prices[t][-8:] for t in tickers if t in market.prices},
            "ewma_returns": {t: market.ewma_returns.get(t, 0.0) for t in tickers},
            "holdings": {t: portfolio.holdings.get(t) for t in portfolio.holdings},
        }
        llm_parsed = llm.parse_json(
            await llm.query(prompt, context, tick_index),
            {},
        )

    mu = compute_expected_returns(market, llm_parsed, tickers, active_cas)

    eligible = [t for t in tickers if len(market.prices.get(t, [])) >= MIN_PRICE_HISTORY_FOR_MVO]
    target_weights = {}
    tv = max(float(portfolio.total_value), 1e-9)
    if len(eligible) >= 2:
        try:
            mu_sub = {t: float(mu.get(t, 0.0)) for t in eligible}
            target_weights = optimizer.optimise(
                tickers=eligible,
                expected_returns=mu_sub,
                price_history={t: market.prices[t] for t in eligible},
                current_weights={
                    t: h["qty"] * market.current_prices.get(t, h["avg_price"]) / tv
                    for t, h in portfolio.holdings.items()
                },
                turnover_budget=max(
                    0.0,
                    MAX_TURNOVER * TURNOVER_EXEC_BUFFER - portfolio.turnover_ratio(),
                ),
            )
        except Exception as exc:
            log.warning(f"Optimizer failed at tick {tick_index}: {exc}")

    batch = weights_to_orders(target_weights, portfolio, market.current_prices) if target_weights else []
    if batch and not order_batch_passes_guard(portfolio, orders_log, snapshots, batch, market.current_prices):
        log.warning(
            f"[Tick {tick_index:3d}] Order batch rejected by constraint guard "
            f"(TC004/TC005 vs results.json turnover)"
        )
        batch = []

    for ticker, side, qty in batch:
        record = portfolio.apply_fill(ticker, side, qty, market.current_prices[ticker], market.current_prices)
        record["tick_index"] = tick_index
        orders_log.append(record)

    snapshots.append(portfolio.snapshot(tick_index))
    portfolio.update_avg_portfolio(float(snapshots[-1]["total_value"]))

    flush_trade_outputs(Path(args.out), orders_log, snapshots)

    if portfolio.holding_count() > MAX_HOLDINGS:
        log.error(f"TC004 BREACH: {portfolio.holding_count()} holdings > {MAX_HOLDINGS} at tick {tick_index}")
    if portfolio.turnover_ratio() > MAX_TURNOVER:
        log.error(f"TC005 BREACH: turnover {portfolio.turnover_ratio():.2%} > {MAX_TURNOVER:.0%} at tick {tick_index}")
    rt = results_style_turnover(orders_log, snapshots)
    if rt > MAX_TURNOVER:
        log.error(f"TC005 (results-style) BREACH: turnover {rt:.2%} > {MAX_TURNOVER:.0%} at tick {tick_index}")

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


def flush_trade_outputs(out_dir: Path, orders_log, snapshots):
    """Rewrite orders and snapshots each tick so a crash still leaves usable JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "orders_log.json", "w") as f:
        json.dump(orders_log, f, indent=2)
    with open(out_dir / "portfolio_snapshots.json", "w") as f:
        json.dump(snapshots, f, indent=2)


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

    # Optional: load fundamentals.json for P/E, beta, ESG score, sector, etc.
    # fundamentals = {}
    # try:
    #     with open(args.fundamentals) as f:
    #         fundamentals = json.load(f)
    # except FileNotFoundError:
    #     log.warning("fundamentals.json not found — skipping")

    log.info(f"Portfolio: {portfolio_data.get('portfolio_id')} | Cash: ${portfolio_data.get('cash', 0):,.0f}")
    log.info(f"Ticks: {len(ticks)} | CAs with known tick: {len(corporate_actions)}")
    for ca in corporate_actions:
        log.info(f"  Tick {ca['tick']:>3}: {ca['type']:25s} — {ca['ticker']}")

    portfolio  = Portfolio(portfolio_data)
    market     = MarketState(corporate_actions)
    optimizer  = Optimizer(
        max_holdings=MAX_HOLDINGS,
        min_weight=MIN_WEIGHT,
        max_return_window=COVAR_RETURN_WINDOW,
    )
    llm        = LLMClient(host=args.llm, token=args.token)
    orders_log = []
    snapshots  = []

    log.info("=== Starting simulation ===")
    for tick in ticks:
        await process_tick(tick, portfolio, market, optimizer, llm, orders_log, snapshots, args)

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
