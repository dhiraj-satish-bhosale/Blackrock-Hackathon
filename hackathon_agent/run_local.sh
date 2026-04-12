#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"
exec python3 agent_candidate.py \
  --token "${TOKEN:?set TOKEN env var}" \
  --llm "${LLM:-localhost:8080}" \
  --feed "${FEED:-$ROOT/starter-kit/Starter_Kit/market_feed_sample.json}" \
  --portfolio "${PORTFOLIO:-$ROOT/starter-kit/Starter_Kit/initial_portfolio.json}" \
  --ca "${CA:-$ROOT/starter-kit/Starter_Kit/corporate_actions.json}" \
  --out "${OUT:-./out}"
