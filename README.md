# Algorithmic Trading Agent - BlackRock Hackathon, IIT Delhi (Apr 2026)

An automated trading agent that ran quantitative strategies across **50 assets** over a **390-tick** simulation, starting from a $10M portfolio.

## Strategy

- **Expected returns:** EWMA-based expected return model over per-ticker price history (lambda 0.94), plus volume-spike and momentum signals.
- **Portfolio construction:** mean-variance optimization with cvxpy, re-solved every tick.
- **Order sizing:** target weights converted to orders with a 0.5% minimum position weight, proportional plus fixed transaction fees accounted for.
- **Hard constraints (enforced every tick):**
  - Max 30 holdings (breach = disqualification)
  - Max 30% daily turnover (breach = disqualification)
  - Corporate-action handling (splits, dividends) with accounting corrections
- **Validation layer:** a separate validator checks every submitted solution against the competition's technical constraints before scoring.

## Repo layout

- `hackathon_agent/` - the trading agent: signal generation (`agent_candidate.py`), the cvxpy optimizer (`optimizer.py`), the constraint validator (`validate_solution.py`), and the run logs (`orders_log.json`, `portfolio_snapshots.json`, `results.json`).
- `starter-kit/` - the competition's original starter files, kept for reference.

## Sample local run (`results.json`)

| Metric | Value |
|---|---|
| Starting value | $10,000,000 |
| Final value | $10,305,650 |
| PnL | +$305,650 (+3.06%) |
| Ticks | 390 |
| Orders | 281 |
| Turnover | 0.29 (within the 0.30 cap) |
| Constraint checks | passed |

## Run it

```bash
cd hackathon_agent
./run_local.sh
```
