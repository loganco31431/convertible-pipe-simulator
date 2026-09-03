# Convertible Note & PIPE Dilution Simulator

Monte Carlo model of the two instruments behind most small-cap structured PIPE financings:

- **Variable-price convertible notes.** The investor funds principal at an original issue
  discount, then converts in tranches at a discount to the trailing VWAP and sells the shares.
  A floor price can pause conversion; unconverted principal is repaid in cash at maturity.
- **Standby equity facilities (SEPA / equity lines).** The issuer draws cash on demand and the
  investor buys newly issued shares at a discount to the pricing-period VWAP.

The question it answers: how much of the investor's return comes from the discount versus the
stock, and what does that cost the issuer in dilution, across volatility, discount, conversion
cadence, and floor-price scenarios.

I built this after a summer at a structured finance fund evaluating live PIPE positions, to put
numbers on the intuition that these deals are priced off flow and structure more than off the
stock's direction.

## What it does

1. Pulls two years of prices from Yahoo Finance, calibrates annualized realized volatility, and
   reads shares outstanding (`pipesim/calibrate.py`). Manual inputs work offline.
2. Simulates daily closes with geometric Brownian motion, zero drift (`pipesim/paths.py`).
3. Runs the conversion schedule on every path at once (`pipesim/engine.py`): on each conversion
   day the conversion price is `(1 - discount) x trailing VWAP`, floored; the investor converts
   a tranche if selling at market clears the conversion price, and sells at the close less
   slippage.
4. Solves annualized IRR per path by vectorized bisection and reports IRR percentiles, MOIC,
   probability of loss, and issuer dilution (`pipesim/metrics.py`).
5. Sweeps volatility x discount and conversion cadence x floor price (`pipesim/scenarios.py`).

## Run it

```
pip install -r requirements.txt
python run.py --ticker GPRO
python run.py --ticker LCID --principal 25e6 --discount 0.12 --floor-pct 0.5
python run.py --s0 3.20 --sigma 0.85 --shares-out 150e6      # offline
python -m pytest tests
```

## Example: GoPro, $10M note, 10% discount to 10-day VWAP, converting $1M every 10 days

Calibrated on 2026-09-03: spot $1.39, realized vol 118%, 184.5M shares outstanding.

| | p10 | p50 | p90 |
|---|---|---|---|
| Investor IRR | 74% | 109% | 158% |
| Issuer dilution | | 5.6% | 13.3% |

![IRR distribution](output/GPRO_irr_hist.png)

![Volatility vs discount](output/GPRO_grid.png)

### What the grid shows

- **Volatility barely moves the investor's IRR.** Across 30% to 120% vol, median IRR at a
  10% discount stays near 110% to 118%. The return is a fee on flow, captured at every
  conversion, not a directional bet on the stock.
- **The discount is the whole trade.** Moving from 5% to 20% roughly quadruples IRR.
- **Volatility is the issuer's problem.** Dilution rises with vol because more conversions
  land at depressed VWAPs, so the issuer hands over more shares per dollar.
- **Cadence sets IRR, not MOIC.** Converting every 5 days versus every 21 days takes median IRR
  from about 44% to about 290% on the same cash multiple, because capital recycles faster.
  This is why these structures push for short pricing periods.
- **Floors protect the issuer at the investor's expense.** A floor at 75% of spot cuts the p10
  IRR sharply and leaves part of the note unconverted at maturity.

## Assumptions and limits

These are the things I would want to be asked about.

- **No price impact.** The investor sells into the market at the close less a fixed slippage
  haircut. In reality selling 5% of a microcap's float pushes the price down, which is the main
  reason real PIPE returns are lower and dilution is higher than this model shows.
- **VWAP is a trailing mean of closes.** Volume is not simulated.
- **Zero-drift GBM.** No jumps, no default, no delisting. Probability of loss is near zero here
  because every permitted conversion locks in the discount; the real loss case is the issuer
  failing before the note converts.
- **Conversion rule is mechanical.** A fixed tranche every `cadence` days when profitable. A
  real desk sizes to volume and daily conversion caps.
- **No lookahead.** Every decision on day t uses closes up to day t only, so the backtest logic
  cannot leak future prices into past conversions.

## Layout

```
pipesim/
  instruments.py   ConvertibleNote and StandbyEquityFacility dataclasses
  paths.py         GBM simulation and vectorized trailing VWAP
  engine.py        conversion engine, vectorized across paths
  metrics.py       IRR by bisection, summary statistics
  calibrate.py     yfinance vol and shares-outstanding calibration
  scenarios.py     volatility x discount grid, cadence x floor stress
run.py             CLI: prints tables, writes charts to output/
tests/             engine and metric sanity checks
```
