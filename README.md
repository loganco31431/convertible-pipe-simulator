# Convertible Note & PIPE Dilution Simulator

Monte Carlo model of the instruments behind most small-cap structured PIPE financings, with
terms read from filed agreements:

- **Variable-price convertible notes, debentures and pre-paid advances.** The investor funds
  principal at a discount to face, then converts in tranches at the lower of a fixed price and a
  discount to VWAP, never below a floor. If the stock falls through the floor, the issuer repays
  in monthly installments instead.
- **Standby equity facilities (SEPA / equity lines).** The issuer sends advance notices and the
  investor buys newly issued shares at a discount to VWAP, priced over the same day (Option 1)
  or over three days (Option 2).
- **Loans with warrants.** A note repaid in cash on a schedule, with the investor's upside in
  warrants issued alongside it.

It answers two questions. How much of the investor's return comes from the discount versus the
stock, and what does that cost the issuer in dilution? And once the shares have to be sold into
a thin stock, how much of that return survives, and what is the best way to sell?

I built this after a summer at a structured finance fund evaluating live PIPE positions, to put
numbers on the intuition that these deals are priced off flow and structure more than off the
stock's direction.

## Who it is for

- **Issuers.** A CFO weighing a variable-price note usually has one number in mind: the share
  count the deal "should" cost at today's price. The model replaces that single number with a
  distribution. It shows how much more dilution the issuer takes than that base case once the
  stock is volatile or the pricing period is short, and what a floor price or a longer cadence
  buys back. That is the negotiation: discount, cadence, floor, and tranche size are the levers,
  and each one has a dilution cost the issuer can now put a number on.
- **Investors.** The same instrument can be valued three ways: as a bond with a conversion
  option, as a discounted-VWAP claim on future share sales, or as the equity itself. The model
  makes the second view explicit by simulating the conversion-and-sell path, so an investor can
  compare the IRR of running the structure against simply holding the stock, and see which
  terms (discount, cadence, floor) actually drive the return and which are noise.
- **The trading desk.** The trade-out page replays the term sheet on intraday bars and scores
  selling strategies against VWAP, the benchmark the purchase price is set from.

## Terms, and where they come from

`pipesim/presets.py` holds terms read from agreements between issuers and YA II PN, Ltd.
(Yorkville), filed with the SEC. Each preset cites its filing and lists what the filing did not
give and the model had to assume. Every term is an input, because terms differ by deal.

| Preset | Filing | Terms read from it |
|---|---|---|
| Pre-paid advance | SunPower 8-K filed 2026-01-30, exhibits 10.1 and 10.2 | Funded at 90% of face, 0% interest, 12 months. Converts at the lower of 125% of the prior day's VWAP and 93% of the lowest daily VWAP over the prior 5 days, floor about 20% of the reference price. VWAP below the floor on 5 of 7 days starts payments of $2.5M a month on $20M plus a 7% premium, from the 7th trading day, until VWAP is above the floor 10 days running. Equity-line advances offset the note. |
| Convertible debenture | NOVONIX 20-F filed 2026-02-26, exhibit 4.13 | Funded at 95% of face. Converts at the lower of 110% of the reference price and 95% of the lowest daily VWAP over the prior 5 days, floor 20% of the reference price. Same 5-of-7-day trigger. |
| Note plus warrants | PDS Biotechnology 8-Ks filed 2026-05-01 and 2026-06-15 | $6.0M face funded at $5.76M, 10% interest, 12 months, repaid in cash. Converts only after a missed payment. Warrants on 2,158,274 shares at $1.1824, exercisable after 6 months, 5-year term. |
| Loan plus warrants | Soluna 8-K filed 2026-04-17, exhibits 4.1, 4.2 and 10.2 | $12M funded at 95% of face, 5% interest, $1.2M a month from day 60 with a 5% premium. Warrants covering 21% of principal at the prior close, 12-month term. |
| Equity line, Option 1 | SunPower exhibit 10.1; Soluna 10-K filed 2026-03-30, exhibit 10.114 | 96% of the VWAP from the notice confirmation to the 4 PM close. Advance cut if volume is below advance / 0.30 (SunPower) or 0.35 (Soluna). |
| Equity line, Option 2 | Same | 97% of the lowest daily VWAP over 3 trading days from the notice. Days below the issuer's minimum acceptable price are excluded and cut the advance by a third each. |

Common to the agreements: a 4.99% ownership cap, a 19.99% exchange cap unless shareholders
approve more, advances capped at 100% of the prior 5-day average volume (SunPower), and a
short-sale ban that still allows selling shares the investor is unconditionally obligated to
buy under a pending notice. VWAP is defined as Bloomberg's.

## What it does

1. Pulls two years of prices from Yahoo Finance, calibrates annualized realized volatility, and
   reads shares outstanding (`pipesim/calibrate.py`). Manual inputs work offline.
2. Simulates daily closes with geometric Brownian motion, zero drift (`pipesim/paths.py`).
3. Runs the deal one trading day at a time, vectorized across paths (`pipesim/execution.py`).
   Each day it accrues interest on the outstanding balance, tests the floor trigger, makes any
   payment due, converts or draws within the ownership and exchange caps, sells, and prices any
   pending advance. `pipesim/engine.py` is the same engine with impact off and unlimited
   volume, so the frictionless and execution-aware models differ only in how shares get sold.
4. Solves annualized IRR per path by vectorized bisection and reports IRR percentiles, MOIC,
   probability of loss, and issuer dilution (`pipesim/metrics.py`). An equity line is reported
   as margin on dollars drawn: the investor sells before it pays, so it has no capital tied up
   and no meaningful IRR.
5. Values warrants by Black-Scholes (`pipesim/warrants.py`) and reports the deal with and
   without them, plus how much warrant value the investor's own selling costs it.
6. Sweeps volatility x discount and conversion cadence x floor price (`pipesim/scenarios.py`).

## Execution model: selling is not free

Put in a ticker and `trade.py` or the dashboard pulls price, volatility, shares outstanding and
average daily volume (`pipesim/market.py`), then sells the shares under real constraints:

- **Volume limit.** The investor sells at most `participation` of each day's volume (10% by
  default) and holds the rest as inventory. It converts only what it can sell before the next
  conversion.
- **Price impact (square-root law).** Selling q shares into daily volume V costs
  `eta x daily vol x sqrt(q / V)` on that day's fills.
- **Carried impact and recovery.** Half of that move carries past the day. Of the carried move,
  30% stays in the price for good and the rest fades with a 10-trading-day half-life, so the
  stock recovers part of the drop once the selling slows.
- **The investor moves its own terms.** Daily VWAP is taken as the day's price less half the
  investor's impact that day. Every VWAP-based term uses it, so heavy selling lowers the
  investor's own conversion price, raises the issuer's dilution, and can push the stock through
  the floor.
- **Best setup.** `pipesim/optimize.py` grid-searches conversion cadence, tranche size and
  selling speed, and ranks setups by median dollar profit among those with P(loss) of 10% or less.
- **Replay.** The same deal runs on the stock's actual last year of prices and volume, with the
  investor's own impact layered on top.

**Dashboard.** Double-click `Simulator.bat` (or run `python -m streamlit run app.py`). It opens on
the desk view and runs as soon as a ticker is entered; Run applies changed terms.

- **Trade-out** (default): pick a structure (Option 1, Option 2, note conversion). A strategy
  leaderboard scores five ways of selling against VWAP across every notice day in the data; the
  execution plan turns the chosen strategy into a half-hour schedule for a notice on the next
  trading day, with expected participation, cost and anything the volume cap leaves for a final
  trade; the day replay shows price, running VWAP and the desk's participation bar by bar; the
  liquidity tab shows the intraday volume pattern, daily dollar volume and days to sell at each
  participation rate.
- **Deal economics:** pick a structure from the filed deals, change any term. No impact vs
  realistic selling, warrants, where the principal goes (shares, cash early, cash at maturity),
  the replay on the last year of real prices, the optimizer and the size sweep.

Both pages export to Excel or CSV from the Export menu.

```
python trade.py --ticker OTLK
python trade.py --ticker GPRO --no-optimize --paths 3000
python trade.py --ticker OTLK --half-life inf        # impact never recovers
```

### Example: Outlook Therapeutics, $10M note, 10% discount, run 2026-09-28

Spot $0.62, realized vol 163%, 243.4M shares outstanding, about $8.8M of stock traded a day, so
the note is about 1.1 days of total volume. Simple variable-price note: 5% funding discount, 6%
interest, 10% discount to the 10-day average, $1M converted every 10 days, 4.99% and 19.99% caps.

| | Frictionless | Impact, no recovery | Impact with recovery |
|---|---|---|---|
| Median IRR | 96% | 77% | 82% |
| p10 IRR | 37% | 23% | 26% |
| Median dilution | 13.0% | 14.9% | 13.6% |
| Price drag from own selling, worst / at end | | 25% / 25% | 11% / 8% |

- **Liquidity decides how much of the discount the investor keeps.** Impact costs about 144 bps
  of every sale, and full exit takes about 150 trading days. On GoPro, which trades about
  $47M a day, the same note loses 5 points of IRR to impact (93% to 89%).
- **Small and frequent beats large.** The best setup converts 5% of principal every 5 days and
  sells at 5% of volume, for about $2.2M median profit.
- **The discount carries the trade on real data too.** Replayed on the last year of actual
  prices, the stock fell 3% over the term and the note still made $2.3M (139% IRR).

## Trade-out vs VWAP: can the desk beat VWAP inside the term sheet?

The dashboard's second page replays the term sheet on real intraday bars (`pipesim/intraday/`).
Every trading day in the data is tried as a notice day, and five selling strategies run on the
same days: even through the day (TWAP), follow the volume curve (VWAP), a fixed share of volume
(POV), front-loaded, and sell into strength. Each is scored on its average sale price against
the interval VWAP and against the VWAP that sets the purchase price (both in bps), how much of
the discount it kept, dollar profit, and how much had to be dumped at the end.

- **Three pricing rules:** Option 1, Option 2, and a note or pre-paid advance conversion, each
  as written in the filings above, including the notice time, the volume threshold, excluded
  days, the fixed price and the floor.
- **The desk sets its own price.** In a VWAP-priced advance the desk's sales are in the VWAP that
  sets its purchase price. The model recomputes VWAP with the desk's trades in it and reports
  how much its selling moved the purchase price.
- **The final size can change.** Under Option 1 the advance shrinks when volume is light; under
  Option 2 it shrinks when a day is excluded. What the desk still holds once the final size is
  known is sold in a clean-up trade after the pricing period, outside the pricing VWAP.
- **Impact:** square-root per bar, scaled so a steady day of selling at share pi of volume ends
  impact strength x daily vol x sqrt(pi) lower, the same law as the daily model. Part stays, part
  fades with an intraday half-life. Placeholders until calibrated on real fills.
- **Data:** Yahoo 5-minute bars (about 60 trading days), a Bloomberg export (Excel or CSV with a
  time column, OHLC or last price, and volume), or a live Bloomberg terminal through `blpapi`
  (IntradayBarRequest, about 140 business days of bars). Bloomberg bars carry dollars traded, so
  the bar price is the bar's true VWAP; Yahoo bars use (high + low + close) / 3 as a proxy. The
  Bloomberg live path is written but has not been run on a terminal yet.
- **Lookahead:** each sale uses earlier bars' prices, the running VWAP so far, and the volume
  pattern and volatility from days before the notice. The bar's own volume is used only as the
  participation cap. Tests shock prices and volumes after a given bar and check that no earlier
  sale changes. The notice days overlap and come from one stretch of one stock, so small
  differences between strategies are noise.

### Example: Outlook Therapeutics, $1M advance, Yahoo 5-minute bars, 2026-07-06 to 2026-09-28

Placeholder impact settings, 25 bps half spread, desk capped at 15% of each bar's volume.

| Pricing rule | Notice days | Best strategy by profit | Median profit | Sale price vs VWAP | Discount kept |
|---|---|---|---|---|---|
| Option 1, 96% of same-day VWAP, sold in 1 day | 50 | Share of volume | $32,000 | 36 bps below | 0.90x |
| Option 2, 97% of lowest VWAP over 3 days | 48 | Sell into strength | $58,000 | 33 bps below | 2.01x |
| Conversion at 93% of lowest VWAP, prior 5 days | 45 | Share of volume | $126,000 | 70 bps above | 1.68x |

- **Option 1 is the hard one.** The purchase price is set by the same session the desk sells
  into, so every basis point below VWAP comes straight out of the 4% discount. All five
  strategies sold 36 to 60 bps below VWAP and kept 0.84x to 0.90x of the discount. Light volume
  cut the advance by 8% on average.
- **Option 2 and conversions pay for the lowest-VWAP rule.** Pricing off the lowest of several
  days is worth more than the stated discount, which is why "discount kept" is above 1.
- **The desk's own selling lowered its purchase price** by about 0.8% under Option 1 and 1.3%
  under Option 2 at the median.

## Run it

```
pip install -r requirements.txt
python run.py --ticker GPRO
python run.py --ticker LCID --principal 25e6 --discount 0.12 --floor-pct 0.5
python run.py --s0 3.20 --sigma 0.85 --shares-out 150e6      # offline
python -m pytest tests
```

## Example: GoPro, $10M note, 10% discount to 10-day VWAP, converting $1M every 10 days

Calibrated on 2026-09-28: spot $1.33, realized vol 114%, 206.4M shares outstanding.
Frictionless model, 5% funding discount, 6% interest.

| | p10 | p50 | p90 |
|---|---|---|---|
| Investor IRR | 63% | 93% | 137% |
| Issuer dilution | | 5.1% | 11.8% |

![IRR distribution](output/GPRO_irr_hist.png)

![Volatility vs discount](output/GPRO_grid.png)

### What the grid shows

- **Volatility barely moves the investor's IRR.** Across 30% to 120% vol, median IRR at a
  10% discount stays between 93% and 95%. The return is a fee on flow, captured at every
  conversion, not a directional bet on the stock.
- **The discount is the whole trade.** Moving from 5% to 20% takes median IRR from about 50% to
  about 240%.
- **Volatility is the issuer's problem.** Dilution rises with vol because more conversions
  land at depressed VWAPs, so the issuer hands over more shares per dollar.
- **Cadence sets IRR, not MOIC.** Converting every 5 days versus every 21 days takes median IRR
  from about 39% to about 255% on a similar cash multiple, because capital recycles faster.
  This is why these structures push for short pricing periods.
- **Floors protect the issuer at the investor's expense.** A floor at 75% of spot cuts the p10
  IRR from 63% to 13% and leaves part of the note unconverted at maturity.

## Assumptions and limits

These are the things I would want to be asked about.

- **No issuer default.** Every cash payment and the repayment at maturity are assumed to be
  made. For a company that needs this kind of financing, not getting paid is the main risk, so
  the returns on a loan, and on any path where the stock falls through the floor, are a
  ceiling. Probability of loss is near zero here for that reason.
- **Impact parameters are placeholders, not fitted.** The square-root coefficient (0.5), the
  carried share (50%), the part that never fades (30%) and the 10-day half-life are
  textbook-scale values. None is fitted to real PIPE selling. The right calibration is a desk's
  own record of how much its selling moved a stock and how fast the price came back. Every one
  is an input, so the results can be checked across a range.
- **Spread and fees** are a flat 2% of every sale in the daily model, on top of impact. That is
  high for a liquid stock and about right for a sub-dollar one that trades in one-cent ticks.
- **Daily VWAP is a proxy.** The daily model has closes, not intraday prices, so VWAP is the
  day's price less half the investor's impact. Simulated volume is lognormal around the average
  and independent across days; it does not rise on down days or react to the investor's selling.
- **Zero-drift GBM.** No jumps and no delisting.
- **Conversion pace is a choice, not a term.** The model converts a fixed tranche on a fixed
  cadence when it is profitable, capped at what can be sold before the next conversion.
- **Warrants** are valued by Black-Scholes with volatility capped at 100% by default and a 4%
  risk-free rate. At microcap volatility, uncapped Black-Scholes values a long-dated warrant
  close to the stock itself. The valuation ignores the ownership blocker, the cost of selling
  the exercised shares, and anti-dilution adjustments.
- **Not modeled:** registration delays, events of default and default interest, the investor's
  right to accelerate payments, mandatory redemption on a new financing, the closing auction as
  a separate venue, and any signaling effect from the market knowing the investor is selling.
- **Terms come from six filings.** They vary by deal, and some presets fill gaps with
  assumptions, listed in `presets.py` and shown in the dashboard.
- **No lookahead.** Every decision on day t uses closes and volume up to day t only, the floor
  test uses VWAPs through day t-1, and impact from day t's sales reaches prices from day t+1 on.
  Option 1 and Option 2 purchase prices are set after the fact by contract, and the cash is
  booked the day after the pricing period ends. Tests shock prices after a given day or bar and
  check that nothing before it changes. The replay has a separate leak: volatility and volume
  are calibrated on the same window it replays, and the fixed price, floor and warrant strike
  are set from today's price.

## Layout

```
pipesim/
  instruments.py   ConvertibleNote, StandbyEquityFacility and Warrant terms
  presets.py       terms read from filed agreements, with sources
  paths.py         GBM simulation and vectorized trailing VWAP
  execution.py     the deal engine: contract terms, volume limits, price impact
  engine.py        frictionless model: the same engine with impact off
  warrants.py      Black-Scholes warrant valuation
  metrics.py       IRR by bisection, margin, summary statistics
  calibrate.py     yfinance vol and shares-outstanding calibration
  market.py        yfinance price, vol, shares outstanding and volume
  scenarios.py     volatility x discount grid, cadence x floor stress
  optimize.py      cadence x tranche x selling-speed grid search
  intraday/data.py      intraday bars from Yahoo, a Bloomberg export, or a Bloomberg terminal
  intraday/tradeout.py  pricing rules, selling strategies, VWAP scoring
views/             dashboard pages (deal economics, trade-out vs VWAP)
app.py             dashboard entry point
run.py             CLI: prints tables, writes charts to output/
trade.py           CLI: ticker in, frictionless vs execution-aware, best setup, historical replay
tests/             42 tests: contract terms, engine, metrics, no lookahead
```
