"""Instrument terms. Field names follow the language in filed Yorkville (YA II PN) agreements;
`presets.py` holds terms read from specific filings."""
from dataclasses import dataclass

TRADING_DAYS = 252
MONTH = 21           # trading days between monthly payments


@dataclass(frozen=True)
class ConvertibleNote:
    """A variable-price convertible note, debenture or pre-paid advance.

    The investor funds `principal` at an original issue discount (`oid`), then converts in
    tranches. Conversion price = the lower of `fixed_price` and the variable price, where the
    variable price is (1 - discount) x the VWAP measure and never below `floor_price`. When
    the stock trades below the floor, conversion stops; depending on `payment_mode` the issuer
    then repays in cash. Set `tranche=0` for a plain loan that never converts.
    """

    principal: float = 10_000_000.0
    oid: float = 0.05                 # investor pays principal * (1 - oid)
    coupon: float = 0.06              # annual, accrues daily on outstanding principal
    discount: float = 0.10            # discount to the VWAP measure at conversion
    vwap_lookback: int = 10           # trading days in the VWAP window
    cadence: int = 10                 # trading days between conversions
    tranche: float = 1_000_000.0      # principal converted per conversion (0 = never converts)
    floor_price: float = 0.0          # variable price never below this (0 = none)
    maturity_days: int = TRADING_DAYS # note tenor in trading days
    sell_slippage: float = 0.02       # haircut to market when the investor sells converted shares
    pricing: str = "average"          # "average" of the window incl. today, or "lowest" daily VWAP of the days before the notice
    fixed_price: float = 0.0          # conversion price never above this (0 = none)
    payment_mode: str = "none"        # "none", "scheduled" (loan-style), "on_trigger" (amortization event)
    monthly_payment: float = 0.0      # principal repaid per monthly payment
    payment_premium: float = 0.0      # premium on principal repaid in cash
    payment_start_day: int = 42       # scheduled: first payment (60 calendar days is about 42 trading days)
    trigger_days: int = 5             # amortization event: VWAP below the floor on this many days...
    trigger_window: int = 7           # ...out of this many consecutive trading days
    trigger_lag: int = 7              # first payment this many trading days after the event
    cure_days: int = 10               # event ends on this consecutive day with VWAP above the floor
    pay_in_shares: bool = False       # payments settled by equity-line advances that offset the note
    sepa_discount: float = 0.03       # discount on those advances
    ownership_cap: float = 0.0499     # max share of shares outstanding held at once (0 = none)
    exchange_cap: float = 0.1999      # max shares issued as a share of shares outstanding (0 = none)
    structuring_fee: float = 0.0      # cash paid to the investor at closing

    @property
    def kind(self) -> str:
        return "convertible_note"


@dataclass(frozen=True)
class StandbyEquityFacility:
    """A standby equity purchase agreement (SEPA / equity line).

    The issuer sends an advance notice every `cadence` days, up to `commitment`, and the
    investor buys newly issued shares at a discount to VWAP and sells them into the market.
    `pricing` selects how the purchase price is set:

    - "option1": (1 - discount) x VWAP from the notice to the close that day. If volume that
      session is below advance / `volume_threshold`, the advance is cut to the larger of
      `volume_threshold` x volume and what the investor sold.
    - "option2": (1 - discount) x the lowest daily VWAP over `vwap_lookback` trading days
      starting on the notice day. A day with VWAP below `min_draw_price` is excluded and cuts
      the advance by 1 / `vwap_lookback`; shares the investor sold that day are bought at
      (1 - discount) x `min_draw_price`.
    - "trailing": (1 - discount) x the average of the last `vwap_lookback` days, known at the
      notice. Not a filed structure; kept as the simple baseline.
    """

    commitment: float = 50_000_000.0
    advance: float = 2_000_000.0
    discount: float = 0.05
    vwap_lookback: int = 3
    cadence: int = 5
    min_draw_price: float = 0.0       # issuer's minimum acceptable price
    maturity_days: int = 2 * TRADING_DAYS
    sell_slippage: float = 0.02
    pricing: str = "trailing"
    volume_threshold: float = 0.30    # option1 (0 = none)
    max_advance_adv: float = 0.0      # advance capped at this multiple of the prior 5-day avg volume (0 = none)
    ownership_cap: float = 0.0499
    exchange_cap: float = 0.1999
    commitment_shares: float = 0.0    # shares issued to the investor as the commitment fee
    structuring_fee: float = 0.0      # cash paid to the investor at signing

    @property
    def kind(self) -> str:
        return "sepa"


@dataclass(frozen=True)
class Warrant:
    """Warrants issued alongside a note. Valued by Black-Scholes; see `warrants.py`."""

    shares: float
    strike: float
    term_days: int = 5 * TRADING_DAYS   # expiry, trading days after closing
    exercisable_after: int = 0          # trading days before exercise is allowed
    vol: float = 0.0                    # valuation volatility (0 = the stock's realized vol, capped at `vol_cap`)
    vol_cap: float = 1.0
    rate: float = 0.04                  # risk-free rate assumption

    @classmethod
    def from_coverage(cls, principal: float, coverage: float, s0: float, strike_pct: float = 1.0, **kw):
        """`coverage` x principal of stock at the reference price, struck at `strike_pct` x that price."""
        return cls(shares=coverage * principal / s0, strike=strike_pct * s0, **kw)
