from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class ConvertibleNote:
    """A variable-price convertible note, the instrument behind most small-cap structured PIPEs.

    The investor funds `principal` at an original issue discount (`oid`), then converts in
    tranches at a discount to the trailing VWAP. A `floor_price` caps how low the conversion
    price can go; when the stock trades below the floor, conversion pauses and the note is
    repaid in cash at maturity with `coupon` accrued.
    """

    principal: float = 10_000_000.0
    oid: float = 0.05                 # investor pays principal * (1 - oid)
    coupon: float = 0.06              # annual, paid in cash at maturity on unconverted principal
    discount: float = 0.10            # discount to VWAP at conversion
    vwap_lookback: int = 10           # trading days in the VWAP window
    cadence: int = 10                 # trading days between conversions
    tranche: float = 1_000_000.0      # principal converted per conversion
    floor_price: float = 0.0          # minimum conversion price (0 = none)
    maturity_days: int = TRADING_DAYS # note tenor in trading days
    sell_slippage: float = 0.02       # haircut to market when the investor sells converted shares

    @property
    def kind(self) -> str:
        return "convertible_note"


@dataclass
class StandbyEquityFacility:
    """A standby equity purchase agreement (SEPA / equity line).

    The issuer draws `advance` dollars every `cadence` days, up to `commitment`, and the
    investor buys newly issued shares at a discount to the pricing-period VWAP, selling them
    into the market. No principal at risk up front, no coupon, no floor.
    """

    commitment: float = 50_000_000.0
    advance: float = 2_000_000.0
    discount: float = 0.05
    vwap_lookback: int = 3
    cadence: int = 5
    min_draw_price: float = 0.0       # issuer will not draw below this price
    maturity_days: int = 2 * TRADING_DAYS
    sell_slippage: float = 0.02

    @property
    def kind(self) -> str:
        return "sepa"
