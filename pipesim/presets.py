"""Deal terms read from filed YA II PN, Ltd. (Yorkville) agreements.

Every preset is built from the stock's current price, because the contracts set the fixed
price, floor and warrant strike off the price when the deal was signed. `source` cites the
filing. `assumed` lists anything the filing did not give and the model fills in.
"""
from dataclasses import dataclass

from .instruments import MONTH, TRADING_DAYS, ConvertibleNote, StandbyEquityFacility, Warrant


@dataclass(frozen=True)
class Preset:
    name: str
    source: str
    assumed: str
    note: ConvertibleNote | None = None
    sepa: StandbyEquityFacility | None = None
    warrant: Warrant | None = None


def sunpower_prepaid(s0: float, principal: float = 20e6) -> Preset:
    return Preset(
        name="Pre-paid advance (SunPower, Jan 2026)",
        source="SunPower 8-K filed 2026-01-30, exhibits 10.1 (SEPA) and 10.2 (convertible promissory note)",
        assumed="Conversion pace (10% of principal every 5 trading days) is the investor's choice, not a term. "
                "Amortization is scaled from $2.5M a month on $20M.",
        note=ConvertibleNote(
            principal=principal, oid=0.10, coupon=0.0, discount=0.07, vwap_lookback=5, pricing="lowest",
            fixed_price=1.25 * s0, floor_price=0.20 * s0, cadence=5, tranche=0.10 * principal,
            maturity_days=TRADING_DAYS, payment_mode="on_trigger", monthly_payment=0.125 * principal,
            payment_premium=0.07, trigger_days=5, trigger_window=7, trigger_lag=7, cure_days=10,
            pay_in_shares=True, sepa_discount=0.03),
    )


def novonix_debenture(s0: float, principal: float = 100e6) -> Preset:
    return Preset(
        name="Convertible debenture (NOVONIX, Feb 2026)",
        source="NOVONIX 20-F filed 2026-02-26, exhibit 4.13",
        assumed="Interest rate, payment premium and payment size were not read from the filing: 0% interest, "
                "no premium and 12.5% of principal a month are placeholders. The 18-month term is read for one "
                "tranche only. The contract's ownership limit is 19.99% (Australian takeover threshold), used here "
                "for both caps.",
        note=ConvertibleNote(
            principal=principal, oid=0.05, coupon=0.0, discount=0.05, vwap_lookback=5, pricing="lowest",
            fixed_price=1.10 * s0, floor_price=0.20 * s0, cadence=5, tranche=0.10 * principal,
            maturity_days=int(1.5 * TRADING_DAYS), payment_mode="on_trigger", monthly_payment=0.125 * principal,
            payment_premium=0.0, ownership_cap=0.1999, exchange_cap=0.1999),
    )


def pds_note_warrant(s0: float, principal: float = 6e6) -> Preset:
    return Preset(
        name="Note plus warrants (PDS Biotech, Jun 2026)",
        source="PDS Biotechnology 8-Ks filed 2026-05-01 and 2026-06-15",
        assumed="The payment schedule is not in the 8-K: equal monthly payments from day 60 to maturity are assumed. "
                "The note converts only after a missed payment, so conversion is off. The warrant strike is set at "
                "the current price; the filing gives $1.1824 without saying how it was set.",
        note=ConvertibleNote(
            principal=principal, oid=0.04, coupon=0.10, tranche=0.0, maturity_days=TRADING_DAYS,
            payment_mode="scheduled", monthly_payment=principal / 10, payment_premium=0.0, payment_start_day=42),
        warrant=Warrant.from_coverage(principal, 2_158_274 * 1.1824 / 6e6, s0, term_days=5 * TRADING_DAYS,
                                      exercisable_after=6 * MONTH),
    )


def soluna_loan_warrant(s0: float, principal: float = 12e6) -> Preset:
    return Preset(
        name="Loan plus warrants (Soluna, Apr 2026)",
        source="Soluna Holdings 8-K filed 2026-04-17, exhibits 4.1 (note), 4.2 (warrant) and 10.2 (purchase agreement)",
        assumed="The 13-month term is approximated as 273 trading days.",
        note=ConvertibleNote(
            principal=principal, oid=0.05, coupon=0.05, tranche=0.0, maturity_days=273,
            payment_mode="scheduled", monthly_payment=0.10 * principal, payment_premium=0.05, payment_start_day=42),
        warrant=Warrant.from_coverage(principal, 0.21, s0, term_days=TRADING_DAYS, exercisable_after=0),
    )


def sepa_option1(s0: float, commitment: float = 25e6, advance: float = 1e6) -> Preset:
    return Preset(
        name="Equity line, Option 1: same-day VWAP (SunPower, Soluna 2026)",
        source="SunPower 8-K filed 2026-01-30, exhibit 10.1; Soluna 10-K filed 2026-03-30, exhibit 10.114",
        assumed="Advance size and how often the issuer draws are the issuer's choice. Commitment shares are scaled "
                "from SunPower's 175,000 shares on $25M, and the structuring fee is its $50,000.",
        sepa=StandbyEquityFacility(
            commitment=commitment, advance=advance, discount=0.04, vwap_lookback=1, cadence=5, pricing="option1",
            volume_threshold=0.30, max_advance_adv=1.0, maturity_days=2 * TRADING_DAYS,
            commitment_shares=175_000 * commitment / 25e6, structuring_fee=50_000.0),
    )


def sepa_option2(s0: float, commitment: float = 25e6, advance: float = 1e6) -> Preset:
    return Preset(
        name="Equity line, Option 2: lowest VWAP over 3 days (SunPower, Soluna 2026)",
        source="SunPower 8-K filed 2026-01-30, exhibit 10.1; Soluna 10-K filed 2026-03-30, exhibit 10.114",
        assumed="Advance size and how often the issuer draws are the issuer's choice. Commitment shares are scaled "
                "from SunPower's 175,000 shares on $25M, and the structuring fee is its $50,000.",
        sepa=StandbyEquityFacility(
            commitment=commitment, advance=advance, discount=0.03, vwap_lookback=3, cadence=5, pricing="option2",
            max_advance_adv=1.0, maturity_days=2 * TRADING_DAYS,
            commitment_shares=175_000 * commitment / 25e6, structuring_fee=50_000.0),
    )


def generic_note(s0: float, principal: float = 10e6) -> Preset:
    return Preset(name="Simple variable-price note", source="Not from a filing", assumed="All terms are inputs.",
                  note=ConvertibleNote(principal=principal, tranche=0.10 * principal))


PRESETS = {
    "Pre-paid advance (SunPower)": sunpower_prepaid,
    "Convertible debenture (NOVONIX)": novonix_debenture,
    "Note plus warrants (PDS)": pds_note_warrant,
    "Loan plus warrants (Soluna)": soluna_loan_warrant,
    "Equity line, Option 1": sepa_option1,
    "Equity line, Option 2": sepa_option2,
    "Simple variable-price note": generic_note,
}
