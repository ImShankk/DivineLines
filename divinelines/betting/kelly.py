"""Kelly staking, with uncertainty and caps.

Full Kelly maximises long-run growth *only* if the probability is exactly
right.  It is not, so the platform never stakes full Kelly: the fraction is
configurable, the probability is shrunk toward the market in proportion to how
uncertain the model is, and hard caps bound the result.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ..config import settings


@dataclass
class StakeRecommendation:
    kelly_full: float          # fraction of bankroll at full Kelly
    kelly_used: float          # after fractional Kelly + uncertainty shrink
    stake: float               # currency amount
    stake_pct: float           # fraction of bankroll
    capped_by: str | None      # which constraint bound the stake, if any
    adjusted_probability: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()}


def kelly_fraction(probability: float, price_decimal: float) -> float:
    """Full-Kelly fraction of bankroll: ``(p*d - 1) / (d - 1)``.

    Returns 0 for a non-positive edge — Kelly never recommends a -EV bet.
    """
    p = float(probability)
    d = float(price_decimal)
    if d <= 1.0:
        raise ValueError(f"decimal odds {d} must exceed 1.0")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability {p} outside [0, 1]")
    edge = p * d - 1.0
    if edge <= 0:
        return 0.0
    return edge / (d - 1.0)


def shrink_probability(model_probability: float, market_probability: float | None,
                       confidence: float) -> float:
    """Blend the model toward the market when confidence is low.

    ``confidence`` of 1 keeps the model probability untouched; 0 defers
    entirely to the market.  This is a staking-time adjustment: it never
    changes the probability the model reports, only how much is risked on it.
    """
    if market_probability is None:
        return float(model_probability)
    weight = min(max(float(confidence), 0.0), 1.0)
    return weight * float(model_probability) + (1.0 - weight) * float(market_probability)


def recommend_stake(
    *,
    model_probability: float,
    price_decimal: float,
    market_probability: float | None = None,
    confidence: float = 1.0,
    bankroll: float | None = None,
    kelly_multiplier: float | None = None,
    max_stake_pct: float | None = None,
) -> StakeRecommendation:
    """Fractional, uncertainty-aware Kelly stake with hard caps."""
    bankroll = settings.betting.bankroll if bankroll is None else float(bankroll)
    multiplier = settings.betting.kelly_fraction if kelly_multiplier is None else float(kelly_multiplier)
    cap_pct = settings.betting.max_stake_pct if max_stake_pct is None else float(max_stake_pct)

    notes: list[str] = []
    adjusted = shrink_probability(model_probability, market_probability, confidence)
    if market_probability is not None and confidence < 1.0:
        notes.append(
            f"probability shrunk toward market ({model_probability:.3f} -> {adjusted:.3f}) "
            f"at confidence {confidence:.2f}"
        )

    full = kelly_fraction(adjusted, price_decimal)
    used = full * multiplier
    capped_by: str | None = None

    if used > cap_pct:
        used = cap_pct
        capped_by = "max_stake_pct"
        notes.append(f"stake capped at {cap_pct:.2%} of bankroll")

    stake = round(used * bankroll, 2)
    return StakeRecommendation(
        kelly_full=full,
        kelly_used=used,
        stake=stake,
        stake_pct=used,
        capped_by=capped_by,
        adjusted_probability=adjusted,
        notes=notes,
    )
