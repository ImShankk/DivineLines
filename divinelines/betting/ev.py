"""Expected value and edge quality.

The original implementation compared a model probability against a *single*
book's raw implied probability and reported EV in units of "dollars per $100".
Two problems: the raw implied probability includes the bookmaker's margin (so
every bet looked better than it was), and mixing percentage and currency units
made the numbers hard to compare across prices.

Here EV is always **per unit staked**, edge is always measured against the
**no-vig** market probability, and every quantity carries its definition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .odds_math import implied_probability


@dataclass
class EvResult:
    """Everything needed to judge one selection at one price."""

    model_probability: float
    market_probability: float | None      # no-vig consensus
    price_decimal: float
    #: Model probability minus no-vig market probability, in probability points.
    edge: float | None
    #: Profit per 1 unit staked, in units.  Positive means +EV.
    ev_per_unit: float
    #: Same number expressed as a percentage return on stake.
    ev_percent: float
    #: Probability at which this price breaks even.
    breakeven_probability: float
    #: Price the model thinks is fair (before any margin).
    fair_price: float
    #: How much better/worse the offered price is than the model's fair price.
    price_edge_percent: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()}


def expected_value(model_probability: float, price_decimal: float,
                   market_probability: float | None = None) -> EvResult:
    """EV of staking one unit on a selection.

    ``ev_per_unit = p * (d - 1) - (1 - p)``: win ``d - 1`` units with
    probability ``p``, lose the 1-unit stake otherwise.
    """
    p = float(model_probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"model probability {p} outside [0, 1]")
    d = float(price_decimal)
    if d <= 1.0:
        raise ValueError(f"decimal odds {d} must exceed 1.0")

    ev = p * (d - 1.0) - (1.0 - p)
    breakeven = implied_probability(d)
    fair_price = float("inf") if p <= 0 else 1.0 / p

    return EvResult(
        model_probability=p,
        market_probability=market_probability,
        price_decimal=d,
        edge=None if market_probability is None else p - float(market_probability),
        ev_per_unit=ev,
        ev_percent=ev * 100.0,
        breakeven_probability=breakeven,
        fair_price=fair_price,
        price_edge_percent=(d / fair_price - 1.0) * 100.0 if fair_price != float("inf") else -100.0,
    )


# ---------------------------------------------------------------- edge score

@dataclass
class EdgeScoreComponent:
    name: str
    score: float     # 0..1
    weight: float
    note: str | None = None


@dataclass
class EdgeScore:
    """A transparent 0-10 quality score for an opportunity.

    Not all +EV opportunities are equal: a 3% edge on a well-modelled game with
    fresh odds and a confirmed lineup is worth more than a 6% edge built on
    stale prices and an uncertain injury report.  Every component and weight is
    returned so the number can be audited rather than trusted.
    """

    score: float
    components: list[EdgeScoreComponent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "components": [
                {"name": c.name, "score": round(c.score, 3), "weight": c.weight, "note": c.note}
                for c in self.components
            ],
        }


def compute_edge_score(
    *,
    edge: float,
    model_confidence: float,
    data_quality: float,
    calibration_quality: float,
    model_agreement: float,
    market_liquidity: float,
    edge_reference: float = 0.06,
) -> EdgeScore:
    """Blend edge size with the reliability of the inputs behind it.

    ``edge_reference`` is the edge treated as "full marks" (6 percentage points
    by default); larger edges do not keep scoring higher, because beyond that
    point a large disagreement with the market is more often a model error than
    an opportunity.
    """
    components = [
        EdgeScoreComponent("edge_size", min(max(edge, 0.0) / edge_reference, 1.0), 0.30,
                           f"{edge * 100:.2f} pts vs {edge_reference * 100:.0f} reference"),
        EdgeScoreComponent("model_confidence", model_confidence, 0.20),
        EdgeScoreComponent("data_quality", data_quality / 100.0, 0.20),
        EdgeScoreComponent("calibration", calibration_quality, 0.15),
        EdgeScoreComponent("model_agreement", model_agreement, 0.10),
        EdgeScoreComponent("market_liquidity", market_liquidity, 0.05),
    ]
    total_weight = sum(c.weight for c in components)
    raw = sum(min(max(c.score, 0.0), 1.0) * c.weight for c in components) / total_weight
    return EdgeScore(score=raw * 10.0, components=components)


def market_liquidity_proxy(n_bookmakers: int, *, saturation: int = 8) -> float:
    """More books quoting a market is the best liquidity proxy we have."""
    return min(max(n_bookmakers, 0) / saturation, 1.0)
