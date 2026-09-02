"""Closing line value — the platform's single CLV convention.

Profit over a few hundred bets is mostly noise. CLV — whether the price we took
was better than the price the market settled on — converges far faster, and is
the strongest short-horizon evidence that a model is finding real information.

**It is not profit, and this module never implies that it is.** A bet can lose
while beating the close, and win while losing to it.

## Sign convention (used everywhere: CLI, database, API, backtester, frontend)

```
clv_price_pct = (entry_odds / closing_odds - 1) * 100
```

Positive means the price we took was *longer* than the close — the market moved
toward our side after we priced it. Some references write this as
``closing / entry - 1``, which produces the same magnitude with the opposite
sign; that reads "the price drifted after we bet", so a good outcome would show
as negative. Practitioners mean the first form when they say "positive CLV", so
that is what the platform uses, in exactly one place, and every consumer imports
it from here rather than re-deriving it.

The probability form is reported alongside, because a price comparison ignores
the bookmaker's margin: a close of 2.00 in a 108% market is not the same claim
as a close of 2.00 in a 102% market.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .closing_line import ClosingLine
from .odds_math import implied_probability, log_odds, remove_vig


@dataclass
class ClvResult:
    taken_price: float
    closing_price: float
    taken_probability: float          # gross implied at the taken price
    closing_fair_probability: float   # no-vig closing probability
    clv_price_pct: float
    clv_prob_points: float
    clv_log_odds: float
    beat_close: bool
    closing_source: str | None = None
    closing_book: str | None = None
    closing_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()}


def compute_clv(taken_price: float, closing_price: float, *,
                closing_fair_probability: float | None = None,
                closing_source: str | None = None,
                closing_book: str | None = None,
                closing_policy: str | None = None) -> ClvResult:
    """CLV for one selection from two decimal prices.

    ``closing_fair_probability`` should be supplied whenever the full closing
    market is known, so the probability-space figure is margin-free.
    """
    taken_price = float(taken_price)
    closing_price = float(closing_price)
    if taken_price <= 1.0 or closing_price <= 1.0:
        raise ValueError("decimal prices must exceed 1.0")

    taken_probability = implied_probability(taken_price)
    fair = (float(closing_fair_probability) if closing_fair_probability is not None
            else implied_probability(closing_price))

    return ClvResult(
        taken_price=taken_price,
        closing_price=closing_price,
        taken_probability=taken_probability,
        closing_fair_probability=fair,
        clv_price_pct=(taken_price / closing_price - 1.0) * 100.0,
        clv_prob_points=(fair - taken_probability) * 100.0,
        clv_log_odds=log_odds(fair) - log_odds(taken_probability),
        beat_close=taken_price > closing_price,
        closing_source=closing_source, closing_book=closing_book,
        closing_policy=closing_policy,
    )


def clv_against_close(taken_price: float, selection: str, close: ClosingLine) -> ClvResult:
    """CLV against a resolved :class:`ClosingLine`."""
    closing_price = close.price_for(selection)
    if closing_price is None:
        raise ValueError(f"closing line has no price for selection '{selection}'")
    return compute_clv(
        taken_price, closing_price,
        closing_fair_probability=close.novig_probabilities.get(selection),
        closing_source=close.source, closing_book=close.bookmaker,
        closing_policy=close.policy,
    )


def closing_line_value(taken_price: float, closing_prices: Mapping[str, float],
                       selection: str, *, devig_method: str = "power") -> ClvResult:
    """CLV against a raw closing market (kept for existing callers and tests)."""
    if selection not in closing_prices:
        raise ValueError(f"closing prices missing selection '{selection}'")
    selections = list(closing_prices.keys())
    fair = dict(zip(selections, remove_vig([closing_prices[s] for s in selections], devig_method)))
    return compute_clv(taken_price, float(closing_prices[selection]),
                       closing_fair_probability=fair[selection])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class ClvSummary:
    n: int
    mean_clv_price_pct: float
    median_clv_price_pct: float
    mean_clv_prob_points: float
    beat_close_rate: float
    std_clv_price_pct: float = 0.0
    ci_low: float | None = None
    ci_high: float | None = None
    percentiles: dict[str, float] | None = None
    #: True only when the interval excludes zero *and* the sample is large
    #: enough to mean anything. Deliberately conservative.
    significant: bool = False
    interpretation: str = "insufficient sample"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in payload.items()}


#: Below this, the platform refuses to characterise a CLV number at all.
MIN_SAMPLE_FOR_INFERENCE = 30


def summarise_clv(results: Sequence[ClvResult] | Sequence[float],
                  *, confidence: float = 0.95) -> ClvSummary:
    """Distribution, not just a mean.

    A handful of large observations can drag an average positive while the
    typical bet loses to the close, so the median, spread and interval are
    reported alongside — and nothing is called an edge on a small sample.
    """
    if not results:
        return ClvSummary(0, 0.0, 0.0, 0.0, 0.0)

    if isinstance(results[0], ClvResult):
        price = np.array([r.clv_price_pct for r in results], dtype=float)
        points = np.array([r.clv_prob_points for r in results], dtype=float)
        beat = np.array([r.beat_close for r in results], dtype=bool)
    else:
        price = np.asarray(results, dtype=float)
        points = np.full_like(price, np.nan)
        beat = price > 0

    n = len(price)
    mean = float(price.mean())
    std = float(price.std(ddof=1)) if n > 1 else 0.0

    ci_low = ci_high = None
    significant = False
    if n >= MIN_SAMPLE_FOR_INFERENCE and std > 0:
        # Normal-approximation interval on the mean. With n >= 30 and the
        # bounded-ish spread of CLV percentages this is adequate; the point of
        # showing it is to stop a +2% mean on 40 bets being read as proof.
        z = 1.959963985 if abs(confidence - 0.95) < 1e-9 else 2.575829304
        margin = z * std / math.sqrt(n)
        ci_low, ci_high = mean - margin, mean + margin
        significant = ci_low > 0 or ci_high < 0

    if n < MIN_SAMPLE_FOR_INFERENCE:
        interpretation = f"insufficient sample (n={n}, need {MIN_SAMPLE_FOR_INFERENCE})"
    elif not significant:
        interpretation = "consistent with zero CLV"
    elif mean > 0:
        interpretation = "positive CLV, interval excludes zero"
    else:
        interpretation = "negative CLV, interval excludes zero"

    return ClvSummary(
        n=n,
        mean_clv_price_pct=mean,
        median_clv_price_pct=float(np.median(price)),
        mean_clv_prob_points=float(np.nanmean(points)) if not np.isnan(points).all() else 0.0,
        beat_close_rate=float(beat.mean()),
        std_clv_price_pct=std,
        ci_low=ci_low, ci_high=ci_high,
        percentiles={
            "p05": float(np.percentile(price, 5)),
            "p25": float(np.percentile(price, 25)),
            "p50": float(np.percentile(price, 50)),
            "p75": float(np.percentile(price, 75)),
            "p95": float(np.percentile(price, 95)),
        },
        significant=significant,
        interpretation=interpretation,
    )
