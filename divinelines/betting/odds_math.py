"""Odds conversion and margin removal.

The platform stores decimal odds internally and converts only at the display
boundary — mixing American and decimal odds inside calculations is a classic
source of sign errors.

Removing the bookmaker margin correctly matters just as much: comparing a
model probability against a *raw* implied probability overstates the market's
view by the whole overround, which manufactures edges that do not exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------- format

def american_to_decimal(american: float) -> float:
    """+150 -> 2.5, -200 -> 1.5."""
    american = float(american)
    if american == 0:
        raise ValueError("American odds cannot be zero")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal_odds: float) -> float:
    """2.5 -> +150, 1.5 -> -200."""
    decimal_odds = float(decimal_odds)
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


def implied_probability(decimal_odds: float) -> float:
    """Gross (vig-inclusive) implied probability of a decimal price."""
    decimal_odds = float(decimal_odds)
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def probability_to_decimal(probability: float) -> float:
    """Fair decimal price for a probability."""
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1")
    return 1.0 / probability


def overround(decimal_prices: Iterable[float]) -> float:
    """Sum of implied probabilities minus 1 (the bookmaker's margin)."""
    return sum(implied_probability(p) for p in decimal_prices) - 1.0


# ------------------------------------------------------------- margin removal

def remove_vig_multiplicative(decimal_prices: Sequence[float]) -> list[float]:
    """Normalise implied probabilities so they sum to 1.

    Simple and unbiased across selections, but it distributes the margin
    proportionally, which slightly understates favourites.
    """
    raw = np.array([implied_probability(p) for p in decimal_prices], dtype=float)
    total = raw.sum()
    if total <= 0:
        raise ValueError("Cannot de-vig non-positive probabilities")
    return (raw / total).tolist()


def remove_vig_power(decimal_prices: Sequence[float], *, tolerance: float = 1e-10,
                     max_iterations: int = 200) -> list[float]:
    """Power method: solve for ``k`` such that ``sum(p_i ** k) == 1``.

    Handles the favourite-longshot bias better than proportional scaling
    because it shrinks longshot prices more than short ones.
    """
    raw = np.array([implied_probability(p) for p in decimal_prices], dtype=float)
    if raw.sum() <= 1.0 + 1e-12:
        return remove_vig_multiplicative(decimal_prices)

    low, high = 0.5, 3.0
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        total = float(np.sum(raw ** mid))
        if abs(total - 1.0) < tolerance:
            break
        if total > 1.0:
            low = mid
        else:
            high = mid
    fair = raw ** mid
    return (fair / fair.sum()).tolist()


def remove_vig(decimal_prices: Sequence[float], method: str = "power") -> list[float]:
    if len(decimal_prices) < 2:
        raise ValueError("De-vigging needs the full set of prices for a market")
    if method == "multiplicative":
        return remove_vig_multiplicative(decimal_prices)
    if method == "power":
        return remove_vig_power(decimal_prices)
    raise ValueError(f"Unknown de-vig method '{method}'")


# ----------------------------------------------------------------- consensus

@dataclass
class MarketConsensus:
    """What the market as a whole thinks, plus where the best price is."""

    fair_probabilities: dict[str, float]
    best_price: dict[str, float]
    best_bookmaker: dict[str, str]
    median_price: dict[str, float]
    overround: float
    n_bookmakers: int

    def to_dict(self) -> dict[str, object]:
        return {
            "fair_probabilities": {k: round(v, 5) for k, v in self.fair_probabilities.items()},
            "best_price": {k: round(v, 3) for k, v in self.best_price.items()},
            "best_bookmaker": self.best_bookmaker,
            "median_price": {k: round(v, 3) for k, v in self.median_price.items()},
            "overround": round(self.overround, 4),
            "n_bookmakers": self.n_bookmakers,
        }


def build_consensus(
    quotes: Mapping[str, Mapping[str, float]],
    *,
    method: str = "power",
) -> MarketConsensus:
    """Aggregate ``{bookmaker: {selection: decimal_price}}`` into one view.

    Each bookmaker is de-vigged **individually** before aggregation, because
    averaging prices across books with different margins mixes margin into the
    probability estimate.  The consensus is then the median of those fair
    probabilities, which is robust to one book being stale or mispriced.
    """
    if not quotes:
        raise ValueError("No quotes supplied")

    selections: list[str] = []
    for book_quotes in quotes.values():
        for selection in book_quotes:
            if selection not in selections:
                selections.append(selection)

    fair_by_selection: dict[str, list[float]] = {s: [] for s in selections}
    overrounds: list[float] = []
    complete_books = 0

    for book_quotes in quotes.values():
        prices = [book_quotes.get(s) for s in selections]
        if any(p is None or p <= 1.0 for p in prices):
            continue  # a partial book cannot be de-vigged safely
        complete_books += 1
        overrounds.append(overround(prices))
        for selection, fair in zip(selections, remove_vig(prices, method)):
            fair_by_selection[selection].append(fair)

    if complete_books == 0:
        raise ValueError("No bookmaker quoted the complete market; cannot de-vig")

    medians = {s: float(np.median(v)) for s, v in fair_by_selection.items()}
    total = sum(medians.values())
    fair_probabilities = {s: v / total for s, v in medians.items()}

    best_price: dict[str, float] = {}
    best_bookmaker: dict[str, str] = {}
    median_price: dict[str, float] = {}
    for selection in selections:
        prices = {b: q[selection] for b, q in quotes.items()
                  if q.get(selection) and q[selection] > 1.0}
        if not prices:
            continue
        book, price = max(prices.items(), key=lambda kv: kv[1])
        best_price[selection] = price
        best_bookmaker[selection] = book
        median_price[selection] = float(np.median(list(prices.values())))

    return MarketConsensus(
        fair_probabilities=fair_probabilities,
        best_price=best_price,
        best_bookmaker=best_bookmaker,
        median_price=median_price,
        overround=float(np.mean(overrounds)) if overrounds else 0.0,
        n_bookmakers=complete_books,
    )


def log_odds(probability: float) -> float:
    probability = min(max(float(probability), 1e-9), 1 - 1e-9)
    return math.log(probability / (1.0 - probability))
