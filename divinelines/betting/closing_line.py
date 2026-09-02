"""Closing-line policy.

"The last odds row in the database" is not the closing line, and treating it as
one is how a backtest quietly starts using in-play prices. This module is the
single place that answers "what was the close?", and every consumer — CLI,
settlement, API, backtester, frontend — goes through it.

The policy has three parts:

1. **A cutoff.** Only observations at or before ``event_start - cutoff`` count.
   Anything after tip-off or kick-off is in-play information.
2. **A source preference.** Some feeds *declare* a closing price (ESPN's
   ``close`` phase, football-data's closing columns). A declared close beats an
   inferred one, because inferring means "the last thing we happened to see",
   which depends on when our poller last ran rather than on the market.
3. **An aggregation.** Best price, median, or a named book — chosen explicitly
   and recorded on the result, so two analyses can never silently compare a
   best-price close against a consensus close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from ..config import settings
from ..db.connection import query_df
from ..logging_setup import get_logger
from .odds_math import implied_probability, remove_vig

log = get_logger(__name__)

Aggregation = Literal["best", "median", "consensus", "book"]

#: Status vocabulary shared by the CLV ledger and the settlement engine.
STATUS_PENDING = "PENDING"
STATUS_CLOSE_FOUND = "CLOSE_FOUND"
STATUS_NO_CLOSE = "NO_CLOSE_AVAILABLE"
STATUS_SETTLED = "SETTLED"
STATUS_INVALID = "INVALID"


@dataclass(frozen=True)
class ClosingLinePolicy:
    """How a closing price is selected. Recorded on every CLV record."""

    #: Observations later than ``event_start - cutoff_seconds`` are rejected.
    #: Zero means "up to kick-off"; a positive value is a safety margin for
    #: sources whose timestamps are coarse.
    cutoff_seconds: int = 0
    aggregation: Aggregation = "median"
    #: Used when ``aggregation == "book"``.
    bookmaker: str | None = None
    #: Prefer a price the source explicitly labelled as the close.
    prefer_declared_close: bool = True
    #: Require every selection of the market to be quoted, so the margin can be
    #: removed. A one-sided quote cannot produce a fair probability.
    require_complete_market: bool = True
    #: Feeds whose prices are quoted in play and must never be used.
    excluded_sources: tuple[str, ...] = ()
    #: Refuse to resolve a close before the event has started. Without this the
    #: settlement engine happily calls the most recent snapshot of a fixture
    #: three days away "the close", which is not a close — it is the current
    #: price, and comparing our entry against it is comparing a number to
    #: itself.
    require_event_started: bool = True

    @property
    def label(self) -> str:
        parts = [self.aggregation]
        if self.aggregation == "book" and self.bookmaker:
            parts.append(self.bookmaker)
        if self.cutoff_seconds:
            parts.append(f"cutoff{self.cutoff_seconds}s")
        if self.prefer_declared_close:
            parts.append("declared-first")
        return "/".join(parts)


DEFAULT_POLICY = ClosingLinePolicy()


@dataclass
class ClosingLine:
    """The resolved close for one market of one event."""

    game_uid: str
    market: str
    prices: dict[str, float]
    novig_probabilities: dict[str, float]
    implied_probabilities: dict[str, float]
    #: Every complete book's closing prices, so a same-book comparison is
    #: possible. Comparing a best-price entry against a consensus close mixes
    #: model skill with line-shopping skill; keeping the per-book prices lets
    #: the two be separated.
    book_prices: dict[str, dict[str, float]]
    bookmaker: str
    source: str
    timestamp: str | None
    policy: str
    phase: str
    n_bookmakers: int
    status: str = STATUS_CLOSE_FOUND
    reason: str | None = None

    def price_for(self, selection: str) -> float | None:
        return self.prices.get(selection)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_uid": self.game_uid, "market": self.market,
            "prices": {k: round(v, 4) for k, v in self.prices.items()},
            "novig_probabilities": {k: round(v, 5) for k, v in self.novig_probabilities.items()},
            "bookmaker": self.bookmaker, "source": self.source,
            "book_prices": {b: {k: round(v, 4) for k, v in q.items()}
                            for b, q in self.book_prices.items()},
            "timestamp": self.timestamp, "policy": self.policy, "phase": self.phase,
            "n_bookmakers": self.n_bookmakers, "status": self.status, "reason": self.reason,
        }


@dataclass
class NoClosingLine:
    """Why no close could be resolved. Never a silent ``None``."""

    game_uid: str
    market: str
    reason: str
    status: str = STATUS_NO_CLOSE
    candidates_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"game_uid": self.game_uid, "market": self.market, "reason": self.reason,
                "status": self.status, "candidates_seen": self.candidates_seen}


def _event_start(game_uid: str) -> tuple[pd.Timestamp | None, str | None]:
    frame = query_df(
        "SELECT kickoff_utc, game_date, status FROM games WHERE game_uid = ?", (game_uid,)
    )
    if frame.empty:
        return None, None
    row = frame.iloc[0]
    status = str(row["status"])
    for value in (row["kickoff_utc"], row["game_date"]):
        if value in (None, "", "None"):
            continue
        try:
            stamp = pd.to_datetime(value, utc=True, format="mixed")
        except (ValueError, TypeError):
            continue
        if not pd.isna(stamp):
            return stamp, status
    return None, status


def resolve_closing_line(
    game_uid: str,
    market: str,
    selections: Sequence[str],
    *,
    policy: ClosingLinePolicy = DEFAULT_POLICY,
    devig_method: str = "power",
) -> ClosingLine | NoClosingLine:
    """Resolve the closing price for one market, or explain why it cannot be."""
    start, game_status = _event_start(game_uid)

    if policy.require_event_started and start is not None:
        if pd.Timestamp.now(tz="UTC") < start and game_status != "final":
            return NoClosingLine(game_uid, market,
                                 "event has not started; no closing line exists yet",
                                 status=STATUS_PENDING)

    frame = query_df(
        "SELECT bookmaker, selection, price_decimal, captured_at, phase, source, is_closing "
        "FROM odds_snapshots WHERE game_uid = ? AND market = ?",
        (game_uid, market),
    )
    if frame.empty:
        return NoClosingLine(game_uid, market, "no price snapshots recorded")

    frame = frame[frame["selection"].isin(list(selections))]
    if policy.excluded_sources:
        frame = frame[~frame["source"].isin(list(policy.excluded_sources))]
    if frame.empty:
        return NoClosingLine(game_uid, market, "no snapshots for the requested selections")

    candidates = len(frame)
    declared = frame[frame["phase"] == "close"]
    observed = frame[frame["phase"] != "close"].copy()

    if not observed.empty:
        observed["ts"] = pd.to_datetime(observed["captured_at"], utc=True,
                                        format="mixed", errors="coerce")
        observed = observed[observed["ts"].notna()]
        if start is not None:
            # The whole point of the cutoff: anything at or after the event
            # start is in-play information and cannot be part of a close.
            deadline = start - timedelta(seconds=policy.cutoff_seconds)
            observed = observed[observed["ts"] <= deadline]

    pool, phase = None, ""
    if policy.prefer_declared_close and not declared.empty:
        pool, phase = declared, "declared_close"
    elif not observed.empty:
        # Latest observation per bookmaker/selection before the cutoff.
        latest = observed.sort_values("ts").groupby(["bookmaker", "selection"], as_index=False).last()
        pool, phase = latest, "latest_pre_event"
    elif not declared.empty:
        pool, phase = declared, "declared_close"

    if pool is None or pool.empty:
        return NoClosingLine(
            game_uid, market,
            "no snapshot survived the pre-event cutoff" if start is not None
            else "no usable snapshot and no event start recorded",
            candidates_seen=candidates,
        )

    quotes: dict[str, dict[str, float]] = {}
    sources: dict[str, str] = {}
    timestamps: list[str] = []
    for _, row in pool.iterrows():
        book = str(row["bookmaker"])
        quotes.setdefault(book, {})[str(row["selection"])] = float(row["price_decimal"])
        sources[book] = str(row["source"])
        if row.get("captured_at"):
            timestamps.append(str(row["captured_at"]))

    complete = {book: prices for book, prices in quotes.items()
                if all(s in prices for s in selections)}
    if not complete:
        if policy.require_complete_market:
            return NoClosingLine(game_uid, market,
                                 "no bookmaker quoted the complete market at the close",
                                 candidates_seen=candidates)
        complete = quotes

    prices, book_label = _aggregate(complete, selections, policy)
    if prices is None:
        return NoClosingLine(game_uid, market,
                             f"aggregation '{policy.aggregation}' produced no price",
                             candidates_seen=candidates)

    ordered = [prices[s] for s in selections]
    try:
        fair = dict(zip(selections, remove_vig(ordered, devig_method)))
    except (ValueError, ZeroDivisionError) as exc:
        return NoClosingLine(game_uid, market, f"could not de-vig the close: {exc}",
                             candidates_seen=candidates)

    return ClosingLine(
        game_uid=game_uid, market=market, prices=prices, novig_probabilities=fair,
        implied_probabilities={s: implied_probability(prices[s]) for s in selections},
        book_prices={book: dict(quotes) for book, quotes in complete.items()},
        bookmaker=book_label,
        source=sources.get(book_label, sorted(set(sources.values()))[0] if sources else "unknown"),
        timestamp=max(timestamps) if timestamps else None,
        policy=policy.label, phase=phase, n_bookmakers=len(complete),
    )


def _aggregate(quotes: dict[str, dict[str, float]], selections: Sequence[str],
               policy: ClosingLinePolicy) -> tuple[dict[str, float] | None, str]:
    if policy.aggregation == "book":
        if not policy.bookmaker:
            return None, ""
        match = next((book for book in quotes if book.lower() == policy.bookmaker.lower()), None)
        if match is None:
            return None, ""
        return {s: quotes[match][s] for s in selections}, match

    if policy.aggregation == "best":
        prices = {}
        winners = []
        for selection in selections:
            book, price = max(
                ((b, q[selection]) for b, q in quotes.items() if selection in q),
                key=lambda item: item[1], default=(None, None),
            )
            if book is None:
                return None, ""
            prices[selection] = price
            winners.append(book)
        return prices, "best:" + "/".join(sorted(set(winners)))

    # median (default) and consensus both summarise the book set; median is
    # robust to one stale or mispriced feed, which is why it is the default.
    prices = {}
    for selection in selections:
        values = [q[selection] for q in quotes.values() if selection in q]
        if not values:
            return None, ""
        prices[selection] = float(np.median(values))
    label = "consensus_median" if policy.aggregation in ("median", "consensus") else policy.aggregation
    return prices, label


def closing_line_coverage(sport: str | None = None) -> pd.DataFrame:
    """How many finished games actually have a resolvable close, by source.

    Coverage is a data-quality fact the dashboard should show: a CLV average
    computed over 4% of games is not a portfolio-level statement.
    """
    clause = "WHERE g.status = 'final'"
    params: list[Any] = []
    if sport:
        clause += " AND g.sport = ?"
        params.append(sport)
    return query_df(
        f"""
        SELECT g.sport, g.league_id,
               COUNT(DISTINCT g.game_uid) AS final_games,
               COUNT(DISTINCT CASE WHEN o.phase = 'close' THEN g.game_uid END) AS declared_close,
               COUNT(DISTINCT CASE WHEN o.game_uid IS NOT NULL THEN g.game_uid END) AS any_price
        FROM games g
        LEFT JOIN odds_snapshots o ON o.game_uid = g.game_uid
        {clause}
        GROUP BY g.sport, g.league_id
        ORDER BY g.sport, g.league_id
        """,
        params,
    )
