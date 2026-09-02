"""Settlement: turn predictions into CLV records and graded results.

The loop this closes is the point of the whole platform:

    predict -> price -> close -> CLV -> result -> model health

Three properties matter more than anything else here:

* **Idempotent.** Running settlement twice must not double-count profit or
  create a second CLV record. Enforced by a unique key on ``prediction_id``
  plus explicit state transitions, not by hoping the caller behaves.
* **Incremental.** Only work that is actually outstanding is processed —
  predictions with no CLV record, records still waiting for a close, and bets
  waiting for a result. A full reconciliation pass exists but is opt-in.
* **Chronological.** A close is only accepted from before the event started,
  via the closing-line policy. Settlement is the easiest place in the system to
  accidentally introduce future information, so it delegates that decision
  rather than reimplementing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from ..config import settings
from ..db.connection import query_df, write_connection
from ..db.repository import now_iso
from ..logging_setup import get_logger
from .closing_line import (
    STATUS_CLOSE_FOUND,
    STATUS_INVALID,
    STATUS_NO_CLOSE,
    STATUS_PENDING,
    STATUS_SETTLED,
    ClosingLine,
    ClosingLinePolicy,
    DEFAULT_POLICY,
    NoClosingLine,
    resolve_closing_line,
)
from .clv import clv_against_close, summarise_clv
from .odds_math import implied_probability

log = get_logger(__name__)

MARKET_SELECTIONS: dict[str, tuple[str, ...]] = {
    "h2h": ("home", "away"),
    "1x2": ("home", "draw", "away"),
}


@dataclass
class SettlementReport:
    scanned: int = 0
    created: int = 0
    close_found: int = 0
    no_close: int = 0
    awaiting_close: int = 0
    awaiting_result: int = 0
    results_settled: int = 0
    already_settled: int = 0
    invalid: int = 0
    clv_summary: dict[str, Any] = field(default_factory=dict)
    paper_roi: float | None = None
    paper_profit: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned, "created": self.created,
            "close_found": self.close_found, "no_close": self.no_close,
            "awaiting_close": self.awaiting_close, "awaiting_result": self.awaiting_result,
            "results_settled": self.results_settled, "already_settled": self.already_settled,
            "invalid": self.invalid, "clv": self.clv_summary,
            "paper_roi": self.paper_roi, "paper_profit": self.paper_profit,
            "errors": self.errors[:5], "error_count": len(self.errors),
        }


def _selections_for(market: str) -> tuple[str, ...]:
    return MARKET_SELECTIONS.get(market, ("home", "away"))


def _executable_predictions(sport: str | None, day: str | None,
                            include_settled: bool) -> pd.DataFrame:
    """Predictions that could be priced: a real price and a real market.

    A prediction with no price was never executable, so it gets no CLV record —
    counting it would quietly dilute every cohort average with rows that never
    had an entry price at all.
    """
    clauses = ["p.price_decimal IS NOT NULL", "p.price_decimal > 1.0"]
    params: list[Any] = []
    if sport:
        clauses.append("p.sport = ?")
        params.append(sport)
    if day:
        clauses.append("date(g.game_date) = date(?)")
        params.append(day)
    if not include_settled:
        clauses.append(
            "(c.clv_id IS NULL OR c.status IN ('PENDING', 'NO_CLOSE_AVAILABLE') "
            "OR (c.result IS NULL AND g.status = 'final'))"
        )

    return query_df(
        f"""
        SELECT p.prediction_id, p.created_at, p.sport, p.league_id, p.game_uid, p.market,
               p.selection, p.model_prob, p.market_prob, p.price_decimal, p.bookmaker,
               p.edge, p.stake, p.data_quality, p.model_id, p.model_version, p.data_version,
               p.prediction_stage, p.lineup_state, p.seconds_to_event, p.superseded_at,
               g.status AS game_status, g.home_score, g.away_score, g.kickoff_utc, g.game_date,
               c.clv_id, c.status AS clv_status, c.result AS clv_result
        FROM predictions p
        JOIN games g ON g.game_uid = p.game_uid
        LEFT JOIN clv_records c ON c.prediction_id = p.prediction_id
        WHERE {' AND '.join(clauses)}
        ORDER BY p.created_at
        """,
        params,
    )


def _grade(market: str, selection: str, home_score: float, away_score: float) -> str | None:
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if market in ("h2h", "1x2", "moneyline"):
        if home_score > away_score:
            winner = "home"
        elif home_score < away_score:
            winner = "away"
        else:
            winner = "draw"
        if selection == "draw":
            return "won" if winner == "draw" else "lost"
        if winner == "draw":
            # A two-way market cannot be drawn; a 1X2 home/away bet loses.
            return "push" if market in ("h2h", "moneyline") else "lost"
        return "won" if winner == selection else "lost"
    if market == "totals":
        try:
            side, line = selection.split("_", 1)
            line_value = float(line)
        except ValueError:
            return None
        total = home_score + away_score
        if total == line_value:
            return "push"
        if side == "over":
            return "won" if total > line_value else "lost"
        if side == "under":
            return "won" if total < line_value else "lost"
    return None


def settle(
    *,
    sport: str | None = None,
    day: str | None = None,
    policy: ClosingLinePolicy = DEFAULT_POLICY,
    dry_run: bool = False,
    full_reconcile: bool = False,
) -> SettlementReport:
    """Create/advance CLV records and grade finished events."""
    report = SettlementReport()
    frame = _executable_predictions(sport, day, include_settled=full_reconcile)
    report.scanned = len(frame)
    if frame.empty:
        report.clv_summary = summarise_clv([]).to_dict()
        return report

    now = now_iso()
    inserts: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    clv_values: list[float] = []

    # One close per (game, market): resolving it per prediction would repeat the
    # same query for every selection on the same match.
    close_cache: dict[tuple[str, str], ClosingLine | NoClosingLine] = {}

    for _, row in frame.iterrows():
        market = str(row["market"])
        key = (str(row["game_uid"]), market)
        if key not in close_cache:
            close_cache[key] = resolve_closing_line(
                str(row["game_uid"]), market, _selections_for(market), policy=policy
            )
        close = close_cache[key]

        game_final = str(row["game_status"]) == "final"
        result = (_grade(market, str(row["selection"]), row["home_score"], row["away_score"])
                  if game_final else None)

        record: dict[str, Any] = {
            "prediction_id": int(row["prediction_id"]),
            "game_uid": str(row["game_uid"]),
            "sport": str(row["sport"]),
            "league_id": row["league_id"],
            "market": market,
            "selection": str(row["selection"]),
            "entry_timestamp": str(row["created_at"]),
            "entry_odds": float(row["price_decimal"]),
            "entry_book": row["bookmaker"],
            "entry_market_prob": _float_or_none(row["market_prob"]),
            "entry_implied_prob": implied_probability(float(row["price_decimal"])),
            "model_probability": float(row["model_prob"]),
            "model_version": row["model_version"],
            "model_id": row["model_id"],
            "data_version": row["data_version"],
            "prediction_stage": row["prediction_stage"],
            "lineup_state": row["lineup_state"],
            "seconds_to_event": _int_or_none(row["seconds_to_event"]),
            "data_quality": _float_or_none(row["data_quality"]),
            "edge": _float_or_none(row["edge"]),
            "stake": _float_or_none(row["stake"]),
            "result": result,
            "updated_at": now,
        }

        if isinstance(close, ClosingLine):
            try:
                clv = clv_against_close(float(row["price_decimal"]), str(row["selection"]), close)
            except ValueError as exc:
                report.errors.append(f"prediction {row['prediction_id']}: {exc}")
                record.update({"status": STATUS_INVALID, "closing_policy": close.policy})
                report.invalid += 1
                _queue(record, row, inserts, updates)
                continue

            # Same-book CLV where the entry book also has a close. This is the
            # comparison that isolates the model: both sides of it come from the
            # same bookmaker, so line-shopping cannot flatter it.
            same_book_price = _same_book_close(close, row["bookmaker"], str(row["selection"]))
            same_book_clv = (
                (float(row["price_decimal"]) / same_book_price - 1.0) * 100.0
                if same_book_price else None
            )

            record.update({
                "closing_same_book_odds": same_book_price,
                "clv_same_book_pct": same_book_clv,
                "clv_basis": "consensus_close" if same_book_price is None else "both",
                "closing_timestamp": close.timestamp,
                "closing_odds": clv.closing_price,
                "closing_book": close.bookmaker,
                "closing_implied_prob": implied_probability(clv.closing_price),
                "closing_novig_prob": clv.closing_fair_probability,
                "closing_source": close.source,
                "closing_policy": close.policy,
                "clv_price_pct": clv.clv_price_pct,
                "clv_prob_points": clv.clv_prob_points,
                "clv_log_odds": clv.clv_log_odds,
                "beat_close": int(clv.beat_close),
                "status": STATUS_SETTLED if result else STATUS_CLOSE_FOUND,
            })
            clv_values.append(clv.clv_price_pct)
            report.close_found += 1
        else:
            # Distinguish "the event has not closed yet" from "this event closed
            # and we never captured a price" — only the second is a data gap.
            pending = not game_final
            record.update({
                "status": STATUS_PENDING if pending else STATUS_NO_CLOSE,
                "closing_policy": policy.label,
            })
            if pending:
                report.awaiting_close += 1
            else:
                report.no_close += 1

        if result:
            stake = _float_or_none(row["stake"]) or 0.0
            price = float(row["price_decimal"])
            record["profit"] = (
                stake * (price - 1.0) if result == "won"
                else (0.0 if result == "push" else -stake)
            )
            record["settled_at"] = now
            report.results_settled += 1
        elif game_final:
            report.errors.append(
                f"prediction {row['prediction_id']}: finished game could not be graded "
                f"for market '{market}' selection '{row['selection']}'"
            )
        else:
            report.awaiting_result += 1

        _queue(record, row, inserts, updates)

    if not dry_run:
        report.created = _persist(inserts, updates, now)
    else:
        report.created = len(inserts)

    settled = query_df(
        "SELECT clv_price_pct, clv_same_book_pct, stake, profit FROM clv_records "
        "WHERE clv_price_pct IS NOT NULL" + (" AND sport = ?" if sport else ""),
        [sport] if sport else [],
    )
    if not settled.empty:
        report.clv_summary = summarise_clv(settled["clv_price_pct"].tolist()).to_dict()
        report.clv_summary["basis"] = "entry (best price) vs consensus close"
        same_book = settled["clv_same_book_pct"].dropna().tolist()
        if same_book:
            report.clv_summary["same_book"] = summarise_clv(same_book).to_dict()
            report.clv_summary["same_book"]["basis"] = "entry vs the same book's close"
        staked = float(settled["stake"].fillna(0).sum())
        profit = float(settled["profit"].fillna(0).sum())
        if staked > 0:
            report.paper_profit = round(profit, 2)
            report.paper_roi = round(profit / staked, 4)
    else:
        report.clv_summary = summarise_clv(clv_values).to_dict()

    log.info("settlement complete", extra=report.to_dict())
    return report


def _same_book_close(close: ClosingLine, entry_book: Any, selection: str) -> float | None:
    """The entry bookmaker's own closing price, if that book quoted the close."""
    if not entry_book or pd.isna(entry_book):
        return None
    wanted = str(entry_book).strip().lower()
    for book, prices in close.book_prices.items():
        if book.strip().lower() == wanted and selection in prices:
            return float(prices[selection])
    return None


def _queue(record: dict[str, Any], row: pd.Series,
           inserts: list[dict[str, Any]], updates: list[dict[str, Any]]) -> None:
    if pd.isna(row.get("clv_id")):
        inserts.append(record)
    else:
        record["clv_id"] = int(row["clv_id"])
        updates.append(record)


def _persist(inserts: list[dict[str, Any]], updates: list[dict[str, Any]], now: str) -> int:
    created = 0
    with write_connection() as conn:
        for record in inserts:
            record = {**record, "created_at": now}
            columns = list(record.keys())
            # ON CONFLICT rather than a plain INSERT: two settlement runs racing
            # on the same prediction must not create two CLV rows.
            assignments = ", ".join(f"{c}=excluded.{c}" for c in columns
                                    if c not in ("prediction_id", "created_at"))
            conn.execute(
                f"INSERT INTO clv_records ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) "
                f"ON CONFLICT(prediction_id) DO UPDATE SET {assignments}",
                tuple(record[c] for c in columns),
            )
            created += 1

        for record in updates:
            clv_id = record.pop("clv_id")
            columns = [c for c in record if c != "prediction_id"]
            conn.execute(
                f"UPDATE clv_records SET {', '.join(f'{c}=?' for c in columns)} WHERE clv_id = ?",
                tuple(record[c] for c in columns) + (clv_id,),
            )

        # Keep the prediction ledger's own state in step so `scan` can tell at a
        # glance what still needs work.
        conn.execute(
            """
            UPDATE predictions SET settlement_state = COALESCE((
                SELECT status FROM clv_records c WHERE c.prediction_id = predictions.prediction_id
            ), settlement_state)
            """
        )
    return created


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    result = _float_or_none(value)
    return None if result is None else int(result)


def settlement_state() -> dict[str, Any]:
    """Counts by status, for `status` and the system page."""
    frame = query_df("SELECT status, COUNT(*) AS n FROM clv_records GROUP BY status")
    counts = {str(row["status"]): int(row["n"]) for _, row in frame.iterrows()}
    pending = query_df(
        "SELECT COUNT(*) AS n FROM predictions p LEFT JOIN clv_records c "
        "ON c.prediction_id = p.prediction_id WHERE c.clv_id IS NULL "
        "AND p.price_decimal IS NOT NULL"
    )
    counts["UNPROCESSED"] = int(pending["n"].iloc[0]) if not pending.empty else 0
    return counts
