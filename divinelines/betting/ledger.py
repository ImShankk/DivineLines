"""Prediction history, paper trading and settlement.

Every prediction the platform makes is persisted with the model version, the
data version, the price available at the time and the features that produced
it.  Once the game finishes, results are graded automatically and closing-line
value is attached.  That record is the foundation of every performance and
calibration analysis in the platform — without it, model evaluation is guesswork.

Paper mode is the default: bets are recorded and graded exactly as live bets
would be, but nothing is ever executed anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from ..config import settings
from ..db.connection import query_df, write_connection
from ..db.repository import now_iso
from ..logging_setup import get_logger
from .clv import closing_line_value

log = get_logger(__name__)


@dataclass
class PredictionRecord:
    sport: str
    game_uid: str
    market: str
    selection: str
    model_probability: float
    league_id: str | None = None
    market_probability: float | None = None
    price_decimal: float | None = None
    bookmaker: str | None = None
    edge: float | None = None
    ev_per_unit: float | None = None
    kelly_fraction: float | None = None
    stake: float | None = None
    confidence: float | None = None
    edge_score: float | None = None
    data_quality: float | None = None
    model_id: str | None = None
    model_version: str | None = None
    data_version: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    created_at: str | None = None
    mode: str | None = None
    # --- V3: what was knowable when this prediction was made -------------
    prediction_stage: str = "scheduled"
    lineup_state: str = "unknown"
    feature_version: str | None = None
    event_start_utc: str | None = None
    seconds_to_event: int | None = None
    information_snapshot: dict[str, Any] = field(default_factory=dict)
    supersedes_id: int | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at or now_iso(),
            "sport": self.sport,
            "league_id": self.league_id,
            "game_uid": self.game_uid,
            "market": self.market,
            "selection": self.selection,
            "model_prob": float(self.model_probability),
            "market_prob": self.market_probability,
            "price_decimal": self.price_decimal,
            "bookmaker": self.bookmaker,
            "edge": self.edge,
            "ev_per_unit": self.ev_per_unit,
            "kelly_fraction": self.kelly_fraction,
            "stake": self.stake,
            "confidence": self.confidence,
            "edge_score": self.edge_score,
            "data_quality": self.data_quality,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "data_version": self.data_version,
            "features": json.dumps(self.features, default=str),
            "explanation": json.dumps(self.explanation, default=str),
            "flags": json.dumps(self.flags),
            "mode": self.mode or settings.mode,
            "prediction_stage": self.prediction_stage,
            "lineup_state": self.lineup_state,
            "feature_version": self.feature_version,
            "event_start_utc": self.event_start_utc,
            "seconds_to_event": self.seconds_to_event,
            "information_snapshot": json.dumps(self.information_snapshot, default=str),
            "supersedes_id": self.supersedes_id,
        }


def supersede_predictions(game_uid: str, market: str, *, before: str,
                          mode: str | None = None) -> int:
    """Mark earlier predictions for a fixture as superseded.

    A new prediction never overwrites an old one — the pre-lineup view and the
    post-lineup view are both evidence, and comparing them is how lineup impact
    gets measured at all. Superseded rows stay queryable but are excluded from
    scoring so one match cannot be counted twice.
    """
    with write_connection() as conn:
        cursor = conn.execute(
            "UPDATE predictions SET superseded_at = ? WHERE game_uid = ? AND market = ? "
            "AND created_at < ? AND superseded_at IS NULL"
            + (" AND mode = ?" if mode else ""),
            (before, game_uid, market, before) + ((mode,) if mode else ()),
        )
        return cursor.rowcount


def record_predictions(records: Sequence[PredictionRecord]) -> list[int]:
    """Persist predictions, returning their ids."""
    if not records:
        return []
    rows = [r.to_row() for r in records]
    columns = list(rows[0].keys())
    sql = (
        f"INSERT OR IGNORE INTO predictions ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})"
    )
    ids: list[int] = []
    with write_connection() as conn:
        for row in rows:
            cursor = conn.execute(sql, tuple(row[c] for c in columns))
            if cursor.lastrowid:
                ids.append(int(cursor.lastrowid))
    log.info("recorded predictions", extra={"count": len(ids)})
    return ids


def place_paper_bets(prediction_ids: Iterable[int], *, mode: str = "paper") -> int:
    """Open paper bets from stored predictions that carry a stake."""
    ids = list(prediction_ids)
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    predictions = query_df(
        f"SELECT * FROM predictions WHERE prediction_id IN ({placeholders})", ids
    )
    predictions = predictions[
        predictions["stake"].notna() & (predictions["stake"] > 0)
        & predictions["price_decimal"].notna()
    ]
    if predictions.empty:
        return 0

    rows = [
        (
            int(row["prediction_id"]), now_iso(), row["sport"], row["game_uid"],
            row["market"], row["selection"], float(row["price_decimal"]),
            row["bookmaker"], float(row["stake"]), float(row["model_prob"]),
            row["market_prob"], "open", mode,
        )
        for _, row in predictions.iterrows()
    ]
    with write_connection() as conn:
        conn.executemany(
            "INSERT INTO bets (prediction_id, placed_at, sport, game_uid, market, selection,"
            " price_decimal, bookmaker, stake, model_prob, market_prob, status, mode)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    log.info("paper bets opened", extra={"count": len(rows), "mode": mode})
    return len(rows)


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------

def _outcome_for_selection(row: pd.Series) -> str | None:
    """Grade a selection against a finished game."""
    home, away = row["home_score"], row["away_score"]
    if pd.isna(home) or pd.isna(away):
        return None
    market, selection = row["market"], row["selection"]

    if market in ("h2h", "1x2", "moneyline"):
        if selection == "home":
            winner = "home" if home > away else ("draw" if home == away else "away")
        elif selection == "away":
            winner = "home" if home > away else ("draw" if home == away else "away")
        elif selection == "draw":
            winner = "draw" if home == away else "other"
        else:
            return None
        if selection == "draw":
            return "won" if winner == "draw" else "lost"
        # In two-way markets (NBA) a draw is impossible, so this also grades h2h.
        if winner == "draw":
            return "push"
        return "won" if winner == selection else "lost"

    if market == "totals":
        try:
            side, line = selection.split("_", 1)
            line_value = float(line)
        except ValueError:
            return None
        total = home + away
        if total == line_value:
            return "push"
        if side == "over":
            return "won" if total > line_value else "lost"
        if side == "under":
            return "won" if total < line_value else "lost"
    return None


def _closing_prices(game_uid: str, market: str) -> dict[str, float]:
    """Closing prices for a market: flagged closers, else the last snapshot."""
    closing = query_df(
        "SELECT selection, AVG(price_decimal) AS price FROM odds_snapshots "
        "WHERE game_uid = ? AND market = ? AND is_closing = 1 GROUP BY selection",
        (game_uid, market),
    )
    if closing.empty:
        closing = query_df(
            """
            SELECT o.selection, AVG(o.price_decimal) AS price
            FROM odds_snapshots o
            JOIN (SELECT selection, MAX(captured_at) mx FROM odds_snapshots
                  WHERE game_uid = ? AND market = ? GROUP BY selection) last
              ON last.selection = o.selection AND last.mx = o.captured_at
            WHERE o.game_uid = ? AND o.market = ?
            GROUP BY o.selection
            """,
            (game_uid, market, game_uid, market),
        )
    if closing.empty:
        return {}
    return {str(r["selection"]): float(r["price"]) for _, r in closing.iterrows()}


def settle_open_bets(*, as_of: str | None = None) -> dict[str, Any]:
    """Grade every open bet whose game has finished; attach profit and CLV."""
    open_bets = query_df(
        """
        SELECT b.*, g.home_score, g.away_score, g.status, g.game_date
        FROM bets b JOIN games g ON g.game_uid = b.game_uid
        WHERE b.status = 'open' AND g.status = 'final'
        """
    )
    if open_bets.empty:
        return {"settled": 0, "with_clv": 0}

    settled = 0
    with_clv = 0
    updates: list[tuple] = []

    for _, row in open_bets.iterrows():
        outcome = _outcome_for_selection(row)
        if outcome is None:
            continue
        stake = float(row["stake"])
        price = float(row["price_decimal"])
        if outcome == "won":
            payout, profit = stake * price, stake * (price - 1.0)
        elif outcome == "push":
            payout, profit = stake, 0.0
        else:
            payout, profit = 0.0, -stake

        closing_price = closing_fair = clv_value = None
        closing = _closing_prices(str(row["game_uid"]), str(row["market"]))
        if closing and str(row["selection"]) in closing and len(closing) >= 2:
            try:
                clv = closing_line_value(price, closing, str(row["selection"]))
                closing_price = clv.closing_price
                closing_fair = clv.closing_fair_probability
                clv_value = clv.clv_price_pct
                with_clv += 1
            except (ValueError, ZeroDivisionError) as exc:
                log.warning("CLV computation failed",
                            extra={"game": row["game_uid"], "error": str(exc)})

        updates.append(
            (outcome, as_of or now_iso(), payout, profit, closing_price,
             closing_fair, clv_value, int(row["bet_id"]))
        )
        settled += 1

    if updates:
        with write_connection() as conn:
            conn.executemany(
                "UPDATE bets SET status=?, settled_at=?, payout=?, profit=?,"
                " closing_price=?, closing_prob=?, clv=? WHERE bet_id=?",
                updates,
            )
    log.info("settled bets", extra={"settled": settled, "with_clv": with_clv})
    return {"settled": settled, "with_clv": with_clv}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def bankroll_curve(mode: str | None = None) -> pd.DataFrame:
    """Cumulative profit over settled bets, in settlement order."""
    clause = "AND mode = ?" if mode else ""
    params = [mode] if mode else []
    df = query_df(
        f"SELECT bet_id, settled_at, sport, stake, profit FROM bets "
        f"WHERE status IN ('won','lost','push') {clause} ORDER BY settled_at, bet_id",
        params,
    )
    if df.empty:
        return df
    df["cumulative_profit"] = df["profit"].cumsum()
    df["bankroll"] = settings.betting.bankroll + df["cumulative_profit"]
    peak = df["bankroll"].cummax()
    df["drawdown"] = df["bankroll"] - peak
    return df


def performance_summary(group_by: str | None = None, mode: str | None = None) -> pd.DataFrame:
    """ROI/hit-rate/CLV, optionally bucketed by sport, market or edge band."""
    clause = "AND b.mode = ?" if mode else ""
    params = [mode] if mode else []
    df = query_df(
        f"""
        SELECT b.*, p.edge, p.model_version, g.league_id
        FROM bets b
        LEFT JOIN predictions p ON p.prediction_id = b.prediction_id
        LEFT JOIN games g ON g.game_uid = b.game_uid
        WHERE b.status IN ('won','lost','push') {clause}
        """,
        params,
    )
    if df.empty:
        return df

    df["won_flag"] = (df["status"] == "won").astype(int)
    if group_by == "edge_bucket":
        df[group_by] = pd.cut(
            df["edge"].fillna(0), [-1, 0.02, 0.04, 0.06, 0.10, 1.0],
            labels=["<2%", "2-4%", "4-6%", "6-10%", "10%+"],
        )
    elif group_by == "odds_bucket":
        df[group_by] = pd.cut(
            df["price_decimal"], [1.0, 1.5, 2.0, 3.0, 5.0, 100.0],
            labels=["heavy fav", "fav", "even/dog", "dog", "longshot"],
        )

    keys = [group_by] if group_by else []
    grouped = df.groupby(keys, observed=True) if keys else [("all", df)]
    rows: list[dict[str, Any]] = []
    for name, group in (grouped if keys else grouped):
        if keys:
            label = name if isinstance(name, str) else str(name)
        else:
            label = "all"
        staked = float(group["stake"].sum())
        profit = float(group["profit"].sum())
        rows.append(
            {
                "group": label,
                "bets": int(len(group)),
                "staked": round(staked, 2),
                "profit": round(profit, 2),
                "roi": round(profit / staked, 4) if staked else 0.0,
                "hit_rate": round(float(group["won_flag"].mean()), 4),
                "avg_price": round(float(group["price_decimal"].mean()), 3),
                "avg_clv_pct": round(float(group["clv"].dropna().mean()), 3)
                if group["clv"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)
