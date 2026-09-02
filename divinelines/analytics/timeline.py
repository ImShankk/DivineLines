"""Event timelines and prediction-change attribution.

For one fixture, merge everything the platform observed into a single ordered
story:

    14:00  model      Home 52%
    16:00  injury     Starting striker out
    16:05  model      Home 49%   (-3.0pp)
    18:32  lineup     Confirmed XI
    18:35  model      Home 44%   (-5.0pp)
    18:45  market     Home 43%   (market followed)

Attribution is deliberately conservative. The platform can say *what changed*
between two predictions and *what information arrived in between*; it cannot
prove the information caused the change. So contributions are labelled as
co-occurring evidence, and a movement with no matching information event is
reported as unexplained rather than assigned to whatever happened to be nearby.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..betting.odds_math import remove_vig
from ..db.connection import query_df
from ..logging_setup import get_logger

log = get_logger(__name__)

#: Probability movement below this is noise from refitting, not news.
MATERIAL_MOVE = 0.01
#: Information arriving within this window before a prediction is treated as
#: potentially explaining it.
ATTRIBUTION_WINDOW = timedelta(hours=3)


@dataclass
class TimelineEntry:
    timestamp: str
    kind: str            # prediction | market | lineup | information
    label: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "kind": self.kind,
                "label": self.label, "detail": self.detail}


def _prediction_entries(game_uid: str, market: str | None) -> list[TimelineEntry]:
    clause = "AND market = ?" if market else ""
    params: list[Any] = [game_uid] + ([market] if market else [])
    frame = query_df(
        f"""
        SELECT prediction_id, created_at, market, selection, model_prob, market_prob,
               price_decimal, bookmaker, edge, model_version, prediction_stage,
               lineup_state, superseded_at, mode, stake
        FROM predictions WHERE game_uid = ? {clause}
        ORDER BY created_at, prediction_id
        """,
        params,
    )
    entries: list[TimelineEntry] = []
    if frame.empty:
        return entries

    for (market_name, selection), group in frame.groupby(["market", "selection"]):
        previous: float | None = None
        for _, row in group.iterrows():
            probability = float(row["model_prob"])
            move = None if previous is None else probability - previous
            entries.append(TimelineEntry(
                timestamp=str(row["created_at"]), kind="prediction",
                label=f"{market_name}/{selection} {probability:.1%}",
                detail={
                    "prediction_id": int(row["prediction_id"]),
                    "market": market_name, "selection": selection,
                    "model_probability": round(probability, 5),
                    "market_probability": (None if pd.isna(row["market_prob"])
                                           else round(float(row["market_prob"]), 5)),
                    "move": None if move is None else round(move, 5),
                    "material": bool(move is not None and abs(move) >= MATERIAL_MOVE),
                    "stage": row["prediction_stage"], "lineup_state": row["lineup_state"],
                    "model_version": row["model_version"], "mode": row["mode"],
                    "price": (None if pd.isna(row["price_decimal"])
                              else float(row["price_decimal"])),
                    "superseded": bool(row["superseded_at"]),
                },
            ))
            previous = probability
    return entries


def _market_entries(game_uid: str, market: str | None) -> list[TimelineEntry]:
    clause = "AND market = ?" if market else ""
    params: list[Any] = [game_uid] + ([market] if market else [])
    frame = query_df(
        f"""
        SELECT captured_at, market, selection, bookmaker, price_decimal, phase, source
        FROM odds_snapshots WHERE game_uid = ? {clause}
        ORDER BY captured_at
        """,
        params,
    )
    if frame.empty:
        return []

    entries: list[TimelineEntry] = []
    for (captured, phase), group in frame.groupby(["captured_at", "phase"]):
        prices = (group.groupby("selection")["price_decimal"].median().to_dict())
        fair = None
        if len(prices) >= 2:
            selections = list(prices)
            try:
                fair = dict(zip(selections, remove_vig([prices[s] for s in selections])))
            except (ValueError, ZeroDivisionError):
                fair = None
        entries.append(TimelineEntry(
            timestamp=str(captured), kind="market",
            label=f"{phase}: " + ", ".join(f"{s} {p:.2f}" for s, p in sorted(prices.items())),
            detail={
                "phase": phase, "prices": {k: round(v, 3) for k, v in prices.items()},
                "novig": None if fair is None else {k: round(v, 5) for k, v in fair.items()},
                "books": int(group["bookmaker"].nunique()),
                "sources": sorted(group["source"].unique().tolist()),
            },
        ))
    return entries


def _lineup_entries(game_uid: str) -> list[TimelineEntry]:
    frame = query_df(
        """
        SELECT observed_at, team_uid, lineup_state, formation,
               SUM(CASE WHEN status = 'starter' THEN 1 ELSE 0 END) AS starters
        FROM lineup_observations WHERE game_uid = ?
        GROUP BY observed_at, team_uid, lineup_state, formation
        ORDER BY observed_at
        """,
        (game_uid,),
    )
    return [
        TimelineEntry(
            timestamp=str(row["observed_at"]), kind="lineup",
            label=f"{row['lineup_state']} XI ({row['formation'] or 'formation unknown'})",
            detail={"team_uid": row["team_uid"], "state": row["lineup_state"],
                    "formation": row["formation"], "starters": int(row["starters"])},
        )
        for _, row in frame.iterrows()
    ]


def _information_entries(game_uid: str) -> list[TimelineEntry]:
    frame = query_df(
        "SELECT observed_at, kind, detail, team_uid, magnitude FROM information_events "
        "WHERE game_uid = ? ORDER BY observed_at",
        (game_uid,),
    )
    return [
        TimelineEntry(
            timestamp=str(row["observed_at"]), kind="information",
            label=f"{row['kind']}: {row['detail'] or ''}".strip(),
            detail={"kind": row["kind"], "team_uid": row["team_uid"],
                    "magnitude": row["magnitude"]},
        )
        for _, row in frame.iterrows()
    ]


def event_timeline(game_uid: str, market: str | None = None) -> dict[str, Any]:
    """Everything observed for one fixture, in order, plus attributions."""
    game = query_df(
        """
        SELECT g.*, th.canonical_name AS home_name, ta.canonical_name AS away_name
        FROM games g
        JOIN teams th ON th.team_uid = g.home_team_uid
        JOIN teams ta ON ta.team_uid = g.away_team_uid
        WHERE g.game_uid = ?
        """,
        (game_uid,),
    )
    if game.empty:
        return {"found": False, "game_uid": game_uid}

    entries = (_prediction_entries(game_uid, market) + _market_entries(game_uid, market)
               + _lineup_entries(game_uid) + _information_entries(game_uid))
    entries.sort(key=lambda entry: entry.timestamp or "")

    clv = query_df(
        "SELECT selection, entry_odds, entry_book, closing_odds, closing_book, "
        "clv_price_pct, clv_same_book_pct, status, result, profit "
        "FROM clv_records WHERE game_uid = ?",
        (game_uid,),
    )

    return {
        "found": True,
        "game": {k: (None if pd.isna(v) else v) for k, v in game.iloc[0].to_dict().items()},
        "timeline": [entry.to_dict() for entry in entries],
        "attributions": attribute_movements(entries),
        "market_vs_model": market_vs_model(entries),
        "clv": clv.to_dict("records") if not clv.empty else [],
    }


def attribute_movements(entries: Sequence[TimelineEntry]) -> list[dict[str, Any]]:
    """Pair material prediction movements with information that preceded them.

    Co-occurrence, not causation — the wording of the output says so, because a
    confident-sounding causal claim built on a three-hour window would be the
    kind of narrative this platform is supposed to avoid.
    """
    information = [e for e in entries if e.kind in ("information", "lineup")]
    attributions: list[dict[str, Any]] = []

    for entry in entries:
        if entry.kind != "prediction" or not entry.detail.get("material"):
            continue
        moment = pd.to_datetime(entry.timestamp, utc=True, errors="coerce")
        if pd.isna(moment):
            continue

        preceding = []
        for candidate in information:
            when = pd.to_datetime(candidate.timestamp, utc=True, errors="coerce")
            if pd.isna(when) or when > moment:
                continue
            if moment - when <= ATTRIBUTION_WINDOW:
                preceding.append({"timestamp": candidate.timestamp, "kind": candidate.kind,
                                  "label": candidate.label})

        attributions.append({
            "timestamp": entry.timestamp,
            "market": entry.detail.get("market"),
            "selection": entry.detail.get("selection"),
            "move": entry.detail.get("move"),
            "candidate_causes": preceding,
            "explained": bool(preceding),
            "note": ("information observed shortly before this movement; "
                     "co-occurrence, not proven causation")
            if preceding else "no information event recorded before this movement",
        })
    return attributions


def market_vs_model(entries: Sequence[TimelineEntry]) -> dict[str, Any]:
    """Did the model move before the market, with it, or against it?"""
    predictions = [e for e in entries if e.kind == "prediction"]
    markets = [e for e in entries if e.kind == "market" and e.detail.get("novig")]
    if len(predictions) < 2 or len(markets) < 2:
        return {"available": False,
                "reason": "needs at least two model and two market observations"}

    def series(items, extract):
        points = []
        for item in items:
            value = extract(item)
            if value is None:
                continue
            when = pd.to_datetime(item.timestamp, utc=True, errors="coerce")
            if not pd.isna(when):
                points.append((when, value))
        return sorted(points)

    selection = predictions[0].detail.get("selection")
    model_points = series(predictions,
                          lambda e: e.detail.get("model_probability")
                          if e.detail.get("selection") == selection else None)
    market_points = series(markets, lambda e: (e.detail.get("novig") or {}).get(selection))
    if len(model_points) < 2 or len(market_points) < 2:
        return {"available": False, "reason": "no comparable selection series"}

    model_move = model_points[-1][1] - model_points[0][1]
    market_move = market_points[-1][1] - market_points[0][1]
    same_direction = (model_move > 0) == (market_move > 0)

    return {
        "available": True, "selection": selection,
        "model_move": round(model_move, 5), "market_move": round(market_move, 5),
        "same_direction": same_direction,
        "model_moved_first": model_points[0][0] < market_points[0][0],
        "interpretation": (
            "model and market moved the same way" if same_direction
            else "model and market moved in opposite directions"
        ),
    }


def prediction_versions(game_uid: str) -> pd.DataFrame:
    """All prediction versions for a fixture, newest last."""
    return query_df(
        """
        SELECT prediction_id, created_at, market, selection, model_prob, market_prob,
               prediction_stage, lineup_state, model_version, superseded_at, mode
        FROM predictions WHERE game_uid = ? ORDER BY created_at, prediction_id
        """,
        (game_uid,),
    )


def lineup_impact_dataset(sport: str = "soccer") -> pd.DataFrame:
    """Pre-lineup vs post-lineup prediction pairs, for future research.

    This is the dataset that will eventually allow player-availability effects
    to be *fitted* instead of assumed. It is deliberately just accumulated here;
    fitting it before enough observations exist would be overfitting a prior
    that is already documented as a prior.
    """
    frame = query_df(
        """
        SELECT p.game_uid, p.market, p.selection, p.model_prob, p.created_at,
               p.prediction_stage, p.lineup_state, g.home_score, g.away_score, g.status
        FROM predictions p JOIN games g ON g.game_uid = p.game_uid
        WHERE p.sport = ? ORDER BY p.game_uid, p.market, p.selection, p.created_at
        """,
        (sport,),
    )
    if frame.empty:
        return frame

    rows: list[dict[str, Any]] = []
    for (game_uid, market, selection), group in frame.groupby(["game_uid", "market", "selection"]):
        pre = group[group["lineup_state"].isin(["unknown", "projected"])]
        post = group[group["lineup_state"].isin(["confirmed", "final"])]
        if pre.empty or post.empty:
            continue
        before = float(pre.iloc[-1]["model_prob"])
        after = float(post.iloc[-1]["model_prob"])
        rows.append({
            "game_uid": game_uid, "market": market, "selection": selection,
            "pre_lineup_prob": before, "post_lineup_prob": after,
            "delta": after - before,
            "pre_at": pre.iloc[-1]["created_at"], "post_at": post.iloc[-1]["created_at"],
            "status": group.iloc[-1]["status"],
        })
    return pd.DataFrame(rows)
