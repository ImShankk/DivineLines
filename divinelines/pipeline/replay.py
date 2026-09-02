"""Historical replay: turn a walk-forward backtest into a real prediction ledger.

V2 could not answer "does DivineLines beat the market in the NBA?" because no
historical NBA prices existed. ESPN's core API publishes an opening and a
closing moneyline per game, so the question is now answerable — but only if
historical predictions live in the same ledger, and flow through the same
settlement and CLV machinery, as live ones. Two parallel evaluation paths would
diverge, and the backtest path would be the one nobody checked.

So replay writes ordinary ``predictions`` rows with ``mode='backtest'``:

    model trained on prior seasons only
        -> prediction at the OPENING price
            -> settled against the CLOSING price
                -> CLV, ROI, market-relative log loss

## What is real here and what is not

* **Real**: the model saw only prior games; the entry price is a genuine
  bookmaker opening price; the closing price is a genuine closing price; the
  result is the real result.
* **Not real**: the *timestamps*. ESPN does not publish when the open was
  posted, so ``created_at`` is a nominal stamp and ``seconds_to_event`` is left
  NULL. Time-to-event CLV cohorts therefore exclude replay rows — they would be
  reporting a bucket derived from a number nobody measured.

Betting the open and being graded against the close is also the most flattering
honest framing available, since prices generally sharpen toward kick-off. That
is stated wherever these results appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..betting.ev import expected_value
from ..betting.kelly import recommend_stake
from ..betting.ledger import PredictionRecord, record_predictions
from ..betting.odds_math import remove_vig
from ..config import settings
from ..db.connection import query_df, write_connection
from ..db.repository import now_iso
from ..logging_setup import get_logger
from ..models.nba_model import DEFAULT_VARIANT, NbaWinModel, resolve_features
from ..models.registry import data_version_hash

log = get_logger(__name__)

REPLAY_MODE = "backtest"
#: Nominal entry timestamp: the opening price has no published capture time, so
#: the ledger records the day before the event and flags the fact rather than
#: inventing a plausible-looking clock reading.
NOMINAL_ENTRY_OFFSET = timedelta(days=1)


@dataclass
class ReplayReport:
    sport: str
    seasons: list[str]
    games_predicted: int = 0
    predictions_written: int = 0
    games_with_open: int = 0
    games_without_open: int = 0
    model_versions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sport": self.sport, "seasons": self.seasons,
            "games_predicted": self.games_predicted,
            "predictions_written": self.predictions_written,
            "games_with_open": self.games_with_open,
            "games_without_open": self.games_without_open,
            "model_versions": self.model_versions, "notes": self.notes,
        }


def _opening_prices(sport: str = "nba", market: str = "h2h") -> dict[str, dict[str, Any]]:
    """Per game: consensus opening price and the best book at the open.

    The median across books is the reference the model is judged against; the
    best price is what a bettor shopping lines could actually have taken. Both
    are returned so the two questions stay separable.
    """
    frame = query_df(
        """
        SELECT game_uid, selection, bookmaker, price_decimal
        FROM odds_snapshots
        WHERE sport = ? AND market = ? AND phase = 'open'
        """,
        (sport, market),
    )
    if frame.empty:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for game_uid, group in frame.groupby("game_uid"):
        books: dict[str, dict[str, float]] = {}
        for _, row in group.iterrows():
            books.setdefault(str(row["bookmaker"]), {})[str(row["selection"])] = float(
                row["price_decimal"]
            )
        complete = {b: q for b, q in books.items() if {"home", "away"} <= set(q)}
        if not complete:
            continue
        consensus = {
            selection: float(np.median([q[selection] for q in complete.values()]))
            for selection in ("home", "away")
        }
        best: dict[str, tuple[str, float]] = {}
        for selection in ("home", "away"):
            book, price = max(((b, q[selection]) for b, q in complete.items()),
                              key=lambda item: item[1])
            best[selection] = (book, price)
        try:
            fair = dict(zip(("home", "away"),
                            remove_vig([consensus["home"], consensus["away"]])))
        except (ValueError, ZeroDivisionError):
            continue
        result[str(game_uid)] = {
            "consensus": consensus, "best": best, "fair": fair, "n_books": len(complete),
        }
    return result


def replay_nba(seasons: Sequence[str] | None = None, *, variant: str | None = None,
               min_train_rows: int = 800, price_at: str = "best",
               clear_existing: bool = True) -> ReplayReport:
    """Walk-forward NBA predictions written to the ledger at opening prices."""
    from ..backtest.nba_backtest import prepare_nba_dataset

    variant = variant or DEFAULT_VARIANT
    dataset = prepare_nba_dataset()
    dataset = dataset[dataset["home_win"].notna()].copy()
    seasons = list(seasons or sorted(dataset["season"].unique())[-3:])
    report = ReplayReport(sport="nba", seasons=seasons)

    opens = _opening_prices("nba", "h2h")
    if not opens:
        report.notes.append("no opening prices stored — run `divinelines backfill-odds` first")
        return report

    if clear_existing:
        with write_connection() as conn:
            conn.execute("DELETE FROM clv_records WHERE prediction_id IN "
                         "(SELECT prediction_id FROM predictions WHERE mode = ?)", (REPLAY_MODE,))
            conn.execute("DELETE FROM predictions WHERE mode = ?", (REPLAY_MODE,))

    features = resolve_features(variant, dataset.columns)
    data_version = data_version_hash()
    kickoffs = _kickoff_lookup(dataset["game_uid"].tolist())

    for season in seasons:
        test = dataset[dataset["season"] == season]
        if test.empty:
            continue
        cutoff = test["game_date"].min()
        history = dataset[dataset["game_date"] < cutoff]
        if len(history) < min_train_rows:
            report.notes.append(f"{season}: insufficient history ({len(history)} rows)")
            continue

        split_at = int(len(history) * 0.85)
        model = NbaWinModel(features=list(features), variant=variant)
        model.fit(history.iloc[:split_at], history.iloc[split_at:])
        model_version = f"{model.model_version}+{variant}"
        if model_version not in report.model_versions:
            report.model_versions.append(model_version)

        detail = model.predict_detail(test)
        records: list[PredictionRecord] = []

        for position, (_, row) in enumerate(test.iterrows()):
            game_uid = str(row["game_uid"])
            market = opens.get(game_uid)
            report.games_predicted += 1
            if not market:
                report.games_without_open += 1
                continue
            report.games_with_open += 1

            home_probability = float(detail["probability"][position])
            entry_stamp = _nominal_entry(kickoffs.get(game_uid), row["game_date"])

            for selection, probability in (("home", home_probability),
                                           ("away", 1.0 - home_probability)):
                fair = market["fair"][selection]
                book, best_price = market["best"][selection]
                price = best_price if price_at == "best" else market["consensus"][selection]
                book_label = book if price_at == "best" else "consensus_open"

                edge = probability - fair
                ev = expected_value(probability, price, fair)
                stake = recommend_stake(
                    model_probability=probability, price_decimal=price,
                    market_probability=fair, confidence=float(detail["agreement"][position]),
                ).stake if ev.ev_per_unit > 0 and edge >= settings.betting.min_edge else 0.0

                records.append(PredictionRecord(
                    sport="nba", game_uid=game_uid, market="h2h", selection=selection,
                    model_probability=probability, league_id="NBA",
                    market_probability=fair, price_decimal=price, bookmaker=book_label,
                    edge=edge, ev_per_unit=ev.ev_per_unit,
                    stake=stake, confidence=float(detail["agreement"][position]),
                    data_quality=None, model_id=None, model_version=model_version,
                    data_version=data_version,
                    features={"variant": variant, "n_books_open": market["n_books"]},
                    flags=["REPLAY"], created_at=entry_stamp, mode=REPLAY_MODE,
                    prediction_stage="pre_event", lineup_state="unknown",
                    feature_version=model.feature_set_version,
                    event_start_utc=kickoffs.get(game_uid),
                    # Left NULL on purpose: the opening price carries no capture
                    # time, so any time-to-event figure here would be fiction.
                    seconds_to_event=None,
                    information_snapshot={"entry_phase": "open",
                                          "entry_basis": price_at,
                                          "timestamps": "nominal"},
                ))

        report.predictions_written += len(record_predictions(records))
        log.info("replayed season", extra={"season": season, "rows": len(records),
                                           "model_version": model_version})

    report.notes.append(
        "Entry prices are bookmaker OPENING moneylines; settlement uses CLOSING "
        "prices. Timestamps are nominal — ESPN publishes no capture time for "
        "either phase — so time-to-event cohorts exclude these rows."
    )
    log.info("nba replay complete", extra=report.to_dict())
    return report


def _kickoff_lookup(game_uids: Sequence[str]) -> dict[str, str | None]:
    if not game_uids:
        return {}
    frame = query_df("SELECT game_uid, kickoff_utc, game_date FROM games WHERE sport = 'nba'")
    return {
        str(row["game_uid"]): (row["kickoff_utc"] or None)
        for _, row in frame.iterrows()
    }


def _nominal_entry(kickoff: str | None, game_date: Any) -> str:
    base = None
    if kickoff:
        base = pd.to_datetime(kickoff, utc=True, errors="coerce")
    if base is None or pd.isna(base):
        base = pd.to_datetime(game_date, utc=True, errors="coerce")
    if pd.isna(base):
        base = pd.Timestamp.now(tz="UTC")
    return (base - NOMINAL_ENTRY_OFFSET).isoformat()
