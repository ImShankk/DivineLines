"""Data validation.

Invalid data must fail loudly, not quietly poison a model.  Checks are split
into ``critical`` (block the pipeline) and ``warning`` (record and continue).
Every issue is written to ``validation_issues`` so the system status page can
surface data health over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from ..logging_setup import get_logger
from .connection import query_df, upsert_rows

log = get_logger(__name__)


class ValidationError(RuntimeError):
    """Raised when critical validation fails and the caller asked to be strict."""


@dataclass
class Issue:
    code: str
    severity: str  # 'critical' | 'warning'
    detail: str
    entity: str | None = None


@dataclass
class ValidationReport:
    dataset: str
    issues: list[Issue] = field(default_factory=list)
    checked: int = 0

    @property
    def critical(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.critical

    def add(self, code: str, severity: str, detail: str, entity: str | None = None) -> None:
        self.issues.append(Issue(code, severity, detail, entity))

    def raise_if_critical(self) -> None:
        if self.critical:
            summary = "; ".join(f"{i.code}: {i.detail}" for i in self.critical[:5])
            raise ValidationError(f"{self.dataset}: {summary}")

    def persist(self) -> None:
        if not self.issues:
            return
        now = datetime.now(timezone.utc).isoformat()
        upsert_rows(
            "validation_issues",
            [
                {
                    "detected_at": now,
                    "dataset": self.dataset,
                    "severity": i.severity,
                    "code": i.code,
                    "detail": i.detail,
                    "entity": i.entity,
                }
                for i in self.issues
            ],
        )

    def log(self) -> None:
        for issue in self.issues:
            logger = log.error if issue.severity == "critical" else log.warning
            logger(
                "validation issue",
                extra={
                    "dataset": self.dataset,
                    "code": issue.code,
                    "entity": issue.entity,
                    "detail": issue.detail,
                },
            )


# --------------------------------------------------------------------------
# Generic checks
# --------------------------------------------------------------------------

def _sample(values: Sequence[Any], n: int = 5) -> str:
    listed = list(values)[:n]
    return ", ".join(str(v) for v in listed)


def validate_games(df: pd.DataFrame, dataset: str = "games") -> ValidationReport:
    """Structural checks that apply to any sport's game table."""
    report = ValidationReport(dataset=dataset, checked=len(df))
    if df.empty:
        report.add("empty_dataset", "critical", "no rows produced")
        return report

    required = {"game_uid", "game_date", "home_team_uid", "away_team_uid", "status"}
    missing = required - set(df.columns)
    if missing:
        report.add("missing_columns", "critical", f"missing {sorted(missing)}")
        return report

    dupes = df["game_uid"][df["game_uid"].duplicated()]
    if len(dupes):
        report.add("duplicate_game_uid", "critical", f"{len(dupes)} duplicates: {_sample(dupes)}")

    same_team = df[df["home_team_uid"] == df["away_team_uid"]]
    if len(same_team):
        report.add("self_matchup", "critical", f"{len(same_team)} rows where home == away",
                   _sample(same_team["game_uid"]))

    null_teams = df[df["home_team_uid"].isna() | df["away_team_uid"].isna()]
    if len(null_teams):
        report.add("unresolved_team", "critical",
                   f"{len(null_teams)} rows with unresolved team identity",
                   _sample(null_teams["game_uid"]))

    dates = pd.to_datetime(df["game_date"], errors="coerce")
    bad_dates = df[dates.isna()]
    if len(bad_dates):
        report.add("invalid_date", "critical", f"{len(bad_dates)} unparseable dates",
                   _sample(bad_dates["game_uid"]))
    else:
        future = df[dates > pd.Timestamp.now(tz='UTC').tz_localize(None) + pd.Timedelta(days=400)]
        if len(future):
            report.add("implausible_future_date", "warning",
                       f"{len(future)} games more than 400 days ahead", _sample(future["game_uid"]))

    finals = df[df["status"] == "final"]
    if len(finals):
        missing_scores = finals[finals["home_score"].isna() | finals["away_score"].isna()]
        if len(missing_scores):
            report.add("final_without_score", "critical",
                       f"{len(missing_scores)} final games missing scores",
                       _sample(missing_scores["game_uid"]))
        negative = finals[(finals["home_score"] < 0) | (finals["away_score"] < 0)]
        if len(negative):
            report.add("negative_score", "critical", f"{len(negative)} negative scores",
                       _sample(negative["game_uid"]))
    return report


def validate_nba_box(df: pd.DataFrame) -> ValidationReport:
    """NBA-specific sanity: two rows per game, plausible box-score ranges."""
    report = ValidationReport(dataset="nba_team_game", checked=len(df))
    if df.empty:
        report.add("empty_dataset", "critical", "no NBA box score rows")
        return report

    per_game = df.groupby("game_uid").size()
    wrong = per_game[per_game != 2]
    if len(wrong):
        report.add("bad_team_count", "critical",
                   f"{len(wrong)} games without exactly 2 team rows", _sample(wrong.index))

    if "pts" in df:
        implausible = df[(df["pts"] < 40) | (df["pts"] > 200)]
        if len(implausible):
            report.add("implausible_points", "critical",
                       f"{len(implausible)} team scores outside [40, 200]",
                       _sample(implausible["game_uid"]))

    for col, lo, hi in (("fga", 40, 140), ("fta", 0, 80), ("reb", 15, 90), ("tov", 0, 40)):
        if col in df:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                report.add(f"implausible_{col}", "warning",
                           f"{len(bad)} rows with {col} outside [{lo}, {hi}]",
                           _sample(bad["game_uid"]))

    if {"fgm", "fga"} <= set(df.columns):
        bad = df[df["fgm"] > df["fga"]]
        if len(bad):
            report.add("fgm_gt_fga", "critical", f"{len(bad)} rows with makes > attempts",
                       _sample(bad["game_uid"]))
    return report


def validate_odds(rows: Sequence[dict[str, Any]]) -> ValidationReport:
    report = ValidationReport(dataset="odds", checked=len(rows))
    for row in rows:
        price = row.get("price_decimal")
        if price is None or not isinstance(price, (int, float)):
            report.add("missing_price", "critical", f"no price for {row.get('selection')}",
                       str(row.get("game_uid")))
        elif price <= 1.0 or price > 1000:
            report.add("implausible_price", "critical",
                       f"decimal odds {price} out of range", str(row.get("game_uid")))
    return report


def validate_probabilities(probs: dict[str, float], *, tolerance: float = 1e-6,
                           dataset: str = "probabilities") -> ValidationReport:
    """Probabilities must be in [0,1] and (for a full market) sum to 1."""
    report = ValidationReport(dataset=dataset, checked=len(probs))
    for name, value in probs.items():
        if value is None or not (0.0 - tolerance <= float(value) <= 1.0 + tolerance):
            report.add("probability_out_of_range", "critical", f"{name}={value}")
    total = sum(float(v) for v in probs.values() if v is not None)
    if probs and abs(total - 1.0) > 1e-3:
        report.add("probabilities_do_not_sum", "critical", f"sum={total:.6f}")
    return report


def run_database_health_checks() -> ValidationReport:
    """Cross-table integrity checks run by the status endpoint and the CLI."""
    report = ValidationReport(dataset="database")

    orphans = query_df(
        "SELECT COUNT(*) AS n FROM nba_team_game t "
        "LEFT JOIN games g ON g.game_uid = t.game_uid WHERE g.game_uid IS NULL"
    )
    if not orphans.empty and int(orphans["n"].iloc[0]) > 0:
        report.add("orphan_box_rows", "critical", f"{int(orphans['n'].iloc[0])} box rows without a game")

    dupes = query_df(
        "SELECT COUNT(*) AS n FROM (SELECT sport, league_id, season, game_date, "
        "home_team_uid, away_team_uid, COUNT(*) c FROM games GROUP BY 1,2,3,4,5,6 HAVING c > 1)"
    )
    if not dupes.empty and int(dupes["n"].iloc[0]) > 0:
        report.add("duplicate_fixtures", "critical", f"{int(dupes['n'].iloc[0])} duplicated fixtures")

    bad_scores = query_df(
        "SELECT COUNT(*) AS n FROM games WHERE status='final' AND "
        "(home_score IS NULL OR away_score IS NULL OR home_score < 0 OR away_score < 0)"
    )
    if not bad_scores.empty and int(bad_scores["n"].iloc[0]) > 0:
        report.add("invalid_final_scores", "critical", f"{int(bad_scores['n'].iloc[0])} bad final scores")

    unresolved = query_df(
        "SELECT COUNT(*) AS n FROM games g LEFT JOIN teams t ON t.team_uid = g.home_team_uid "
        "WHERE t.team_uid IS NULL"
    )
    if not unresolved.empty and int(unresolved["n"].iloc[0]) > 0:
        report.add("unknown_team_reference", "critical",
                   f"{int(unresolved['n'].iloc[0])} games referencing unknown teams")

    return report
