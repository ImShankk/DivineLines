"""Lineup features.

The platform has no external player-rating feed for soccer, so "lineup
strength" cannot be a sum of player ratings. What it *can* be is a set of
quantities derivable from the lineup history itself:

* how much of the XI are the team's regular starters,
* how many regulars are missing,
* whether the usual goalkeeper is playing,
* how much continuity the XI has with recent selections.

That is a deliberately modest feature set. It is also the honest one: inventing
a player-quality score from data the platform does not have would produce a
number that looks sophisticated and means nothing.

Like every other builder here this is a single chronological pass — a team's
"regular starters" at match *n* are computed from matches 1..n-1 only.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..identity import normalize_key
from ..logging_setup import get_logger

log = get_logger(__name__)

FEATURE_SET_VERSION = "lineup_v3.0"

#: Positional weights for the "weighted regulars missing" feature. A goalkeeper
#: is not one eleventh of a team: the position is uniquely non-substitutable,
#: and a reserve keeper is usually a much larger drop-off than a reserve
#: midfielder. These are priors, exposed here rather than buried, and the
#: ablation decides whether the weighting earns its place at all.
POSITION_WEIGHTS: dict[str, float] = {
    "goalkeeper": 2.0,
    "defender": 1.0,
    "midfielder": 1.0,
    "forward": 1.2,
}
DEFAULT_POSITION_WEIGHT = 1.0

#: Matches of history used to decide who counts as a regular.
REGULAR_WINDOW = 10
#: Start rate at or above which a player is a "regular starter".
REGULAR_THRESHOLD = 0.6

LINEUP_FEATURES: tuple[str, ...] = (
    "diff_xi_regular_share",
    "diff_missing_regulars",
    "diff_weighted_missing",
    "diff_xi_continuity",
    "home_gk_is_regular",
    "away_gk_is_regular",
    "diff_gk_is_regular",
    "lineup_coverage",
)


@dataclass
class TeamLineupState:
    """Per-team selection history, as known before the current match."""

    #: Most recent XIs, as sets of normalised player keys.
    recent_xis: deque = field(default_factory=lambda: deque(maxlen=REGULAR_WINDOW))
    #: Starts per player across the tracked window.
    start_counts: Counter = field(default_factory=Counter)
    #: Goalkeepers seen starting, most frequent = the regular keeper.
    gk_counts: Counter = field(default_factory=Counter)
    #: Position group last seen for each player, for weighting absences.
    positions: dict[str, str] = field(default_factory=dict)
    matches_observed: int = 0

    def regulars(self) -> set[str]:
        if self.matches_observed == 0:
            return set()
        threshold = REGULAR_THRESHOLD * min(self.matches_observed, REGULAR_WINDOW)
        return {player for player, count in self.start_counts.items() if count >= threshold}

    def regular_goalkeeper(self) -> str | None:
        return self.gk_counts.most_common(1)[0][0] if self.gk_counts else None

    def apply(self, xi: set[str], goalkeeper: str | None,
              positions: dict[str, str]) -> None:
        if len(self.recent_xis) == self.recent_xis.maxlen and self.recent_xis:
            for player in self.recent_xis[0]:
                self.start_counts[player] -= 1
                if self.start_counts[player] <= 0:
                    del self.start_counts[player]
        self.recent_xis.append(xi)
        self.start_counts.update(xi)
        if goalkeeper:
            self.gk_counts[goalkeeper] += 1
        self.positions.update(positions)
        self.matches_observed += 1


class LineupFeatureBuilder:
    """Builds lineup features for matches, in time order."""

    def __init__(self) -> None:
        self.states: dict[str, TeamLineupState] = defaultdict(TeamLineupState)
        self.version = FEATURE_SET_VERSION

    def features_for(self, home_uid: str, away_uid: str,
                     home_xi: dict[str, Any] | None,
                     away_xi: dict[str, Any] | None) -> dict[str, float]:
        """Emit features from the state *before* this match is applied."""
        home = self._team_features(home_uid, home_xi)
        away = self._team_features(away_uid, away_xi)

        record: dict[str, float] = {}
        for key in ("xi_regular_share", "missing_regulars", "weighted_missing",
                    "xi_continuity", "gk_is_regular"):
            record[f"home_{key}"] = home[key]
            record[f"away_{key}"] = away[key]
            record[f"diff_{key}"] = (
                np.nan if (np.isnan(home[key]) or np.isnan(away[key]))
                else home[key] - away[key]
            )
        # Coverage says how much of this row is real rather than imputed; the
        # model should be able to learn to distrust rows where it is 0.
        record["lineup_coverage"] = float(
            (0.5 if home_xi else 0.0) + (0.5 if away_xi else 0.0)
        )
        return record

    def _team_features(self, team_uid: str, xi: dict[str, Any] | None) -> dict[str, float]:
        state = self.states[team_uid]
        if not xi or not xi.get("starters"):
            return {k: np.nan for k in ("xi_regular_share", "missing_regulars",
                                        "weighted_missing", "xi_continuity", "gk_is_regular")}

        starters: set[str] = {normalize_key(name) for name in xi["starters"]}
        regulars = state.regulars()
        window = min(state.matches_observed, REGULAR_WINDOW) or 1

        if not regulars:
            # No history yet: report absence rather than a confident zero.
            return {k: np.nan for k in ("xi_regular_share", "missing_regulars",
                                        "weighted_missing", "xi_continuity", "gk_is_regular")}

        present = starters & regulars
        missing = regulars - starters
        weighted_missing = sum(
            POSITION_WEIGHTS.get(state.positions.get(player, ""), DEFAULT_POSITION_WEIGHT)
            for player in missing
        )
        continuity = float(np.mean([state.start_counts.get(p, 0) / window for p in starters]))

        regular_gk = state.regular_goalkeeper()
        gk = normalize_key(xi["goalkeeper"]) if xi.get("goalkeeper") else None
        gk_is_regular = (np.nan if (regular_gk is None or gk is None)
                         else float(gk == regular_gk))

        return {
            "xi_regular_share": len(present) / max(len(regulars), 1),
            "missing_regulars": float(len(missing)),
            "weighted_missing": float(weighted_missing),
            "xi_continuity": continuity,
            "gk_is_regular": gk_is_regular,
        }

    def apply(self, team_uid: str, xi: dict[str, Any] | None) -> None:
        if not xi or not xi.get("starters"):
            return
        starters = {normalize_key(name) for name in xi["starters"]}
        goalkeeper = normalize_key(xi["goalkeeper"]) if xi.get("goalkeeper") else None
        positions = {
            normalize_key(name): group
            for name, group in (xi.get("positions") or {}).items()
        }
        self.states[team_uid].apply(starters, goalkeeper, positions)


# --------------------------------------------------------------------------
# Loading observations
# --------------------------------------------------------------------------

def load_lineup_xis(game_uids: Iterable[str] | None = None, *,
                    allow_final: bool = True,
                    as_of_column: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """``(game_uid, team_uid) -> {starters, goalkeeper, positions, state}``.

    ``allow_final`` must be left on only for research: ``final`` rows were
    observed after kick-off, so they describe what happened rather than what was
    knowable. Live prediction calls this with ``allow_final=False``.
    """
    from ..db.connection import query_df

    clauses = ["status = 'starter'"]
    params: list[Any] = []
    if not allow_final:
        clauses.append("lineup_state != 'final'")
    if game_uids is not None:
        uids = list(game_uids)
        if not uids:
            return {}
        clauses.append(f"game_uid IN ({','.join('?' for _ in uids)})")
        params.extend(uids)

    frame = query_df(
        f"""
        SELECT game_uid, team_uid, player_name, position_group, formation,
               lineup_state, observed_at
        FROM lineup_observations
        WHERE {' AND '.join(clauses)}
        ORDER BY observed_at
        """,
        params,
    )
    if frame.empty:
        return {}

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for (game_uid, team_uid), group in frame.groupby(["game_uid", "team_uid"]):
        # Keep the most recent observation only; earlier ones are superseded.
        latest = group[group["observed_at"] == group["observed_at"].max()]
        goalkeepers = latest[latest["position_group"] == "goalkeeper"]["player_name"].tolist()
        result[(str(game_uid), str(team_uid))] = {
            "starters": latest["player_name"].tolist(),
            "goalkeeper": goalkeepers[0] if goalkeepers else None,
            "positions": dict(zip(latest["player_name"], latest["position_group"].fillna(""))),
            "state": str(latest["lineup_state"].iloc[0]),
            "formation": latest["formation"].iloc[0] if "formation" in latest else None,
            "observed_at": str(latest["observed_at"].iloc[0]),
        }
    return result


def attach_lineup_features(dataset: pd.DataFrame, *, allow_final: bool = True
                           ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add lineup features to a match dataset, in chronological order.

    Returns the augmented frame plus coverage statistics, because a feature
    present on 8% of rows is a different thing from one present on all of them
    and the ablation needs to know which it is looking at.
    """
    if dataset.empty:
        return dataset, {"rows": 0, "with_lineups": 0, "coverage": 0.0}

    ordered = dataset.sort_values("game_date").reset_index(drop=True)
    xis = load_lineup_xis(ordered["game_uid"].tolist(), allow_final=allow_final)
    builder = LineupFeatureBuilder()

    rows: list[dict[str, float]] = []
    with_lineups = 0
    for _, row in ordered.iterrows():
        home_uid, away_uid = row["home_team_uid"], row["away_team_uid"]
        home_xi = xis.get((row["game_uid"], home_uid))
        away_xi = xis.get((row["game_uid"], away_uid))
        rows.append(builder.features_for(home_uid, away_uid, home_xi, away_xi))
        if home_xi and away_xi:
            with_lineups += 1
        builder.apply(home_uid, home_xi)
        builder.apply(away_uid, away_xi)

    features = pd.DataFrame(rows, index=ordered.index)
    augmented = pd.concat([ordered, features], axis=1)
    stats = {
        "rows": len(ordered),
        "with_lineups": with_lineups,
        "coverage": round(with_lineups / len(ordered), 4),
        "usable_rows": int(features["diff_xi_regular_share"].notna().sum()),
        "feature_version": FEATURE_SET_VERSION,
    }
    log.info("attached lineup features", extra=stats)
    return augmented, stats
