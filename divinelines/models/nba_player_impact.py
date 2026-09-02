"""NBA player impact and availability adjustment.

The original system had no concept of a player at all.  A crude replacement
(``team_has_injuries = true``) would be barely better: losing a franchise
player, a starting centre and a end-of-bench guard are not the same event.

This module estimates a per-player margin impact, converts a team's missing
minutes into a change in expected margin, and turns that into a change in win
probability.  Where a player's availability is uncertain it does **not** guess:
it evaluates the scenarios and returns a probability-weighted answer, plus an
explicit uncertainty measure that the staking engine uses to bet smaller.

Two honesty notes that the platform states wherever these numbers appear:

* the conversion from margin to win probability is **fitted** on the platform's
  own game data, so it is empirical;
* the conversion from box-score impact (PIE-minutes) to margin points is a
  **documented prior**, not a fitted parameter, because no historical injury
  data exists here to fit it against.  It is configurable and is surfaced in
  the explanation of every adjusted prediction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)

#: PIE of a freely available replacement-level NBA player.  League-average PIE
#: is 0.10 by construction; replacement level sits meaningfully below it.
REPLACEMENT_PIE = 0.055

#: Points of margin produced by one full game (48 minutes) of a player who is
#: one PIE point above replacement.  Anchored so that a 0.20-PIE star playing
#: 36 minutes is worth roughly 5 points of margin, which is the commonly cited
#: magnitude for a franchise player's on/off impact.  A PRIOR, not a fit.
PIE_POINTS_SCALE = 45.0

#: Fallback probability that a player with a given status takes the floor.
STATUS_PLAY_PROBABILITY: dict[str, float] = {
    "out": 0.0, "suspended": 0.0, "doubtful": 0.25, "questionable": 0.50,
    "probable": 0.85, "available": 1.0, "rest": 0.0, "unknown": 0.5,
}


@dataclass
class PlayerImpact:
    player_name: str
    team_uid: str
    minutes: float
    pie: float
    usage: float
    net_rating: float
    games_played: int
    #: Expected points of margin the player adds over a replacement, per game.
    margin_impact: float
    position: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player_name, "team_uid": self.team_uid,
            "minutes": round(self.minutes, 1), "pie": round(self.pie, 3),
            "usage": round(self.usage, 3), "games_played": self.games_played,
            "margin_impact": round(self.margin_impact, 2), "position": self.position,
        }


@dataclass
class AvailabilityAdjustment:
    """The effect of a team's availability picture on its expected margin."""

    team_uid: str
    expected_margin_delta: float
    certain_margin_delta: float
    uncertain_margin_delta: float
    missing_players: list[dict[str, Any]] = field(default_factory=list)
    uncertain_players: list[dict[str, Any]] = field(default_factory=list)
    #: Standard deviation of the margin impact across availability scenarios.
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_uid": self.team_uid,
            "expected_margin_delta": round(self.expected_margin_delta, 2),
            "certain_margin_delta": round(self.certain_margin_delta, 2),
            "uncertain_margin_delta": round(self.uncertain_margin_delta, 2),
            "uncertainty": round(self.uncertainty, 2),
            "missing_players": self.missing_players,
            "uncertain_players": self.uncertain_players,
        }


def build_player_impacts(player_stats: pd.DataFrame, *, min_games: int = 8,
                         pie_scale: float = PIE_POINTS_SCALE) -> dict[str, PlayerImpact]:
    """Convert advanced season stats into per-player margin impact.

    ``margin_impact = (PIE - replacement) * (minutes / 48) * scale`` — a
    player's box-score production above replacement, weighted by how much of
    the game he actually plays.
    """
    from ..db.repository import nba_team_uid  # local import avoids a cycle

    if player_stats is None or player_stats.empty:
        return {}

    required = {"PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PIE", "GP"}
    missing = required - set(player_stats.columns)
    if missing:
        raise KeyError(f"player stats missing columns: {sorted(missing)}")

    impacts: dict[str, PlayerImpact] = {}
    for _, row in player_stats.iterrows():
        games = int(row.get("GP") or 0)
        if games < min_games:
            continue
        team_uid = nba_team_uid(row.get("TEAM_ABBREVIATION"))
        if not team_uid:
            continue
        minutes = float(row.get("MIN") or 0.0)
        pie = float(row.get("PIE") or 0.0)
        impact = max(pie - REPLACEMENT_PIE, -0.05) * (minutes / 48.0) * pie_scale
        impacts[_key(row["PLAYER_NAME"])] = PlayerImpact(
            player_name=str(row["PLAYER_NAME"]),
            team_uid=team_uid,
            minutes=minutes,
            pie=pie,
            usage=float(row.get("USG_PCT") or 0.0),
            net_rating=float(row.get("NET_RATING") or 0.0),
            games_played=games,
            margin_impact=float(impact),
        )
    log.info("built player impacts", extra={"players": len(impacts)})
    return impacts


def _key(name: str) -> str:
    from ..identity import normalize_key  # local import avoids a cycle

    return normalize_key(name)


def team_availability(
    team_uid: str,
    statuses: Sequence[Any],
    impacts: dict[str, PlayerImpact],
    *,
    max_scenarios: int = 4,
) -> AvailabilityAdjustment:
    """Expected margin change for one team given its injury report.

    Players whose availability is certain (out/suspended) contribute directly.
    Uncertain players contribute their probability-weighted impact, and the
    spread across scenarios becomes an uncertainty measure that reduces stake
    size downstream.
    """
    certain: list[dict[str, Any]] = []
    uncertain: list[tuple[float, PlayerImpact, str]] = []

    for status in statuses:
        impact = impacts.get(_key(getattr(status, "player_name", "")))
        if impact is None or impact.team_uid != team_uid:
            continue
        state = getattr(status, "status", "unknown")
        play_probability = getattr(status, "play_probability", None)
        if play_probability is None:
            play_probability = STATUS_PLAY_PROBABILITY.get(state, 0.5)

        if play_probability <= 0.0:
            certain.append({**impact.to_dict(), "status": state, "play_probability": 0.0})
        elif play_probability < 1.0:
            uncertain.append((float(play_probability), impact, state))

    certain_delta = -sum(item["margin_impact"] for item in certain)
    expected_uncertain = -sum((1.0 - p) * impact.margin_impact for p, impact, _ in uncertain)

    # Scenario spread: enumerate combinations for the most influential
    # uncertain players so the uncertainty is real rather than assumed.
    uncertainty = 0.0
    if uncertain:
        ranked = sorted(uncertain, key=lambda item: -abs(item[1].margin_impact))[:max_scenarios]
        deltas: list[float] = []
        weights: list[float] = []
        for combination in itertools.product([True, False], repeat=len(ranked)):
            weight = 1.0
            delta = 0.0
            for plays, (probability, impact, _) in zip(combination, ranked):
                weight *= probability if plays else (1.0 - probability)
                if not plays:
                    delta -= impact.margin_impact
            deltas.append(delta)
            weights.append(weight)
        deltas_array = np.array(deltas)
        weights_array = np.array(weights)
        mean = float(np.sum(deltas_array * weights_array))
        variance = float(np.sum(weights_array * (deltas_array - mean) ** 2))
        uncertainty = float(np.sqrt(max(variance, 0.0)))

    return AvailabilityAdjustment(
        team_uid=team_uid,
        expected_margin_delta=certain_delta + expected_uncertain,
        certain_margin_delta=certain_delta,
        uncertain_margin_delta=expected_uncertain,
        missing_players=certain,
        uncertain_players=[
            {**impact.to_dict(), "status": state, "play_probability": probability}
            for probability, impact, state in uncertain
        ],
        uncertainty=uncertainty,
    )


# --------------------------------------------------------------------------
# Margin <-> probability
# --------------------------------------------------------------------------

def fit_margin_to_probability(dataset: pd.DataFrame,
                              feature: str = "diff_adj_net") -> float:
    """Fit how many logits of win probability one point of margin is worth.

    Uses the platform's own games: a logistic regression of the home win on a
    pre-game margin estimate.  Returns the coefficient in logits per point, so
    the injury adjustment is grounded in this data rather than a rule of thumb.
    """
    from sklearn.linear_model import LogisticRegression  # local import

    frame = dataset[[feature, "home_win"]].dropna()
    if len(frame) < 500:
        log.warning("insufficient data to fit margin conversion; using default",
                    extra={"rows": len(frame)})
        return 0.115
    model = LogisticRegression(max_iter=1000)
    model.fit(frame[[feature]].to_numpy(), frame["home_win"].astype(int).to_numpy())
    coefficient = float(model.coef_[0][0])
    log.info("fitted margin-to-probability conversion",
             extra={"logits_per_point": round(coefficient, 5), "rows": len(frame)})
    return coefficient


def apply_margin_adjustment(probability: float, margin_delta: float,
                            logits_per_point: float) -> float:
    """Shift a win probability by a margin change, in logit space."""
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    logit = np.log(probability / (1 - probability)) + logits_per_point * margin_delta
    # Clip the result too: a probability of exactly 0 or 1 would make Kelly and
    # log loss undefined downstream, and no adjustment justifies certainty.
    return float(np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-6, 1 - 1e-6))


def scenario_weighted_probability(
    base_probability: float,
    home_adjustment: AvailabilityAdjustment,
    away_adjustment: AvailabilityAdjustment,
    logits_per_point: float,
) -> dict[str, Any]:
    """Combine both teams' availability into one adjusted probability.

    The returned ``uncertainty`` is the probability-space spread implied by the
    uncertain players, which the staking engine uses to shrink Kelly.
    """
    net_margin_delta = home_adjustment.expected_margin_delta - away_adjustment.expected_margin_delta
    adjusted = apply_margin_adjustment(base_probability, net_margin_delta, logits_per_point)

    combined_uncertainty = float(np.hypot(home_adjustment.uncertainty,
                                          away_adjustment.uncertainty))
    upper = apply_margin_adjustment(base_probability, net_margin_delta + combined_uncertainty,
                                    logits_per_point)
    lower = apply_margin_adjustment(base_probability, net_margin_delta - combined_uncertainty,
                                    logits_per_point)
    return {
        "base_probability": base_probability,
        "adjusted_probability": adjusted,
        "margin_delta": net_margin_delta,
        "probability_delta": adjusted - base_probability,
        "uncertainty_margin": combined_uncertainty,
        "probability_range": [min(lower, upper), max(lower, upper)],
        "home": home_adjustment.to_dict(),
        "away": away_adjustment.to_dict(),
        "logits_per_point": logits_per_point,
        "method": "PIE-minutes impact over replacement, scenario-weighted",
    }
