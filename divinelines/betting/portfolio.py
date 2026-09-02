"""Portfolio-level risk control.

Ten independently +EV bets on one slate are not ten independent bets: they
share model error, they share a day, and several of them touch the same game
or team.  Staking each at its own Kelly size can therefore risk far more of
the bankroll than any single recommendation implies.

This module takes a set of candidate stakes and scales them down until every
exposure limit holds, reporting exactly which constraint bound each one.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import settings
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Candidate:
    """One prospective bet entering the risk engine."""

    key: str                     # unique id for the selection
    game_uid: str
    sport: str
    market: str
    selection: str
    teams: tuple[str, ...]       # team uids involved (correlation grouping)
    price_decimal: float
    stake: float                 # pre-risk stake from Kelly
    model_probability: float
    edge: float
    edge_score: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Allocation:
    candidate: Candidate
    stake: float
    scaled_by: float
    binding_constraints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.candidate.key,
            "game_uid": self.candidate.game_uid,
            "sport": self.candidate.sport,
            "market": self.candidate.market,
            "selection": self.candidate.selection,
            "price_decimal": self.candidate.price_decimal,
            "requested_stake": round(self.candidate.stake, 2),
            "stake": round(self.stake, 2),
            "scaled_by": round(self.scaled_by, 4),
            "binding_constraints": self.binding_constraints,
            "edge": round(self.candidate.edge, 4),
            "edge_score": round(self.candidate.edge_score, 2),
        }


@dataclass
class PortfolioResult:
    allocations: list[Allocation]
    total_stake: float
    bankroll: float
    dropped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exposure_pct(self) -> float:
        return self.total_stake / self.bankroll if self.bankroll else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bankroll": self.bankroll,
            "total_stake": round(self.total_stake, 2),
            "exposure_pct": round(self.exposure_pct, 4),
            "n_bets": len(self.allocations),
            "allocations": [a.to_dict() for a in self.allocations],
            "dropped": self.dropped,
        }


def _scale_group(allocations: list[Allocation], keys: Iterable[str],
                 limit: float, label: str) -> None:
    """Scale a group of allocations down proportionally to respect ``limit``."""
    members = [a for a in allocations if a.candidate.key in set(keys)]
    total = sum(a.stake for a in members)
    if total <= limit or total <= 0:
        return
    factor = limit / total
    for allocation in members:
        allocation.stake *= factor
        allocation.scaled_by *= factor
        allocation.binding_constraints.append(label)


def build_portfolio(
    candidates: Sequence[Candidate],
    *,
    bankroll: float | None = None,
    max_slate_pct: float | None = None,
    max_game_pct: float | None = None,
    max_team_pct: float | None = None,
    max_sport_pct: float | None = None,
    existing_exposure: float = 0.0,
) -> PortfolioResult:
    """Apply per-game, per-team, per-sport and slate exposure caps.

    Candidates are processed best-first by edge score, so when a cap bites it
    is the weakest opportunities that give up the most stake.
    """
    bankroll = settings.betting.bankroll if bankroll is None else float(bankroll)
    slate_limit = bankroll * (settings.betting.max_slate_exposure_pct
                              if max_slate_pct is None else max_slate_pct)
    game_limit = bankroll * (settings.betting.max_game_exposure_pct
                             if max_game_pct is None else max_game_pct)
    team_limit = bankroll * (settings.betting.max_team_exposure_pct
                             if max_team_pct is None else max_team_pct)
    sport_limit = bankroll * (max_sport_pct if max_sport_pct is not None
                              else settings.betting.max_slate_exposure_pct)

    ordered = sorted(candidates, key=lambda c: (-c.edge_score, -c.edge))
    allocations = [
        Allocation(candidate=c, stake=float(c.stake), scaled_by=1.0, binding_constraints=[])
        for c in ordered if c.stake > 0
    ]
    dropped = [
        {"key": c.key, "reason": "zero stake from Kelly"} for c in ordered if c.stake <= 0
    ]
    if not allocations:
        return PortfolioResult([], 0.0, bankroll, dropped)

    by_game: dict[str, list[str]] = defaultdict(list)
    by_team: dict[str, list[str]] = defaultdict(list)
    by_sport: dict[str, list[str]] = defaultdict(list)
    for allocation in allocations:
        candidate = allocation.candidate
        by_game[candidate.game_uid].append(candidate.key)
        by_sport[candidate.sport].append(candidate.key)
        for team in candidate.teams:
            by_team[team].append(candidate.key)

    for game_uid, keys in by_game.items():
        _scale_group(allocations, keys, game_limit, f"game_cap:{game_uid}")
    for team, keys in by_team.items():
        _scale_group(allocations, keys, team_limit, f"team_cap:{team}")
    for sport, keys in by_sport.items():
        _scale_group(allocations, keys, sport_limit, f"sport_cap:{sport}")

    remaining_slate = max(slate_limit - existing_exposure, 0.0)
    _scale_group(allocations, [a.candidate.key for a in allocations],
                 remaining_slate, "slate_cap")

    # Drop stakes that risk-scaling reduced below a meaningful size.
    kept: list[Allocation] = []
    for allocation in allocations:
        if allocation.stake < 0.01:
            dropped.append({"key": allocation.candidate.key,
                            "reason": "stake scaled below minimum by risk limits"})
        else:
            allocation.stake = round(allocation.stake, 2)
            kept.append(allocation)

    total = sum(a.stake for a in kept)
    log.info("portfolio built",
             extra={"candidates": len(candidates), "accepted": len(kept),
                    "total_stake": round(total, 2), "bankroll": bankroll})
    return PortfolioResult(kept, total, bankroll, dropped)


def correlation_warnings(candidates: Sequence[Candidate]) -> list[str]:
    """Flag structurally correlated selections a user should see."""
    warnings: list[str] = []
    by_game: dict[str, list[Candidate]] = defaultdict(list)
    by_team: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_game[candidate.game_uid].append(candidate)
        for team in candidate.teams:
            by_team[team].append(candidate)

    for game_uid, group in by_game.items():
        if len(group) > 1:
            markets = ", ".join(sorted({f"{c.market}/{c.selection}" for c in group}))
            warnings.append(f"{len(group)} correlated selections on {game_uid} ({markets})")
    for team, group in by_team.items():
        if len(group) > 2:
            warnings.append(f"{len(group)} selections involving {team}")
    return warnings
