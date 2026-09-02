"""Soccer match model: Dixon-Coles goals + Elo + a discriminative classifier.

Soccer is modelled natively as a three-outcome sport.  Treating it as
"home vs away" throws away the draw, which is 25% of matches in most leagues
and the single most mispriced selection for naive models.

The primary engine is **Dixon-Coles**: a bivariate Poisson goal model with
team attack and defence strengths, a home-advantage term, a low-score
dependence correction (``rho``) and exponential time decay so recent form
counts for more.  From the fitted goal expectations the entire scoreline
distribution follows, which yields 1X2 *and* totals probabilities from one
coherent model rather than three unrelated ones.

Two extensions matter for real use:

* an L2 penalty shrinks team strengths toward league average, which is exactly
  what a newly promoted club with a handful of matches needs;
* several divisions of one country can be fitted jointly with per-league
  scoring offsets, so a promoted club arrives with a rating earned in the
  division below instead of a blank slate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ..config import settings
from ..logging_setup import get_logger
from .calibration import multiclass_brier, multiclass_log_loss

log = get_logger(__name__)

MODEL_VERSION = "soccer-dc-ens-2.0"
OUTCOMES = ("home", "draw", "away")
MAX_GOALS = 10


# --------------------------------------------------------------------------
# Dixon-Coles
# --------------------------------------------------------------------------

@dataclass
class DixonColesConfig:
    #: Exponential time-decay rate per day.  0.0035 halves a match's weight
    #: after roughly 200 days, the value Dixon & Coles found optimal for
    #: English league data; it is refitted-friendly, not sacred.
    xi: float = 0.0035
    l2: float = 0.02          # shrinkage of team strengths toward average
    # A joint multi-division fit carries ~450 free parameters; the default
    # of 400 iterations stopped short of convergence on the full dataset.
    max_iterations: int = 1500
    min_matches_per_team: int = 3


@dataclass
class DixonColesFit:
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    league_offset: dict[str, float]
    log_likelihood: float
    n_matches: int
    teams: list[str]
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_advantage": round(self.home_advantage, 4),
            "rho": round(self.rho, 4),
            "n_matches": self.n_matches,
            "n_teams": len(self.teams),
            "converged": self.converged,
            "log_likelihood": round(self.log_likelihood, 2),
            "league_offset": {k: round(v, 4) for k, v in self.league_offset.items()},
        }


def _tau(home_goals: np.ndarray, away_goals: np.ndarray, lambda_home: np.ndarray,
         lambda_away: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles dependence correction for 0-0, 1-0, 0-1 and 1-1.

    Independent Poissons systematically under-predict low-scoring draws; this
    term repairs exactly those four cells and leaves the rest untouched.
    """
    tau = np.ones_like(lambda_home, dtype=float)
    mask = (home_goals == 0) & (away_goals == 0)
    tau[mask] = 1.0 - lambda_home[mask] * lambda_away[mask] * rho
    mask = (home_goals == 0) & (away_goals == 1)
    tau[mask] = 1.0 + lambda_home[mask] * rho
    mask = (home_goals == 1) & (away_goals == 0)
    tau[mask] = 1.0 + lambda_away[mask] * rho
    mask = (home_goals == 1) & (away_goals == 1)
    tau[mask] = 1.0 - rho
    return np.clip(tau, 1e-9, None)


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles with time decay and shrinkage."""

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config = config or DixonColesConfig()
        self.fit_result: DixonColesFit | None = None

    # ------------------------------------------------------------------- fit

    def fit(self, matches: pd.DataFrame, *, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        required = {"home_team_uid", "away_team_uid", "home_score", "away_score", "game_date"}
        missing = required - set(matches.columns)
        if missing:
            raise KeyError(f"Dixon-Coles needs columns {sorted(missing)}")

        frame = matches.dropna(subset=["home_score", "away_score"]).copy()
        if frame.empty:
            raise ValueError("no completed matches to fit Dixon-Coles")

        as_of = pd.Timestamp(as_of or frame["game_date"].max())
        age_days = (as_of - pd.to_datetime(frame["game_date"])).dt.days.clip(lower=0)
        weights = np.exp(-self.config.xi * age_days.to_numpy(dtype=float))

        teams = sorted(set(frame["home_team_uid"]) | set(frame["away_team_uid"]))
        team_index = {team: i for i, team in enumerate(teams)}
        leagues = sorted(frame["league_id"].unique()) if "league_id" in frame.columns else ["_"]
        league_index = {league: i for i, league in enumerate(leagues)}

        home_idx = frame["home_team_uid"].map(team_index).to_numpy()
        away_idx = frame["away_team_uid"].map(team_index).to_numpy()
        league_idx = (frame["league_id"].map(league_index).to_numpy()
                      if "league_id" in frame.columns else np.zeros(len(frame), dtype=int))
        home_goals = frame["home_score"].to_numpy(dtype=float)
        away_goals = frame["away_score"].to_numpy(dtype=float)

        n_teams = len(teams)
        n_leagues = len(leagues)
        # [attack (n_teams), defence (n_teams), home_advantage, rho, league offsets]
        start = np.concatenate([
            np.zeros(n_teams), np.zeros(n_teams),
            [0.25, -0.05], np.full(n_leagues, np.log(max(frame[["home_score", "away_score"]]
                                                          .to_numpy().mean(), 0.3))),
        ])

        l2 = self.config.l2

        def unpack(params: np.ndarray):
            attack = params[:n_teams]
            defence = params[n_teams:2 * n_teams]
            home_advantage = params[2 * n_teams]
            rho = params[2 * n_teams + 1]
            offsets = params[2 * n_teams + 2:]
            # Identifiability: attack and defence are only defined up to a
            # common shift, so centre them.
            attack = attack - attack.mean()
            defence = defence - defence.mean()
            return attack, defence, home_advantage, rho, offsets

        def negative_log_likelihood(params: np.ndarray) -> float:
            attack, defence, home_advantage, rho, offsets = unpack(params)
            base = offsets[league_idx]
            lambda_home = np.exp(base + attack[home_idx] - defence[away_idx] + home_advantage)
            lambda_away = np.exp(base + attack[away_idx] - defence[home_idx])
            lambda_home = np.clip(lambda_home, 1e-6, 12.0)
            lambda_away = np.clip(lambda_away, 1e-6, 12.0)

            tau = _tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            log_likelihood = (
                np.log(tau)
                + home_goals * np.log(lambda_home) - lambda_home
                + away_goals * np.log(lambda_away) - lambda_away
            )
            penalty = l2 * (np.sum(attack ** 2) + np.sum(defence ** 2))
            return float(-np.sum(weights * log_likelihood) + penalty)

        bounds = ([(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 1.5), (-0.3, 0.3)]
                  + [(-2.0, 2.0)] * n_leagues)
        result = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.config.max_iterations})

        attack, defence, home_advantage, rho, offsets = unpack(result.x)
        self.fit_result = DixonColesFit(
            attack=dict(zip(teams, attack)),
            defence=dict(zip(teams, defence)),
            home_advantage=float(home_advantage),
            rho=float(rho),
            league_offset={league: float(offsets[i]) for league, i in league_index.items()},
            log_likelihood=float(-result.fun),
            n_matches=len(frame),
            teams=teams,
            converged=bool(result.success),
        )
        if not result.success:
            log.warning("Dixon-Coles did not fully converge",
                        extra={"reason": str(result.message), "matches": len(frame)})
        return self

    # --------------------------------------------------------------- predict

    def expected_goals(self, home_team: str, away_team: str,
                       league_id: str | None = None) -> tuple[float, float]:
        if self.fit_result is None:
            raise RuntimeError("Dixon-Coles model is not fitted")
        fit = self.fit_result
        offsets = fit.league_offset
        base = offsets.get(league_id or "_", float(np.mean(list(offsets.values()))))
        # An unseen club sits at league average: attack and defence of zero.
        attack_home = fit.attack.get(home_team, 0.0)
        attack_away = fit.attack.get(away_team, 0.0)
        defence_home = fit.defence.get(home_team, 0.0)
        defence_away = fit.defence.get(away_team, 0.0)
        lambda_home = float(np.exp(base + attack_home - defence_away + fit.home_advantage))
        lambda_away = float(np.exp(base + attack_away - defence_home))
        return min(lambda_home, 12.0), min(lambda_away, 12.0)

    def score_matrix(self, home_team: str, away_team: str, league_id: str | None = None,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        lambda_home, lambda_away = self.expected_goals(home_team, away_team, league_id)
        home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
        away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
        matrix = np.outer(home_pmf, away_pmf)

        rho = self.fit_result.rho
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
        matrix[0, 1] *= 1.0 + lambda_home * rho
        matrix[1, 0] *= 1.0 + lambda_away * rho
        matrix[1, 1] *= 1.0 - rho
        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    def predict_match(self, home_team: str, away_team: str, league_id: str | None = None
                      ) -> dict[str, Any]:
        matrix = self.score_matrix(home_team, away_team, league_id)
        home_probability = float(np.tril(matrix, -1).sum())
        draw_probability = float(np.trace(matrix))
        away_probability = float(np.triu(matrix, 1).sum())
        lambda_home, lambda_away = self.expected_goals(home_team, away_team, league_id)

        totals = {}
        indices = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        for line in (1.5, 2.5, 3.5):
            over = float(matrix[indices > line].sum())
            totals[f"over_{line}"] = over
            totals[f"under_{line}"] = 1.0 - over

        return {
            "home": home_probability, "draw": draw_probability, "away": away_probability,
            "expected_home_goals": lambda_home, "expected_away_goals": lambda_away,
            "btts_yes": float(matrix[1:, 1:].sum()),
            "totals": totals,
        }

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in frame.iterrows():
            prediction = self.predict_match(
                row["home_team_uid"], row["away_team_uid"], row.get("league_id")
            )
            rows.append([prediction["home"], prediction["draw"], prediction["away"]])
        return np.asarray(rows, dtype=float)

    def team_strengths(self) -> pd.DataFrame:
        if self.fit_result is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "team_uid": self.fit_result.teams,
                "attack": [self.fit_result.attack[t] for t in self.fit_result.teams],
                "defence": [self.fit_result.defence[t] for t in self.fit_result.teams],
            }
        ).sort_values("attack", ascending=False)


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------

SOCCER_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "form": tuple(
        f"diff_{m}_r{w}" for m in ("goals_for", "goals_against", "goal_diff", "points",
                                   "shots_for", "sot_for", "shots_against", "sot_against")
        for w in (5, 10)
    ),
    "form_long": tuple(f"diff_{m}_r20" for m in ("goal_diff", "points", "goals_for",
                                                 "goals_against")),
    "efficiency": ("diff_shot_accuracy_r10", "diff_conversion_r10",
                   "home_attack_vs_defence", "away_attack_vs_defence"),
    "ratings": ("diff_elo", "elo_prob_home", "elo_prob_draw", "elo_prob_away"),
    "schedule": ("diff_rest_days", "diff_congestion", "home_matches_14d", "away_matches_14d"),
    "context": ("home_is_new_to_league", "away_is_new_to_league", "league_strength",
                "league_home_advantage", "league_avg_goals"),
    "h2h": ("h2h_matches", "h2h_home_points", "h2h_goal_diff"),
    "dixon_coles": ("dc_prob_home", "dc_prob_draw", "dc_prob_away",
                    "dc_expected_home_goals", "dc_expected_away_goals"),
}

SOCCER_VARIANTS: dict[str, tuple[str, ...]] = {
    "elo_only": ("ratings",),
    "dixon_coles_only": ("dixon_coles",),
    "form_only": ("form", "form_long"),
    "form_ratings": ("form", "form_long", "ratings"),
    "form_ratings_dc": ("form", "form_long", "ratings", "dixon_coles"),
    "full": tuple(SOCCER_FEATURE_GROUPS.keys()),
}

#: Columns holding the de-vigged market view at decision time.  Using them is
#: not leakage — the prices exist before kick-off — but it does change what the
#: model *is*: a market-aware model measures residual edge over the market
#: rather than independent skill, so both versions are kept separable.
MARKET_COLUMNS = ("market_prob_home", "market_prob_draw", "market_prob_away")


def resolve_soccer_features(variant: str | Sequence[str] = "full",
                            available: Iterable[str] | None = None) -> list[str]:
    groups = SOCCER_VARIANTS.get(variant) if isinstance(variant, str) else tuple(variant)
    if groups is None:
        raise ValueError(f"unknown soccer feature variant '{variant}'")
    columns: list[str] = []
    for group in groups:
        columns.extend(SOCCER_FEATURE_GROUPS[group])
    if available is not None:
        available_set = set(available)
        columns = [c for c in columns if c in available_set]
    return list(dict.fromkeys(columns))


class TemperatureCalibrator:
    """Multiclass calibration by a single temperature parameter.

    Deliberately minimal: with a few thousand matches a richer calibrator
    (Dirichlet, vector scaling) has enough parameters to fit noise, and a
    badly calibrated calibrator is worse than none.
    """

    def __init__(self) -> None:
        self.temperature = 1.0
        self.fitted = False

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> "TemperatureCalibrator":
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
        outcomes = np.asarray(outcomes, dtype=int)
        if len(outcomes) < 100:
            self.fitted = True
            return self

        def objective(log_temperature: np.ndarray) -> float:
            temperature = float(np.exp(log_temperature[0]))
            scaled = probabilities ** (1.0 / temperature)
            scaled = scaled / scaled.sum(axis=1, keepdims=True)
            return multiclass_log_loss(outcomes, scaled)

        result = minimize(objective, np.array([0.0]), method="Nelder-Mead",
                          options={"maxiter": 120, "xatol": 1e-4, "fatol": 1e-6})
        self.temperature = float(np.exp(result.x[0])) if result.success else 1.0
        self.fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
        if not self.fitted or abs(self.temperature - 1.0) < 1e-6:
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        scaled = probabilities ** (1.0 / self.temperature)
        return scaled / scaled.sum(axis=1, keepdims=True)


@dataclass
class SoccerMatchModel:
    """Dixon-Coles + Elo + gradient-boosted classifier, blended and calibrated."""

    features: list[str] = field(default_factory=list)
    dc_config: DixonColesConfig = field(default_factory=DixonColesConfig)
    model_version: str = MODEL_VERSION
    feature_set_version: str = "soccer_v2.0"
    variant: str = "full"
    use_classifier: bool = True
    #: Include the de-vigged market as an ensemble component.  When enabled the
    #: model's probabilities are anchored to the market and any reported edge is
    #: *residual* edge; when disabled the model is fully independent of prices.
    use_market: bool = False

    dixon_coles: DixonColesModel | None = None
    classifier: Any = None
    weights: np.ndarray | None = None
    calibrator: TemperatureCalibrator | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------- fit

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, *,
            raw_matches: pd.DataFrame | None = None) -> "SoccerMatchModel":
        """Fit on ``train`` and learn blend weights/calibration on ``valid``."""
        history = raw_matches if raw_matches is not None else train
        fit_source = history[history["game_date"] <= train["game_date"].max()]
        self.dixon_coles = DixonColesModel(self.dc_config).fit(
            fit_source.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
            if "home_score" not in fit_source.columns else fit_source,
            as_of=train["game_date"].max(),
        )

        train_augmented = self.attach_dc_features(train)
        valid_augmented = self.attach_dc_features(valid)

        if self.use_classifier and self.features:
            from xgboost import XGBClassifier  # local import keeps module import cheap

            usable = [f for f in self.features if f in train_augmented.columns]
            self.features = usable
            X_train = train_augmented[usable].astype(float).to_numpy()
            y_train = train_augmented["outcome"].astype(int).to_numpy()
            self.classifier = XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.85,
                colsample_bytree=0.7, min_child_weight=15, reg_lambda=3.0,
                objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                tree_method="hist", random_state=settings.model.random_seed,
            )
            self.classifier.fit(X_train, y_train, verbose=False)

        component_names, components_valid = self._component_matrices(valid_augmented)
        y_valid = valid_augmented["outcome"].astype(int).to_numpy()
        self.weights = self._fit_weights(components_valid, y_valid)
        blended_valid = self._blend(components_valid)

        self.calibrator = TemperatureCalibrator().fit(blended_valid, y_valid)
        calibrated_valid = self.calibrator.transform(blended_valid)
        self.metrics = {
            "validation": {
                "log_loss": multiclass_log_loss(y_valid, calibrated_valid),
                "brier": multiclass_brier(y_valid, calibrated_valid),
                "accuracy": float(np.mean(calibrated_valid.argmax(axis=1) == y_valid)),
                "n": int(len(y_valid)),
            },
            "components_validation": {
                name: {
                    "log_loss": multiclass_log_loss(y_valid, matrix),
                    "brier": multiclass_brier(y_valid, matrix),
                }
                for name, matrix in zip(component_names, components_valid)
            },
            "blend_weights": dict(zip(component_names, [float(w) for w in self.weights])),
            "temperature": self.calibrator.temperature,
            "dixon_coles": self.dixon_coles.fit_result.to_dict(),
            "n_train": int(len(train)),
        }
        log.info("fitted soccer model", extra={"weights": self.metrics["blend_weights"],
                                               "valid_logloss": self.metrics["validation"]["log_loss"]})
        return self

    # --------------------------------------------------------------- predict

    def attach_dc_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add Dixon-Coles outputs as features for the discriminative model."""
        if self.dixon_coles is None:
            return frame
        augmented = frame.copy()
        rows = [
            self.dixon_coles.predict_match(row["home_team_uid"], row["away_team_uid"],
                                           row.get("league_id"))
            for _, row in frame.iterrows()
        ]
        augmented["dc_prob_home"] = [r["home"] for r in rows]
        augmented["dc_prob_draw"] = [r["draw"] for r in rows]
        augmented["dc_prob_away"] = [r["away"] for r in rows]
        augmented["dc_expected_home_goals"] = [r["expected_home_goals"] for r in rows]
        augmented["dc_expected_away_goals"] = [r["expected_away_goals"] for r in rows]
        return augmented

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        blended = self._blend(self._components(frame))
        return (self.calibrator.transform(blended) if self.calibrator is not None
                else blended)

    def predict_detail(self, frame: pd.DataFrame) -> dict[str, Any]:
        names, components = self._component_matrices(self.attach_dc_features(frame))
        blended = self._blend(components)
        calibrated = (self.calibrator.transform(blended) if self.calibrator else blended)
        spread = np.max(np.stack([c[:, 0] for c in components]), axis=0) - \
            np.min(np.stack([c[:, 0] for c in components]), axis=0)
        return {
            "probabilities": calibrated,
            "components": {name: matrix for name, matrix in zip(names, components)},
            "agreement": np.clip(1.0 - spread / 0.25, 0.0, 1.0),
        }

    def _components(self, frame: pd.DataFrame) -> list[np.ndarray]:
        return self._component_matrices(self.attach_dc_features(frame))[1]

    def _component_matrices(self, augmented: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
        """All ensemble components for an already DC-augmented frame."""
        names = ["dixon_coles", "elo"]
        components = [
            augmented[["dc_prob_home", "dc_prob_draw", "dc_prob_away"]].to_numpy(dtype=float)
            if "dc_prob_home" in augmented.columns
            else self.dixon_coles.predict_frame(augmented),
            _elo_matrix(augmented),
        ]
        if self.classifier is not None and self.features:
            matrix = augmented[self.features].astype(float).to_numpy()
            components.append(self.classifier.predict_proba(matrix))
            names.append("classifier")
        if self.use_market and all(c in augmented.columns for c in MARKET_COLUMNS):
            market = augmented[list(MARKET_COLUMNS)].astype(float).to_numpy()
            if not np.isnan(market).all():
                market = np.where(np.isnan(market), 1.0 / 3.0, market)
                market = np.clip(market, 1e-6, None)
                components.append(market / market.sum(axis=1, keepdims=True))
                names.append("market")
        return names, components

    def _blend(self, components: list[np.ndarray]) -> np.ndarray:
        weights = self.weights if self.weights is not None else np.full(
            len(components), 1.0 / len(components)
        )
        blended = sum(w * c for w, c in zip(weights, components))
        return blended / blended.sum(axis=1, keepdims=True)

    @staticmethod
    def _fit_weights(components: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        stacked = np.stack(components)  # (n_components, n_rows, 3)

        def objective(weights: np.ndarray) -> float:
            blended = np.tensordot(weights, stacked, axes=(0, 0))
            blended = np.clip(blended, 1e-9, None)
            blended = blended / blended.sum(axis=1, keepdims=True)
            return multiclass_log_loss(y_true, blended)

        n = len(components)
        start = np.full(n, 1.0 / n)
        result = minimize(objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                          constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                          options={"maxiter": 200, "ftol": 1e-9})
        weights = np.clip(result.x if result.success else start, 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else start


def _elo_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = ("elo_prob_home", "elo_prob_draw", "elo_prob_away")
    if not all(c in frame.columns for c in columns):
        return np.full((len(frame), 3), 1.0 / 3.0)
    matrix = frame[list(columns)].astype(float).fillna(1 / 3).to_numpy()
    matrix = np.clip(matrix, 1e-6, None)
    return matrix / matrix.sum(axis=1, keepdims=True)
