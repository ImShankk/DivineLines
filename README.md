# DivineLines

Quantitative sports research, market analysis and risk management for **NBA** and **soccer**.

DivineLines estimates outcome probabilities from historical data, compares them against
de-vigged bookmaker prices, sizes positions under portfolio risk limits, and records every
prediction so that its own accuracy can be measured later.

**It does not claim to be profitable.** The evidence below is mixed and is reported as
found: the soccer model loses to the market, the NBA model does not beat the market either
but its selections carry a small measurable signal, and its +4.8% simulated ROI is *not*
statistically significant. Reporting "no edge" honestly is the feature; a dashboard that
always finds a bet is the failure mode this was designed to avoid.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Data sources](#data-sources)
- [Database](#database)
- [Methodology](#methodology)
- [CLV methodology](#clv-methodology)
- [Lineups](#lineups)
- [Match Centre](#match-centre)
- [Model health](#model-health)
- [Results](#results-measured-not-claimed)
- [Betting engine](#betting-engine)
- [CLI](#cli)
- [API](#api)
- [Frontend](#frontend)
- [Configuration](#configuration)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Where the v1 code went](#where-the-v1-code-went)

---

## What it does

```
   SOURCES          INGESTION            MODELLING              DECISION
┌────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
│ stats.nba  │   │   fetch      │   │  leak-free       │   │  no-vig market │
│ ESPN       │──▶│   validate   │──▶│  features        │──▶│  comparison    │
│ The-Odds   │   │   normalise  │   │  Elo / adjusted  │   │  EV + edge     │
│ football-  │   │   store      │   │  XGB / logistic  │   │  score         │
│ data.co.uk │   │   + provenance   │  Dixon-Coles     │   │  fractional    │
└────────────┘   └──────────────┘   │  calibration     │   │  Kelly         │
                                     └──────────────────┘   │  portfolio caps│
                                                            └───────┬────────┘
                                            ┌───────────────────────▼────────┐
                                            │ prediction ledger → settlement │
                                            │ → CLV, ROI, calibration        │
                                            │ → back into model selection    │
                                            └────────────────────────────────┘
```

- **NBA**: win probability from an ensemble (XGBoost + logistic + Elo), calibrated, with a
  player-level availability adjustment applied at prediction time.
- **Soccer**: native 1X2 modelling via **Dixon-Coles** bivariate Poisson — plus totals from
  the same fitted scoreline distribution — across 8 competitions and 9 seasons.
- **Market**: multi-bookmaker snapshots, margin removal, consensus and best price, line
  movement, closing-line value.
- **Risk**: fractional Kelly shrunk by model confidence, with per-game, per-team, per-sport
  and slate exposure caps.
- **Evidence**: walk-forward backtesting, calibration curves, feature ablation, a model
  registry, and a prediction ledger that grades itself once games finish.
- **CLV**: a first-class ledger — entry price, closing price, and a decomposition that
  separates line shopping from market drift from actual model skill.
- **Lineups**: timestamped observations that can never leak backwards, prediction
  versioning across lineup stages, and an event timeline with change attribution.
- **Model health**: predictive health and betting health tracked separately, by sport,
  market and time window, with evidence-based statuses and regression detection.

---

## Quick start

```bash
pip install -r requirements.txt      # or: pip install -e .
cp .env.example .env                 # add ODDS_API_KEY for live prices (optional)

python -m divinelines.cli migrate    # import the legacy v1 database
python -m divinelines.cli refresh    # results, fixtures, injuries, odds
python -m divinelines.cli train      # fit + register models
python -m divinelines.cli scan --sport soccer --days 7
python -m divinelines.cli serve      # API on http://127.0.0.1:8000
```

```bash
cd frontend && npm install && npm run dev    # dashboard on http://localhost:5173
```

Installed with `pip install -e .`, every command is also available as `divinelines <command>`.
`python main.py` still opens an interactive menu.

Everything except live odds works without an API key.

---

## Architecture

```
divinelines/
  config.py            all tunables; seasons derive from the clock, never hard-coded
  identity.py          canonical team/club resolution — the only place names are matched
  logging_setup.py     structured logging (text or JSON lines)
  db/
    schema.sql         canonical schema, v2
    migrations.py      versioned migrations (v3-v8), safe on a populated database
    connection.py      WAL, serialised writes, one PRAGMA policy
    repository.py      typed persistence; stamps provenance on every row
    validation.py      critical/warning checks; failures are recorded, not swallowed
    migrate.py         v1 import, including the duplicate-id and partial-game repairs
  sources/
    base.py            retries, backoff, rate limiting, caching, last-success tracking
    nba_stats.py       stats.nba.com box scores + advanced player stats
    espn_nba.py        injuries, schedule, rosters
    espn_odds.py       historical NBA opening/closing prices (14 books)
    espn_lineups.py    timestamped lineup observations, soccer-first
    espn_match.py      match detail: events, box scores, player lines, context
    odds_api.py        multi-book prices, quota-aware
    football_data.py   soccer results, match stats and historical closing odds
  features/
    ratings.py         Elo (MOV, season regression) + online opponent-adjusted efficiency
    nba_features.py    chronological, season-aware, leak-free by construction
    soccer_features.py same, with promotion/relegation and congestion handling
  analytics/
    clv_analysis.py    CLV cohorts, distribution, sample-size-aware inference
    clv_skill.py       splits CLV into shopping / market drift / model selection
    model_health.py    predictive vs betting health, statuses, regression detection
    timeline.py        event timelines and prediction-change attribution
    lineup_experiment.py  the oracle-bound lineup ablation
  models/
    calibration.py     Brier / log loss / ECE / reliability; isotonic + Platt
    nba_model.py       ensemble, blend weights fitted on a held-out block
    soccer_model.py    Dixon-Coles + Elo + classifier, temperature-calibrated
    nba_player_impact.py  per-player margin impact and scenario-weighted availability
    registry.py        model + experiment records; reproducibility metadata
    experiments.py     feature ablation
  betting/
    closing_line.py    the formal closing-line policy — one answer to "what closed?"
    settlement.py      idempotent, incremental settlement into the CLV ledger
    odds_math.py       conversions, margin removal (multiplicative + power), consensus
    ev.py              expected value and the transparent edge-quality score
    kelly.py           fractional Kelly with uncertainty shrinkage and caps
    portfolio.py       per-game / per-team / per-sport / slate exposure limits
    clv.py             closing line value
    ledger.py          prediction history, paper trading, settlement, performance
  backtest/
    walkforward.py     rolling refits + market simulation
    metrics.py         ROI, drawdown, buckets, probability quality
    nba_backtest.py    probability-quality evaluation (no odds history exists)
    soccer_backtest.py full simulation against real historical prices
  pipeline/
    backfill_odds.py   historical price backfill + ESPN id mapping
    ingest_lineups.py  lineup observations and the information-event stream
    replay.py          replays seasons into the ledger at real opening prices
    refresh.py         orchestration; a failing stage never blocks the others
    ingest_soccer.py   football-data ingestion
    train.py           training + registration
    predict.py         game-day chain: features → model → market → EV → risk → ledger
  api/                 FastAPI app, schemas, and the v1-compatible endpoint
  cli.py               operator interface
frontend/src/          React + TypeScript dashboard (no chart library; hand-rolled SVG)
tests/                 332 tests, fully offline (incl. stored source-contract fixtures)
```

**Design rules that shaped this:**

1. *Feature builders are single chronological passes.* Each row is emitted from state that
   existed before that game, then the result is applied. Leakage becomes structurally
   impossible instead of something you have to remember to prevent with `.shift(1)`.
2. *Identity resolution lives in one module.* Nothing else compares team names.
3. *Every stored fact carries `source` and `retrieved_at`.*
4. *A source that fails raises or returns stale-flagged cache — it never invents data.*

---

## Data sources

| Source | Provides | Notes |
|---|---|---|
| `data/processed/nba_data.db` (v1) | 5,984 NBA games, 2021-22 → 2025-26 | Read-only; imported by `migrate` |
| stats.nba.com (`nba_api`) | Box scores, advanced player stats | Season derived from the clock |
| ESPN site API | NBA injuries, schedule, rosters | Current state only — no history |
| The-Odds-API | Live multi-book prices | Free tier: current odds only; call budget enforced |
| ESPN core API | **Historical NBA opening and closing moneylines**, 14 books | Closes V2's biggest gap; in-play feeds excluded |
| ESPN summary API | Soccer/NBA lineups: formation, XI, positions | Publishes *what*, never *when* — see [Lineups](#lineups) |
| football-data.co.uk | Soccer results, shots, cards, referee, **and closing odds** | 26,500+ matches; the reason soccer can be backtested honestly |

**Ingested today:** 26,544 soccer matches across 8 competitions (2017-18 → 2026-27),
6,025 NBA games, **646,000+ price snapshots** (including 63,554 historical NBA open/close
quotes across 3,482 games), and 760 Premier League lineups (30,377 player rows).

Sources are polite by construction: bounded retries with exponential backoff, a minimum
interval between requests to a host, on-disk caching with per-dataset TTLs, and a hard cap
on odds-API calls per run.

---

## Database

SQLite, WAL journalling, process-wide write serialisation. Core tables:

`leagues`, `teams`, `players`, `games`, `nba_team_game`, `soccer_match_stats`,
`player_status`, `transactions`, `odds_snapshots`, `models`, `experiments`,
`predictions`, `bets`, `source_status`, `validation_issues`.

Identity is canonical and stable: `nba:LAL`, `soccer:manchester-united`. Club ids are
**league-independent**, so a club keeps its identity through promotion and relegation.

### Three real corruptions found in stored data

Both are fixed by `divinelines migrate`, and both are covered by regression tests.

1. **The 2025-26 season was stored twice.** Game ids appeared both zero-padded
   (`0022500002`) and unpadded (`22500002`). `PRIMARY KEY (TEAM_ID, GAME_ID)` treated
   them as different games, so 1,036 games were counted twice — every rolling-10 window
   silently contained each of those games twice. After normalising ids, each completed
   season contains exactly **1,230 games**, which is the correct NBA regular-season count.
2. **Games captured mid-play were stored as final.** The last rows had no `WL` and `MIN` of
   0 — a team "losing 18-25". Only games with a decided result and a full complement of
   minutes are now accepted; the rest are kept as fixtures.
3. **A soccer feed served the wrong division, and the platform believed the URL.**
   *(Found during the V5 audit.)* At the start of the 2026-27 season football-data.co.uk
   had not yet published `E0.csv`; the request for it returned a file whose every row read
   `Div=EC` — the National League. The same happened for `SP1.csv`, which came back
   carrying the Portuguese Primeira Liga. The ingest labelled rows by the league it had
   asked for rather than the one the file declared, so **21 fixtures entered the canonical
   store under the wrong competition** — Altrincham v Southend as a Premier League match,
   Porto v Alverca as La Liga. They carried 417 price rows with them.

   The adapter now validates the `Div` column against the requested division and refuses
   the file outright when it disagrees (older files that omit `Div` are still accepted —
   refusing them would delete a decade of history to fix a 2026 problem). The 21 rows were
   removed, the removal is recorded in `validation_issues` as `LEAGUE_MISMATCH`, and the
   deleted rows are archived in `data/artifacts/removed_mislabelled_2627_fixtures.json`.

   This is the class of failure the platform exists to catch: nothing errored, nothing was
   stale, and the only symptom was a Match Centre offering to show you a Premier League
   fixture between two National League clubs.

---

## Methodology

### Leak-free features

Both builders walk games in time order carrying per-team state. Rolling windows (5/10/20),
EWMA, rest days, back-to-backs, travel distance (real arena coordinates), time-zone shifts,
and head-to-head history are all emitted **before** the current game is applied.

Season boundaries are respected: current-season form resets, and Elo regresses toward the
mean by a configurable factor, because an October roster is not the previous April's.

**Bayesian shrinkage** blends current-season form with the prior season using
`w = n / (n + prior_games)` — the standard conjugate-normal weight, with `prior_games`
configurable rather than an arbitrary constant.

Absence is represented as absence. The v1 pipeline hard-coded `H2H_WIN_PCT = 0.50` for
every game; here, no prior meeting means `NaN`, which XGBoost handles natively and which
cannot be mistaken for evidence.

### NBA model

Ensemble of XGBoost, regularised logistic regression, and Elo. Blend weights are fitted by
minimising log loss on a validation block the components never trained on; the blended
probability is then isotonic-calibrated on that same held-out block, and evaluated on a
later one.

Opponent-adjusted efficiency is fitted **online** (SGD on `observed_ortg ≈ league_mean +
offense[team] + defence[opponent]`), so the rating available before a game reflects only
earlier games — a guarantee a batch season-long regression cannot make.

### Soccer model

**Dixon-Coles**: a bivariate Poisson goal model with team attack/defence strengths, home
advantage, the low-score dependence correction (`rho`), and exponential time decay. Two
extensions matter in practice:

- an **L2 penalty** shrinks team strengths toward league average — exactly what a newly
  promoted club with a handful of matches needs;
- **divisions are fitted jointly** with per-league scoring offsets, so a club promoted from
  the Championship arrives with a rating earned in the division below rather than a blank slate.

The fitted scoreline distribution yields 1X2 *and* over/under from one coherent model. The
implementation is verified against synthetic data generated from known parameters: it
recovers home advantage to ±0.003 and team strengths at r > 0.99.

### Player impact and availability (NBA)

Per-player margin impact = `(PIE − replacement) × (minutes / 48) × scale`. Missing minutes
are converted to a change in expected margin, then to a change in win probability through a
coefficient **fitted on this platform's own games**.

Uncertain availability is not guessed. A questionable player produces a probability-weighted
outcome across scenarios, plus an uncertainty measure that shrinks the Kelly stake:

```
P(win) = P(plays) · P(win | plays) + P(out) · P(win | out)
```

Two honest caveats, stated wherever these numbers appear in the UI: the margin→probability
conversion is fitted, but the PIE→margin scale is a **documented prior**, because no
historical injury data exists here to fit it against. And since ESPN publishes no injury
history, **injury features are never trained into the model** — they are a transparent
post-model adjustment.

---

## CLV methodology

Closing line value asks one question: **was the price we took better than the price the
market settled on?** It converges far faster than profit, which is why it is tracked as a
first-class dataset rather than a column.

### The convention

```
clv_price_pct = (entry_odds / closing_odds - 1) x 100
```

Positive means the price we took was *longer* than the close — the market moved toward our
side after we priced it. Some references write this as `closing / entry - 1`, which yields
the same magnitude with the opposite sign and would make a good outcome read as negative.
Practitioners mean the first form by "positive CLV", so that is what the platform uses,
defined in exactly one place (`betting/clv.py`) and imported by the CLI, database,
API, backtester and frontend. A test asserts all entry points agree.

Alongside the price form, a **vig-aware probability form** is reported: entry implied
probability against the *no-vig* closing probability. A close of 2.00 in a 108% market is
not the same claim as 2.00 in a 102% market.

### What counts as the close

Defined in `betting/closing_line.py`, never inferred ad hoc:

1. **Cutoff** — only observations at or before `event_start - cutoff_seconds` qualify.
   Anything at or after kick-off is in-play information.
2. **A fixture that has not started has no close.** The settlement engine refuses to
   resolve one. (Without this it happily called the current price of a fixture three days
   away "the close", which produced a spurious +7.4% CLV on the first run.)
3. **Declared beats inferred.** ESPN and football-data both publish an explicit closing
   price; that is preferred over "the last snapshot we happened to poll", which measures
   our cron schedule rather than the market.
4. **Aggregation is explicit** — `median` (default), `best`, `consensus`, or a named book —
   and is recorded on every CLV record, so a best-price close is never silently compared
   against a consensus one.

### CLV is not profit

The two are reported separately everywhere — CLI, API and dashboard. A bet can lose while
beating the close and win while losing to it. Nothing in the platform presents CLV as
evidence of profitability.

### Decomposition: is the CLV even ours?

The headline CLV number mixes three unrelated effects, so `analytics/clv_skill.py` splits
them:

| Component | What it is |
|---|---|
| **Line shopping** | Gain from taking the best of ~14 books. Real money, no model involved. |
| **Market drift** | Consensus movement from open to close, which applies to every selection. |
| **Model selection** | Do the selections the model *recommends* beat the close by more than the ones it *rejects*? |

Only the third measures the model. It is reported as a difference between groups with a
Welch confidence interval, and a both-sides control (averaging both selections of the same
game) confirms the measurement basis is unbiased.

### Settlement

`divinelines settle` is idempotent (unique key on `prediction_id` plus explicit state
transitions), incremental by default (`--full` forces reconciliation), and supports
`--dry-run`, `--sport`, `--date` and the closing-line policy flags. States are explicit:
`PENDING`, `CLOSE_FOUND`, `NO_CLOSE_AVAILABLE`, `SETTLED`, `INVALID` — never an ambiguous
NULL.

---

## Lineups

### The timestamp problem, stated plainly

ESPN publishes *what* a lineup is. It does not publish *when* that lineup became public.
So every observation is stamped with `observed_at` — when **this platform** saw it — and
that is the only timestamp anything filters on.

The consequence is deliberate: for a match played before the platform started polling, the
actual XI is retrievable but its publication time is not. Those rows are stored as
`lineup_state='final'` and are **barred from live prediction** by `lineup_state_at()`,
which defaults to `allow_final=False`. They exist for research only.

`test_a_lineup_observed_later_is_invisible_earlier` asserts that an XI observed at 18:15
cannot appear in a prediction made at 16:00.

### Prediction stages and versioning

A prediction is never overwritten. A new one **supersedes** the old, so both survive:

```
scheduled -> pre_lineup -> projected_lineup -> confirmed_lineup
```

Every prediction records its stage, lineup state, feature version, event start,
seconds-to-event and an information snapshot. Scoring excludes superseded rows so one
fixture is never counted twice.

### Features

No external player-rating feed exists for soccer, so "lineup strength" is not a sum of
player ratings — that would be a number invented from data the platform does not have.
What is derivable from lineup history itself:

- share of the XI who are the team's regular starters,
- how many regulars are missing (position-weighted; a goalkeeper is not one eleventh of a
  team),
- whether the usual goalkeeper is playing,
- selection continuity with recent XIs.

All computed in a single chronological pass: regulars at match *n* come from matches
1..n-1 only.

---

## Match Centre

The soccer side of the platform is not only a probability engine. V5 unpacks the part of
ESPN's summary document that V3 never read — the play-by-play — and builds a match view on
top of it.

### What the feed actually contains

I audited every cached payload before writing a line of visualisation code, because the
difference between "we have match events" and "we have match data" decides which panels
can exist at all.

| Available | Count across 760 Premier League matches |
|---|---|
| Play-by-play events (shots, goals, corners, fouls, offsides, cards, subs, VAR) | 56,580 |
| …of which carry a **field position** | 43,361 |
| Shots with a location (the basis of the shot map) | 18,281 |
| Team box-score metrics (possession, passes, tackles, crosses, long balls) | 28 per team |
| Per-player box-score metrics | 15 per player |
| Venue, attendance, officials, formations, team form | yes |

| Not available | Consequence |
|---|---|
| Pass events (passer, recipient, coordinates) | **No passing network can be drawn.** |
| Player tracking | Heatmaps are event-location density, not player positions. |
| Expected goals | No xG is shown, and no proxy is labelled as one. |
| Player ratings | No rating column, and none is invented. |

The passing network therefore ships as a real component with a real endpoint that returns
`NO_DATA` and the reason, next to the aggregate pass counts that *do* exist. An empty pitch
with an explanation is a more useful screen than a plausible-looking network built out of
nothing, and it is the only honest one.

### Coordinates

ESPN gives ball events an `X`/`Y` pair in 0–1, measured **from the goal the acting team is
attacking**. I worked that out from the data rather than from documentation, because there
is none: shots have a median X of 0.24 and corners 0.09 (both near the attacked goal) while
fouls sit at 0.66, which is where a defending side commits them.

Because the frame is relative to whoever acted, two shots at opposite ends have the same X.
Drawing them raw puts both teams' attacks in the same half. So:

- the **raw** source values are stored untouched in `match_events.source_x/source_y`;
- normalisation into a 105 × 68 metre pitch frame (home attacking right) happens once, in
  `matchcenter/spatial.py`, on the way out.

`0.0, 0.0` means "no position", not "the corner flag" — ESPN sends it for cards and
substitutions rather than omitting the field, and treating it as real piles phantom events
into one corner of the pitch.

### Momentum

A **descriptive** statistic, not a prediction and deliberately not a model feature:

```
momentum(t) = Σ  weight(event) · 0.5 ^ ((t - t_event) / 8 minutes)
             events with t_event ≤ t
```

Weights are a stated prior, not a fit — a goal 10, a shot on target 4, a blocked shot 1.5,
a corner 1, a card negative. There is no target to fit them against, and inventing one
would be the dishonest move. Every parameter is versioned (`momentum/v1`) and travels in
the payload, so two curves computed months apart can be compared or told apart.

A visually convincing curve is exactly the kind of thing that talks its way into a champion
model. If momentum is ever to become a feature it goes the same route as any other
candidate: chronological experiment first, promotion only on evidence.

Swings name an **associated event**, never a cause. The curve moved because that event
entered a weighted sum; whether it changed the match is not something an event feed can
establish.

### Replay, and the two clocks

A bounded Match Centre view is subject to two independent cut-offs, and conflating them is
a bug I actually shipped and then fixed:

| Bound | Applies to | Meaning |
|---|---|---|
| `minute` | the event stream, on the event's own match clock | what had **happened** by then |
| `as_of` | prices, predictions, lineups, and `observed_at` on events | what we had **seen** by then |

Replaying to minute 32 of a match that kicked off at 15:00 derives an information cut-off
of 15:32 for the market and the model, while bounding events on their match clock.
Applying the observation filter to events as well produced an empty match at every replay
position — a backfilled match has an `observed_at` in August no matter which minute you
ask for.

Two more things the replay does rather than hide:

- **Full-time box statistics are withheld at a replay position.** Possession and passing
  are only published as match totals, so showing 58% possession next to a 32nd-minute score
  would be wrong in the most convincing way possible. What can be recounted from events
  (shots, corners, cards, offsides) is recounted; the rest is listed as withheld.
- **Retrospectively ingested matches say so.** For a match read after full time, a replay
  reconstructs what *happened* by a minute — not what the platform *knew* at that minute.

All of this is enforced in the service and asserted against the API. A frontend that merely
hides future events is one refresh away from showing them.

### What is on the page

**Scoreboard** rendering from a normalised match state (`SCHEDULED` / `LIVE_FIRST_HALF` /
`HALFTIME` / `LIVE_SECOND_HALF` / `EXTRA_TIME` / `PENALTIES` / `FINISHED` / `POSTPONED` /
`CANCELLED`) — a fixture existing in the database never makes it read "LIVE" · **momentum**
with goal, card and substitution markers · **event timeline** with a shared selection ·
**statistics** as two-sided comparisons · **shot map** on a real pitch · **event-location
density** · **passing network** (`NO_DATA`, with the reason) · **lineups** with the
formation, and whether the XI was known before the prediction · **player lines**, position
aware, with no invented rating · **market** with best price and consensus kept separate ·
**model vs market** in probability points, with the staking line the risk engine actually
produced · **league table** as it stood before the fixture · **match intelligence**
component by component · **written report**, generated from the same payload as the panels
so the two cannot drift.

### Match intelligence, not a score

A single "data quality: 87%" would hide exactly what a reader needs. Each component reports
its own state with the count behind it:

```
[ok  ] Match events         62 events from the play-by-play feed
[ok  ] Event coordinates    46 of 62 events carry a field position
[ok  ] Team statistics      56 box-score values
[ok  ] Lineups              22 starters recorded, state 'final'
[ok  ] Market prices        18 price snapshots
[--  ] Model predictions    no prediction was recorded for this fixture
[--  ] Passing network      no source publishes pass events with coordinates
[--  ] Player tracking      no tracking provider is configured
[--  ] Expected goals       no source publishes xG and none is estimated
```

The three permanently-absent rows are informational and do not affect the headline grade —
marking every match down for a platform-wide gap would make the badge meaningless.

### Coverage

Event data is ingested for the **Premier League only** (2024-25 and 2025-26, 760 matches).
The adapter works for all eight configured competitions; ESPN event ids have simply not
been backfilled for the others yet. The match list shows event, lineup, price and
prediction counts per fixture so this is visible before the click, not after it.

---

## Model health

`analytics/model_health.py` separates two questions V2 conflated:

- **Predictive health** — Brier, log loss, calibration, skill over the base rate, and skill
  against the market. Needs only graded predictions.
- **Betting health** — ROI (with a confidence interval), CLV, drawdown. Needs prices and
  settled stakes.

A model can be predictively excellent and commercially useless because the market is simply
better priced. That is exactly what both sports show, so a single blended "model score"
would hide the finding.

### Statuses and their thresholds

| Status | Requires |
|---|---|
| `INSUFFICIENT_SAMPLE` | fewer than 100 graded predictions |
| `DEGRADED` | negative Brier skill — worse than predicting the base rate |
| `UNPROVEN` | Brier skill below 0.02 |
| `VALIDATED_FOR_PREDICTION` | Brier skill ≥ 0.02 |
| `MARKET_BEATING` | log loss better than the no-vig market by ≥ 0.005 over ≥ 200 priced predictions |

Every status carries the sentence that justifies it. `MARKET_BEATING` is deliberately hard
to earn; neither sport currently earns it.

Also tracked: **prediction stability** (how much a fixture's probability swings between
versions), **regression detection** (recent vs lifetime log loss), and a
champion/candidate/retired lifecycle table.


---

## Results (measured, not claimed)

### NBA — now measured against real prices

V2 could report no NBA betting performance at all because no historical NBA odds existed.
ESPN's core API supplies opening **and** closing moneylines, so `divinelines replay`
re-runs the walk-forward predictions into the ledger at genuine opening prices and settles
them against genuine closes. **7,034 predictions across 3,517 games, three seasons:**

| | Value |
|---|---|
| Model log loss | 0.6164 |
| **No-vig market log loss** | **0.5959** |
| Brier / ECE | 0.2139 / 0.0275 |
| Brier skill vs base rate | **+14.4%** |
| Simulated ROI (opening prices, line shopping) | +4.79% |
| **ROI 95% CI** | **[−3.4%, +7.7%] — not significant** |
| Mean CLV | +2.90% (beat-close rate 54.3%) |

**The NBA model does not beat the market** (0.6164 vs 0.5959), and the +4.79% ROI is
consistent with break-even once variance is accounted for. Entry at the opening price is
also the most favourable honest framing available, since opening lines are softer and carry
lower limits.

#### But the CLV decomposition finds a real signal

| Component | Value |
|---|---|
| Reported mean CLV | +2.90% |
| — line shopping (best of 14 books) | +2.64% |
| — market drift (both-sides control) | +0.30% |
| **— model selection effect** | **+1.98pp, 95% CI [1.24, 2.73]** |

Selections the model recommends beat the close by **1.98 percentage points more** than the
ones it rejects, measured consensus-to-consensus so no line shopping is involved. The
same-book comparison agrees (+2.26pp, CI [1.38, 3.13]). A zero-skill model betting both
sides would show zero here; the both-sides control confirms the basis is unbiased at
+0.30%.

That is a genuine, if modest, finding: **the model's choice of side carries information
about where the market is going, even though its probabilities are worse than the
market's.**

### NBA — probability quality by component

Same walk-forward evaluation, **3,519 games**:

| Model | Log loss | Brier | Accuracy | ECE | Brier skill |
|---|---|---|---|---|---|
| **Ensemble** | **0.6203** | 0.2145 | 65.2% | 0.030 | **+13.5%** |
| XGBoost alone | 0.6204 | 0.2154 | 65.9% | 0.029 | +13.2% |
| Logistic alone | 0.6188 | 0.2150 | 65.3% | 0.035 | +13.3% |
| Elo alone | 0.6185 | 0.2149 | 64.9% | 0.051 | +13.4% |
| Always the base rate | 0.6892 | 0.2480 | 54.4% | 0.000 | 0.0% |

(V2 reported no ROI here at all, correctly, because no prices existed. A synthetic market
was built during V2, produced a fake 10% ROI, and was deleted — it was priced off the
model's own Elo component and so was beatable by construction. The figures above use real
bookmaker prices instead.)

### Soccer — full simulation against real historical prices

8 competitions, 4 walk-forward seasons, **11,738 matches**, bets struck at the pre-match
price with the closing line used only for CLV:

| | Log loss | Brier |
|---|---|---|
| Model | 0.9881 | 0.5917 |
| **No-vig market** | **0.9712** | **0.5783** |

**The model does not beat the market.** Simulated betting returned **−5.7% ROI over 5,269
bets**. When the market is added as an ensemble component, the weight optimiser assigns it
**100% of the weight** — the model contributes nothing measurable over the closing line.

Over/under 2.5 goals from the same Dixon-Coles fit fared **worse**, not better: **−11.5% ROI
over 5,901 bets** across all eight competitions. (A two-league, two-season subset had come in
at −1.9%, which is a useful reminder that a favourable-looking slice is not a result — the
full-scale number is the one reported.)

This finding is wired into the product: `model_credibility()` reads the stored backtest at
prediction time, and when the model fails to beat the market every selection is flagged
`MODEL_UNPROVEN` with a banner explaining why.

### Lineups — the most important test, and it failed

**Does knowing the lineup improve the model?** Measured as an **oracle upper bound**: the
model is given the *actual* starting XI, which is strictly more information than any live
system could have, because the source publishes no lineup timestamps. If perfect lineup
knowledge does not help, chasing lineups in production cannot be worth it.

Both arms trained and scored on identical rows (760 Premier League matches with lineups,
747 usable once "regular starter" has meaning), split chronologically:

| Split | Baseline log loss | + lineups | Delta | n test |
|---|---|---|---|---|
| 20% test | 1.04435 | 1.08294 | **+0.039** | 150 |
| 30% test | 1.06646 | 1.10069 | **+0.034** | 225 |
| 40% test | 1.03739 | 1.06274 | **+0.025** | 299 |
| 50% test | 1.03718 | 1.06122 | **+0.024** | 374 |

**Lineup features made out-of-sample probabilities worse, consistently across every split.**
Accuracy nudged up (0.44 vs 0.41 at the 30% split) while log loss and Brier both degraded —
the model got more confident and less well calibrated.

**They were therefore not promoted into the champion model.** They remain implemented,
ingested and available for research, and the infrastructure that would let them be
re-tested (or fitted properly, once enough live pre/post-lineup pairs accumulate) is in
place. An intuitively useful feature that does not survive testing stays out.

The honest scope of this result: it says *these* lineup features — regular-starter share,
continuity, goalkeeper regularity — do not help on 747 Premier League matches. It does not
prove lineups are worthless; a richer representation built on player-quality data the
platform does not have might do better.

### Feature ablation

Identical walk-forward folds, n = 2,289:

| Variant | Log loss | Brier | Accuracy |
|---|---|---|---|
| **Ratings only (Elo + adjusted efficiency)** | **0.6122** | **0.2123** | 63.8% |
| Form + ratings | 0.6164 | 0.2139 | 65.4% |
| Form only | 0.6165 | 0.2135 | 65.8% |
| Form + ratings + schedule | 0.6179 | 0.2144 | 66.0% |
| Everything | 0.6183 | 0.2146 | 65.0% |

Rolling form, schedule/fatigue and head-to-head features made the model **slightly worse**
out of sample. The gaps are small relative to sampling noise, which is precisely the
argument for the simpler model: the extra features buy nothing measurable while adding
variance. **The default NBA feature set was therefore changed to ratings-only.** The richer
variants remain implemented and are re-tested by `divinelines ablate`.

---

## Betting engine

- **Margin removal** before any comparison. Comparing a model against a *raw* implied
  probability overstates the market's view by the entire overround and manufactures edges
  that do not exist. Both multiplicative and power (favourite-longshot aware) methods are
  implemented; each bookmaker is de-vigged individually before aggregation.
- **EV per unit staked**: `p·(d−1) − (1−p)`. One unit, one definition.
- **Edge-quality score (0–10)** blending edge size, model confidence, data quality,
  calibration, model agreement and liquidity. Every component and weight is returned so the
  number can be audited. Edge size saturates at 6 points — beyond that, a large disagreement
  with the market is more often model error than opportunity.
- **Fractional Kelly**, with the probability shrunk toward the market in proportion to
  uncertainty, then capped.
- **Portfolio limits**: per-game, per-team, per-sport and slate exposure caps applied
  best-first, with the binding constraint reported on each stake.
- **Confidence ≠ probability.** A model can say 67% with high confidence or with low
  confidence; only confidence scales the stake.
- **CLV** against the no-vig closing line, in both price and probability terms.
- **Paper trading by default.** Nothing is ever executed anywhere.

---

## CLI

```
divinelines migrate     import the legacy database (with the repairs above)
divinelines refresh     fetch results, fixtures, injuries, odds
divinelines backfill    historical NBA odds / ESPN id mapping
divinelines lineups     ingest lineups (upcoming, or --historical for research)
divinelines match       soccer match centre
                        ingest|backfill <uid>   pull events, box scores, context
                        live                    re-read matches in progress
                        show|report <uid>       read one match back [--minute N]
divinelines train       train and register models
divinelines replay      replay seasons into the ledger at opening prices
divinelines scan        predictions and +EV opportunities   [--paper] [--dry-run]
divinelines settle      resolve closes, compute CLV, grade results
                        [--sport] [--date] [--dry-run] [--full]
                        [--cutoff] [--close-aggregation] [--close-book]
divinelines clv         CLV distribution, cohorts, and the skill decomposition
                        [--sport] [--basis] [--cohort] [--skill]
divinelines health      predictive vs betting health by sport and window [--persist]
divinelines backtest    walk-forward evaluation                      [--save]
divinelines ablate      feature-ablation study
divinelines evaluate    realised performance from the ledger
divinelines status      data, sources, closing lines, lineups, model status
divinelines serve       run the API
```

`scan`, `settle` and `match live` are safe to run repeatedly on a schedule: predictions
supersede rather than duplicate, settlement is idempotent, and a match re-read replaces its
own event rows rather than accumulating them.

## API

**V2 routes** (unchanged): `GET /api/health` · `/api/config` · `/api/predictions` ·
`/api/games` · `/api/games/{uid}` · `/api/odds/{uid}` · `/api/teams` · `/api/teams/{uid}` ·
`/api/injuries` · `/api/search` · `/api/performance` · `/api/models` · `/api/experiments` ·
`/api/backtests` · `/api/predictions/history` — plus `POST /api/scan` and `POST /api/predict`.

**V3 routes**: `/api/model-health` · `/api/model-health/all` · `/api/model-health/history` ·
`/api/models/lifecycle` · `/api/clv` · `/api/clv/skill` · `/api/clv/coverage` ·
`/api/events/{uid}/timeline` · `/api/events/{uid}/predictions` · `/api/lineups` ·
`/api/lineups/{uid}` · `/api/source-health` · `/api/data-quality` — plus `POST /api/settle`.

**V5 soccer routes**: `/api/soccer/matches` · `/api/soccer/standings` ·
`/api/soccer/match/{uid}` (the whole Match Centre in one request) and its per-panel
siblings `/events` · `/momentum` · `/shots` · `/heatmap` · `/passes` · `/stats` ·
`/players` · `/markets` · `/report`. Every one takes `minute` and `as_of`, and the bound is
applied server-side.

`POST /api/predict` **keeps the v1 contract** — same request body, same response keys
(`home_win_probability`, `quant_edge`, `metrics.last_h2h`, …) — so anything already talking
to this service keeps working. The numbers behind it now come from the calibrated ensemble
and are priced against de-vigged multi-book consensus. Interactive docs at `/docs`.

## Frontend

React + TypeScript, no chart library (SVG charts are hand-rolled and theme-aware).

**Dashboard** · **Match Centre** (soccer: scoreboard, momentum, timeline, statistics,
shot map, event density, lineups, players, market, model, replay slider, written report) ·
**+EV Scanner** (filters, sorting, and a per-selection breakdown of model
agreement, edge-score components, data-quality components, SHAP contributions and the
injury adjustment) · **Model Health & CLV** (status with its justification, calibration,
CLV distribution and cumulative series, the shopping/drift/skill decomposition, cohort
tables with intervals, closing-line coverage) · **Games** with line movement, **lineups**
and a full **event timeline** with change attribution · **Performance** (ledger +
backtests) · **Teams** · **Models** · **System**.

Insufficient samples render as "insufficient sample", never as `0.0%`, and a +0.3% CLV on
six bets never gets green edge styling.

Light/dark/system themes. Chart colours are the validated categorical palette, checked
against both surfaces (all-pairs CVD ΔE 9.2 light / 9.4 dark).

## Configuration

Everything lives in `divinelines/config.py` and is overridable by environment variable —
see `.env.example`. Bankroll, Kelly fraction, exposure caps, minimum edge, calibration
method, shrinkage strength, freshness TTLs, request budgets and supported leagues are all
configuration, not code.

Seasons are derived from the clock. There is no `SEASON = "2025-26"` anywhere.

## Testing

```bash
python -m pytest -q          # 414 tests, ~20s, no network
```

Covering: identity resolution and alias merging · the two v1 data corruptions ·
validation rules · odds conversion, margin removal and consensus · EV, Kelly, portfolio
caps and CLV · calibration metrics and calibrators · Dixon-Coles parameter recovery from
known ground truth · player impact and scenario weighting · settlement grading for
moneyline, 1X2 and totals · API status codes and error handling.

V3 adds: source **contract tests** against stored ESPN payloads (so a field rename fails
loudly rather than silently ingesting nothing), the closing-line policy, settlement
idempotency and incrementality, CLV convention agreement across modules, lineup chronology,
prediction versioning, ROI intervals, and the V3 API including its empty states.

Three tests carry more weight than the rest:

- `test_features_are_identical_when_the_future_is_removed` — rebuilds features from a
  truncated dataset and asserts every shared row is byte-identical. If any feature read the
  future, this fails.
- `test_a_lineup_observed_later_is_invisible_earlier` — an XI seen at 18:15 must not appear
  in a prediction made at 16:00.
- `test_settlement_is_idempotent` — running settlement twice must not duplicate a CLV
  record or double-count profit.

V5 adds adversarial replay tests, which are stronger than checking that a bounded view
looks right. They inject a 72nd-minute goal, a price captured after the cut-off, a
prediction created after it and a lineup observed after it, then assert the earlier view is
**unchanged** — same events, same score, same momentum series, same shot map:

- `test_a_goal_scored_later_is_invisible_at_an_earlier_minute`
- `test_a_price_captured_after_the_replay_position_is_excluded`
- `test_a_prediction_created_after_the_replay_position_is_excluded`
- `test_a_lineup_observed_after_the_replay_position_is_excluded`
- `test_both_sides_attack_their_own_end` — without the coordinate flip, both teams' shots
  land in the same half, which is the single most misleading thing a shot map can do
- `test_a_file_declaring_another_division_is_refused` — see the corruption below

---

## Known limitations

Stated plainly, because a platform that hides these is worse than useless:

1. **The soccer model loses money in backtest** (−5.7% ROI) and is beaten by the no-vig
   market on log loss. Treat its edges as unproven.
2. **The NBA model does not beat the market either** (log loss 0.6164 vs 0.5959). Its
   +4.79% simulated ROI is **not statistically significant** (95% CI [−3.4%, +7.7%]), and
   it depends on entering at opening prices with best-of-14-books line shopping — softer
   prices, lower limits, and an idealised assumption that the best price is always
   obtainable.
3. **The +1.98pp model selection effect is one sport, three seasons, one market.** It is
   measured cleanly, but it is not yet a validated edge.
4. **Lineup features made the model worse** under an oracle bound and are not in the
   champion model. Lineup data is ingested and available, but currently earns nothing.
5. **Historical lineups carry no publication timestamps**, so lineup value can only be
   bounded from above, never backtested honestly. Real measurement requires accumulating
   live pre/post-lineup prediction pairs forward.
6. **NBA replay timestamps are nominal.** ESPN publishes no capture time for opening or
   closing prices, so time-to-event CLV cohorts exclude replay rows entirely.
7. **Injury impact is unvalidated.** ESPN publishes no injury history, so the adjustment
   cannot be backtested. The PIE→margin scale is a documented prior, not a fitted parameter.
8. **No true xG.** Neither football-data.co.uk nor ESPN publishes expected goals. The
   platform uses shots, shots on target and shot locations, and never calls any of them xG.
   The Match Centre shows an explicit "expected goals: not available" row rather than a
   proxy wearing the name.
9. **No pass events anywhere.** ESPN's summary feed carries `totalPasses` and
   `accuratePasses` per team and nothing about who passed to whom or from where. A passing
   network cannot be derived from any configured source, so the panel ships as a real
   endpoint returning `NO_DATA` with the reason. The schema and the component are in place
   for a provider that does publish passes.
10. **Heatmaps are event-location density, not tracking.** Positions exist only for the
   ball at discrete recorded events. Where a player spent the match is not derivable and is
   not claimed.
11. **No player ratings.** No source publishes one for these competitions and DivineLines
   does not compute one. A rating assembled from five box-score counters would be
   confident fiction.
12. **Momentum is descriptive, not predictive.** Weights and decay are a stated prior with
   nothing to fit them against. It has not been tested as a model feature and is
   deliberately not one.
13. **Match event coverage is one league.** 760 Premier League matches (2024-25 and
   2025-26) have events, box scores and match context; the other seven competitions have
   fixtures, prices and results but no in-match events, because their ESPN event ids have
   not been backfilled.
14. **Soccer lineup coverage is the same one league**, for the same reason.
15. **Soccer odds timestamps are approximate.** The source gives a pre-match and a closing
   price without exact capture times, so snapshots are stored at match-day midnight and
   kick-off respectively. Every analysis keys off the `is_closing` flag, never those times.
16. **Best-price backtests assume you can actually get the best price.** `market_best` is the
   maximum across tracked books; real limits and availability are not modelled.
17. **No transfers, coaching changes or trades are ingested.** The schema and the
   `transactions` table support them; no free source is connected.
18. **Bet correlation is controlled by exposure caps, not modelled jointly.**
19. **SQLite is single-writer.** Fine for this workload; a concurrent multi-process
    ingestion would want Postgres.

## Where the v1 code went

The v1 scripts are all still present and runnable. Nothing was deleted.

| v1 | Status |
|---|---|
| `main.py` | Now a launcher for the CLI; the interactive menu remains |
| `core/api.py` | Deprecation shim re-exporting `divinelines.api.app` (same port, same contract) |
| `core/syncData.py`, `nba/data_refresh.py` | Superseded by `sources/nba_stats.py` + `pipeline/refresh.py` (fixes the hard-coded season, the id duplication and the in-progress games) |
| `core/sqlDatabaseBuild.py`, `loadData.py`, `resetDB.py`, `testSQL.py` | Untouched; operate on the v1 database, which is now read-only input |
| `nba/models/buildFeatures.py`, `coupleGameSamples.py` | Superseded by `features/nba_features.py` |
| `nba/models/trainModel.py` | Superseded by `models/nba_model.py` + `pipeline/train.py` (chronological split, calibration, registry) |
| `nba/models/backTest.py` | Superseded by `backtest/` (the v1 version scored the model on its own training data at a flat −110) |
| `nba/models/expectedValueScanner.py`, `matchupPredict.py` | Superseded by `pipeline/predict.py` (also removes f-string SQL) |
| `soccer/` (FotMob/Sofascore player props) | Untouched. It solves a different problem — player props — and depends on Playwright/ScraperFC scraping. The new soccer engine models match outcomes from a stable, licence-friendly source instead |

## Recommended next steps

Ordered by what the evidence says is worth doing, not by what sounds impressive.

1. **Accumulate live pre/post-lineup pairs.** The oracle bound says these lineup features
   do not help, but the live question — does *acting on* a confirmed lineup move us toward
   the close — can only be answered forward. The ledger, staging and timeline are built for
   it; it needs weeks of `divinelines scan` and `divinelines settle`.
2. **Re-test the model selection effect on 2026-27 out of sample.** +1.98pp on three past
   seasons is the most interesting result here; one clean forward season would say whether
   it is real.
3. **Extend historical odds to 2021-22 and 2022-23.** The adapter and backfill already
   work; two more seasons would roughly double the CLV sample.
4. **Backfill lineups for the other seven competitions** to test whether the null result
   holds outside the Premier League.
5. **Model totals and Asian handicaps.** Dixon-Coles already produces the scoreline
   distribution; ESPN publishes historical totals prices alongside the moneylines.
6. **Fit the player-impact scale** once enough live availability observations accumulate,
   replacing the documented prior with something measured.

---

*Nothing here is financial advice. The platform is a research tool, it trades on paper only,
and its own backtests say it has no demonstrated betting edge.*
