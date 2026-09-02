-- DivineLines canonical schema (v2).
-- Every fact table carries provenance: which source produced it and when it
-- was retrieved.  Nothing in the platform stores a number it cannot trace.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------- entities

CREATE TABLE IF NOT EXISTS leagues (
    league_id   TEXT PRIMARY KEY,      -- 'NBA', 'ENG_PL', ...
    sport       TEXT NOT NULL,         -- 'nba' | 'soccer'
    name        TEXT NOT NULL,
    country     TEXT,
    tier        INTEGER,
    strength    REAL                   -- coarse cross-league prior (documented)
);

CREATE TABLE IF NOT EXISTS teams (
    team_uid        TEXT PRIMARY KEY,  -- 'nba:LAL' | 'soccer:manchester-city'
    sport           TEXT NOT NULL,
    canonical_name  TEXT NOT NULL,
    abbr            TEXT,
    country         TEXT,
    external_ids    TEXT,              -- JSON: {"nba_team_id": 1610612747}
    lat             REAL,
    lon             REAL,
    tz              TEXT,
    first_seen      TEXT,
    last_seen       TEXT
);
CREATE INDEX IF NOT EXISTS idx_teams_sport ON teams(sport);
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(canonical_name);

CREATE TABLE IF NOT EXISTS players (
    player_uid    TEXT PRIMARY KEY,
    sport         TEXT NOT NULL,
    full_name     TEXT NOT NULL,
    position      TEXT,
    team_uid      TEXT REFERENCES teams(team_uid),
    external_ids  TEXT,
    active        INTEGER,
    source        TEXT,
    retrieved_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_uid);
CREATE INDEX IF NOT EXISTS idx_players_name ON players(sport, full_name);

-- ------------------------------------------------------------------- games

CREATE TABLE IF NOT EXISTS games (
    game_uid        TEXT PRIMARY KEY,  -- 'nba:0022500002' | 'soccer:ENG_PL:...'
    sport           TEXT NOT NULL,
    league_id       TEXT NOT NULL REFERENCES leagues(league_id),
    season          TEXT NOT NULL,     -- '2025-26' (nba) | '2425' (soccer)
    game_date       TEXT NOT NULL,     -- ISO date of local tip-off/kick-off
    kickoff_utc     TEXT,              -- ISO timestamp when known
    status          TEXT NOT NULL,     -- 'scheduled' | 'final'
    home_team_uid   TEXT NOT NULL REFERENCES teams(team_uid),
    away_team_uid   TEXT NOT NULL REFERENCES teams(team_uid),
    home_score      INTEGER,
    away_score      INTEGER,
    neutral_site    INTEGER DEFAULT 0,
    venue           TEXT,
    source          TEXT NOT NULL,
    retrieved_at    TEXT NOT NULL,
    UNIQUE (sport, league_id, season, game_date, home_team_uid, away_team_uid)
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(sport, game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(sport, season);
CREATE INDEX IF NOT EXISTS idx_games_home ON games(home_team_uid, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away ON games(away_team_uid, game_date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status, game_date);

-- One row per team per NBA game (box score).
CREATE TABLE IF NOT EXISTS nba_team_game (
    game_uid    TEXT NOT NULL REFERENCES games(game_uid) ON DELETE CASCADE,
    team_uid    TEXT NOT NULL REFERENCES teams(team_uid),
    opp_uid     TEXT NOT NULL REFERENCES teams(team_uid),
    is_home     INTEGER NOT NULL,
    won         INTEGER,
    min         REAL, fgm REAL, fga REAL, fg3m REAL, fg3a REAL,
    ftm REAL, fta REAL, oreb REAL, dreb REAL, reb REAL,
    ast REAL, stl REAL, blk REAL, tov REAL, pf REAL,
    pts REAL, plus_minus REAL,
    source        TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    PRIMARY KEY (game_uid, team_uid)
);
CREATE INDEX IF NOT EXISTS idx_ntg_team ON nba_team_game(team_uid);

-- Match-level soccer detail (football-data.co.uk fields).
CREATE TABLE IF NOT EXISTS soccer_match_stats (
    game_uid     TEXT PRIMARY KEY REFERENCES games(game_uid) ON DELETE CASCADE,
    ht_home      INTEGER, ht_away INTEGER,
    home_shots   INTEGER, away_shots INTEGER,
    home_sot     INTEGER, away_sot INTEGER,
    home_corners INTEGER, away_corners INTEGER,
    home_fouls   INTEGER, away_fouls INTEGER,
    home_yellow  INTEGER, away_yellow INTEGER,
    home_red     INTEGER, away_red INTEGER,
    referee      TEXT,
    source       TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);

-- ------------------------------------------------- availability / roster

CREATE TABLE IF NOT EXISTS player_status (
    status_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    player_uid    TEXT NOT NULL,
    team_uid      TEXT REFERENCES teams(team_uid),
    sport         TEXT NOT NULL,
    status        TEXT NOT NULL,   -- out|doubtful|questionable|probable|available|suspended|rest|day-to-day
    detail        TEXT,
    expected_return TEXT,
    as_of         TEXT NOT NULL,   -- when the source says this was true
    source        TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status_team ON player_status(team_uid, as_of);
CREATE INDEX IF NOT EXISTS idx_status_player ON player_status(player_uid, as_of);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    sport           TEXT NOT NULL,
    player_uid      TEXT,
    player_name     TEXT NOT NULL,
    from_team_uid   TEXT,
    to_team_uid     TEXT,
    kind            TEXT NOT NULL,   -- trade|signing|waiver|transfer|loan|loan_return|draft
    effective_date  TEXT NOT NULL,
    fee             REAL,
    detail          TEXT,
    source          TEXT NOT NULL,
    retrieved_at    TEXT NOT NULL,
    UNIQUE (sport, player_name, kind, effective_date, to_team_uid)
);
CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(sport, effective_date);

-- ------------------------------------------------------------------ market

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    game_uid      TEXT REFERENCES games(game_uid) ON DELETE CASCADE,
    sport         TEXT NOT NULL,
    market        TEXT NOT NULL,      -- 'h2h' | '1x2' | 'totals'
    selection     TEXT NOT NULL,      -- 'home' | 'draw' | 'away' | 'over_2.5'
    bookmaker     TEXT NOT NULL,
    price_decimal REAL NOT NULL,
    captured_at   TEXT NOT NULL,      -- when WE observed it
    book_updated  TEXT,               -- when the book says it changed
    is_closing    INTEGER DEFAULT 0,
    source        TEXT NOT NULL,
    UNIQUE (game_uid, market, selection, bookmaker, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_uid, market, captured_at);
CREATE INDEX IF NOT EXISTS idx_odds_captured ON odds_snapshots(captured_at);

-- ------------------------------------------------------ models & predictions

CREATE TABLE IF NOT EXISTS models (
    model_id            TEXT PRIMARY KEY,
    sport               TEXT NOT NULL,
    league_id           TEXT,
    kind                TEXT NOT NULL,   -- 'xgboost'|'logistic'|'elo'|'dixon_coles'|'ensemble'
    model_version       TEXT NOT NULL,
    feature_set         TEXT NOT NULL,   -- JSON list
    feature_set_version TEXT,
    hyperparameters     TEXT,            -- JSON
    train_start         TEXT, train_end TEXT,
    valid_start         TEXT, valid_end TEXT,
    metrics             TEXT,            -- JSON
    data_version        TEXT,
    random_seed         INTEGER,
    artifact_path       TEXT,
    trained_at          TEXT NOT NULL,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_models_sport ON models(sport, trained_at);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    sport          TEXT NOT NULL,
    variant        TEXT NOT NULL,
    feature_set    TEXT NOT NULL,
    model_kind     TEXT NOT NULL,
    hyperparameters TEXT,
    train_range    TEXT, valid_range TEXT,
    metrics        TEXT NOT NULL,
    n_train        INTEGER, n_valid INTEGER,
    random_seed    INTEGER,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name, created_at);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,     -- decision timestamp
    sport             TEXT NOT NULL,
    league_id         TEXT,
    game_uid          TEXT REFERENCES games(game_uid) ON DELETE CASCADE,
    market            TEXT NOT NULL,
    selection         TEXT NOT NULL,
    model_prob        REAL NOT NULL,
    market_prob       REAL,              -- no-vig consensus at decision time
    price_decimal     REAL,
    bookmaker         TEXT,
    edge              REAL,
    ev_per_unit       REAL,
    kelly_fraction    REAL,
    stake             REAL,
    confidence        REAL,              -- 0-1, distinct from probability
    edge_score        REAL,              -- 0-10 quality score
    data_quality      REAL,              -- 0-100
    model_id          TEXT REFERENCES models(model_id),
    model_version     TEXT,
    data_version      TEXT,
    features          TEXT,              -- JSON snapshot of model inputs
    explanation       TEXT,              -- JSON contributions
    flags             TEXT,              -- JSON list of alerts
    mode              TEXT NOT NULL,     -- 'live' | 'research' | 'paper'
    UNIQUE (created_at, game_uid, market, selection, model_id)
);
CREATE INDEX IF NOT EXISTS idx_pred_game ON predictions(game_uid);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(sport, created_at);

CREATE TABLE IF NOT EXISTS bets (
    bet_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    placed_at       TEXT NOT NULL,
    sport           TEXT NOT NULL,
    game_uid        TEXT REFERENCES games(game_uid) ON DELETE CASCADE,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    price_decimal   REAL NOT NULL,
    bookmaker       TEXT,
    stake           REAL NOT NULL,
    model_prob      REAL,
    market_prob     REAL,
    status          TEXT NOT NULL DEFAULT 'open',  -- open|won|lost|push|void
    settled_at      TEXT,
    payout          REAL,
    profit          REAL,
    closing_price   REAL,
    closing_prob    REAL,
    clv             REAL,             -- log(our no-vig prob / closing no-vig prob)
    mode            TEXT NOT NULL     -- 'paper' | 'live'
);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status, placed_at);
CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_uid);

-- --------------------------------------------------------- data operations

CREATE TABLE IF NOT EXISTS source_status (
    source        TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    last_attempt  TEXT,
    last_success  TEXT,
    status        TEXT,              -- ok|error|degraded
    rows          INTEGER,
    latency_ms    INTEGER,
    message       TEXT,
    PRIMARY KEY (source, dataset)
);

CREATE TABLE IF NOT EXISTS validation_issues (
    issue_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at  TEXT NOT NULL,
    dataset      TEXT NOT NULL,
    severity     TEXT NOT NULL,      -- critical|warning
    code         TEXT NOT NULL,
    detail       TEXT,
    entity       TEXT
);
CREATE INDEX IF NOT EXISTS idx_issues_detected ON validation_issues(detected_at);
