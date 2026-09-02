/* Typed client for the DivineLines API.
   Every call goes through `request` so errors surface as real error states in
   the UI rather than as a blank panel. */

const BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

/** Exported so domain-scoped clients (match centre, ...) share one error path. */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch {
    throw new ApiError(
      'Cannot reach the DivineLines API. Start it with `divinelines serve`.',
      0,
    );
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
    } catch { /* non-JSON error body */ }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

/* ----------------------------------------------------------------- types */

export type Sport = 'nba' | 'soccer';

export interface QualityComponent {
  name: string;
  score: number;
  weight: number;
  note?: string | null;
}

export interface Opportunity {
  game_uid: string;
  sport: string;
  league_id: string;
  game_date: string;
  kickoff_utc?: string | null;
  home_name: string;
  away_name: string;
  home_team_uid?: string;
  away_team_uid?: string;
  market: string;
  selection: string;
  model_probability: number;
  market_probability: number | null;
  price_decimal: number | null;
  bookmaker: string | null;
  edge: number | null;
  ev_per_unit: number | null;
  kelly_fraction: number;
  stake: number;
  confidence: number;
  edge_score: number;
  data_quality: number;
  model_id: string | null;
  model_version: string | null;
  components: Record<string, number>;
  agreement: number;
  explanation: { feature: string; contribution: number; direction: string }[];
  availability: Record<string, any> | null;
  flags: string[];
  quality_detail: { score: number; grade: string; components: QualityComponent[] };
  edge_detail: { score: number; components: QualityComponent[] };
  n_bookmakers: number;
  probability_range?: number[] | null;
}

export interface ScanResponse {
  generated_at: string;
  sport: string;
  opportunities: Opportunity[];
  predictions: Opportunity[];
  portfolio: {
    bankroll?: number;
    total_stake?: number;
    exposure_pct?: number;
    n_bets?: number;
    allocations?: any[];
    dropped?: { key: string; reason: string }[];
  };
  warnings: string[];
  notes: string[];
  cached: boolean;
}

export interface Game {
  game_uid: string;
  sport: string;
  league_id: string;
  season: string;
  game_date: string;
  kickoff_utc?: string | null;
  status: string;
  home_name: string;
  away_name: string;
  home_team_uid: string;
  away_team_uid: string;
  home_score: number | null;
  away_score: number | null;
  venue?: string | null;
}

export interface SourceHealth {
  source: string;
  dataset: string;
  status: string | null;
  last_success: string | null;
  age_minutes: number | null;
  state: string;
  message: string | null;
}

export interface Health {
  status: 'ok' | 'degraded' | 'down';
  mode: string;
  checked_at: string;
  database: Record<string, any>;
  sources: SourceHealth[];
  models: { model_id: string; sport: string; kind: string; model_version: string; trained_at: string }[];
  validation: { ok: boolean; critical: any[]; warnings: any[] };
}

export interface Performance {
  overall: any[];
  by_sport: any[];
  by_edge_bucket: any[];
  by_odds_bucket: any[];
  bankroll_curve: any[];
  clv: { n: number; mean_pct: number | null; beat_close_rate: number | null };
  open_bets: number;
  note?: string | null;
}

export interface ClvSummary {
  n: number;
  mean_clv_price_pct: number;
  median_clv_price_pct: number;
  beat_close_rate: number;
  std_clv_price_pct: number;
  ci_low: number | null;
  ci_high: number | null;
  percentiles: Record<string, number> | null;
  significant: boolean;
  interpretation: string;
  basis?: string;
}

export interface CohortRow {
  cohort: string;
  n: number;
  mean_clv: number;
  median_clv: number;
  positive_rate: number;
  ci_low: number | null;
  ci_high: number | null;
  significant: boolean;
  interpretation: string;
}

export interface ClvReport {
  basis: string;
  basis_description: string;
  overall: ClvSummary;
  sufficient_sample: boolean;
  min_sample_for_inference: number;
  cohorts: Record<string, CohortRow[]>;
  cumulative: { entry_timestamp: string; clv: number; cumulative_mean: number; n: number }[];
  distribution: { bucket: string; count: number }[];
  sample_size: number;
  vs_profit?: {
    clv: ClvSummary | null;
    roi: number | null;
    roi_interval?: {
      n: number;
      roi: number | null;
      ci_low: number | null;
      ci_high: number | null;
      significant: boolean;
      interpretation: string;
    };
    profit: number;
    staked: number;
    settled_bets: number;
    note: string;
  };
  disclaimer?: string;
}

export interface SkillTest {
  recommended: { label: string; n: number; mean: number; median: number; positive_rate: number };
  rejected: { label: string; n: number; mean: number; median: number; positive_rate: number };
  difference: number;
  ci_low: number | null;
  ci_high: number | null;
  significant: boolean;
  interpretation: string;
  basis: string;
}

export interface ClvSkill {
  available: boolean;
  reason?: string;
  n?: number;
  components?: {
    reported_clv_mean_pct: number | null;
    line_shopping_pct: number;
    market_drift_pct: number;
    note: string;
  };
  skill_tests?: Record<string, SkillTest>;
  both_sides_control?: { n_games: number; mean_of_both_sides_pct?: number; note: string };
}

export interface ModelHealth {
  sport: string;
  market: string | null;
  model_version: string | null;
  window: string;
  sample_size: number;
  predictive: Record<string, number>;
  market_comparison: Record<string, any>;
  betting: {
    clv: ClvSummary | null;
    clv_same_book: ClvSummary | null;
    settled_bets: number;
    staked: number;
    profit: number;
    roi: number | null;
    max_drawdown: number | null;
  };
  calibration_curve: { predicted: number; observed: number; count: number }[];
  stability: Record<string, any>;
  status: string;
  status_reason: string;
  note?: string;
}

export interface TimelineEntry {
  timestamp: string;
  kind: 'prediction' | 'market' | 'lineup' | 'information';
  label: string;
  detail: Record<string, any>;
}

export interface EventTimeline {
  found: boolean;
  game: Record<string, any>;
  timeline: TimelineEntry[];
  attributions: {
    timestamp: string;
    market: string;
    selection: string;
    move: number | null;
    candidate_causes: { timestamp: string; kind: string; label: string }[];
    explained: boolean;
    note: string;
  }[];
  market_vs_model: Record<string, any>;
  clv: Record<string, any>[];
}

export interface ModelRow {
  model_id: string;
  sport: string;
  league_id: string | null;
  kind: string;
  model_version: string;
  feature_set: string;
  feature_set_version: string;
  train_start: string | null;
  train_end: string | null;
  valid_start: string | null;
  valid_end: string | null;
  data_version: string | null;
  random_seed: number | null;
  trained_at: string;
  notes: string | null;
  metrics: Record<string, any>;
}

/* --------------------------------------------------------------- endpoints */

export const api = {
  health: () => request<Health>('/api/health'),
  config: () => request<any>('/api/config'),
  predictions: (sport: Sport, daysAhead = 3) =>
    request<ScanResponse>(`/api/predictions?sport=${sport}&days_ahead=${daysAhead}`),
  runScan: (sports: Sport[], daysAhead = 3, paperTrade = false) =>
    request<any>('/api/scan', {
      method: 'POST',
      body: JSON.stringify({ sports, days_ahead: daysAhead, paper_trade: paperTrade }),
    }),
  games: (sport: Sport, status?: string, days = 14) =>
    request<{ games: Game[]; count: number }>(
      `/api/games?sport=${sport}&days=${days}${status ? `&status=${status}` : ''}`,
    ),
  game: (gameUid: string) => request<any>(`/api/games/${encodeURIComponent(gameUid)}`),
  odds: (gameUid: string) => request<any>(`/api/odds/${encodeURIComponent(gameUid)}`),
  teams: (sport: Sport) => request<{ teams: any[]; count: number }>(`/api/teams?sport=${sport}`),
  team: (teamUid: string) => request<any>(`/api/teams/${encodeURIComponent(teamUid)}`),
  injuries: (sport: Sport = 'nba') => request<any>(`/api/injuries?sport=${sport}`),
  performance: (mode?: string) =>
    request<Performance>(`/api/performance${mode ? `?mode=${mode}` : ''}`),
  models: (sport?: string) =>
    request<{ models: ModelRow[]; count: number }>(`/api/models${sport ? `?sport=${sport}` : ''}`),
  experiments: () => request<any>('/api/experiments'),
  backtests: () => request<any>('/api/backtests'),
  search: (q: string) => request<any>(`/api/search?q=${encodeURIComponent(q)}`),
  history: (sport?: string) =>
    request<any>(`/api/predictions/history${sport ? `?sport=${sport}` : ''}`),

  // --- V3 -----------------------------------------------------------------
  modelHealth: (sport: Sport, window = 'all_time') =>
    request<ModelHealth>(`/api/model-health?sport=${sport}&window=${window}`),
  modelHealthAll: (sport: Sport) =>
    request<{ sport: string; windows: Record<string, ModelHealth>; regression: any; stability: any }>(
      `/api/model-health/all?sport=${sport}`,
    ),
  clv: (sport?: Sport, basis: 'consensus' | 'same_book' = 'consensus') =>
    request<ClvReport>(`/api/clv?basis=${basis}${sport ? `&sport=${sport}` : ''}`),
  clvSkill: (sport: Sport = 'nba', mode = 'backtest') =>
    request<ClvSkill>(`/api/clv/skill?sport=${sport}&mode=${mode}`),
  clvCoverage: (sport?: Sport) =>
    request<any>(`/api/clv/coverage${sport ? `?sport=${sport}` : ''}`),
  timeline: (gameUid: string) =>
    request<EventTimeline>(`/api/events/${encodeURIComponent(gameUid)}/timeline`),
  lineups: (gameUid: string) => request<any>(`/api/lineups/${encodeURIComponent(gameUid)}`),
  lineupCoverage: (sport: Sport = 'soccer') => request<any>(`/api/lineups?sport=${sport}`),
  sourceHealth: () => request<any>('/api/source-health'),
  dataQuality: () => request<any>('/api/data-quality'),
};

/** Percent values that already arrive in percentage units (CLV, not fractions). */
export const pctPoints = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${value >= 0 ? '+' : ''}${value.toFixed(digits)}%`;

/* ------------------------------------------------------------- formatting */

export const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${(value * 100).toFixed(digits)}%`;

export const signedPct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : value.toFixed(digits);

export const money = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '—'
    : `${value < 0 ? '-' : ''}$${Math.abs(value).toFixed(digits)}`;

/** Decimal odds -> American, for readers who think in American prices. */
export const american = (decimal: number | null | undefined) => {
  if (!decimal || decimal <= 1) return '—';
  return decimal >= 2
    ? `+${Math.round((decimal - 1) * 100)}`
    : `${Math.round(-100 / (decimal - 1))}`;
};

export const shortDate = (value: string | null | undefined) => {
  if (!value) return '—';
  const date = new Date(value.length <= 10 ? `${value}T00:00:00Z` : value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
};

export const timeAgo = (value: string | null | undefined) => {
  if (!value) return 'never';
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return value;
  const minutes = Math.floor((Date.now() - then) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
};
