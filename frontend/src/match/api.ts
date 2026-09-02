/* Typed client and types for the soccer Match Centre.

   Kept beside the match components rather than in the shared api module: this
   is a domain surface with a lot of shapes, and the rest of the app does not
   need to know about any of them.

   Note that `minute` is a *server-side* bound, not a display filter. Every
   request carries it so the payload that comes back is already truncated —
   the frontend is never in a position to leak a future event by forgetting to
   hide it. */

import { request } from '../api';

/** Every panel is in exactly one of these, and says which. */
export type PanelState = 'DATA' | 'LOADING' | 'NO_DATA' | 'ERROR' | 'STALE';

export interface MatchEvent {
  event_row_id: number;
  sequence: number;
  event_type: string;
  source_type: string | null;
  period: number | null;
  minute: number | null;
  clock_display: string | null;
  wallclock_utc: string | null;
  team_uid: string | null;
  team_name: string | null;
  home_away: 'home' | 'away' | null;
  player_uid: string | null;
  player_name: string | null;
  assist_player_name: string | null;
  home_score: number | null;
  away_score: number | null;
  source_x: number | null;
  source_y: number | null;
  text: string | null;
  short_text: string | null;
}

export interface MomentumPoint {
  minute: number;
  home: number;
  away: number;
  net: number;
}

export interface MomentumMarker {
  minute: number;
  clock_display: string | null;
  event_type: string;
  home_away: string | null;
  player_name: string | null;
  event_row_id: number | null;
  text: string | null;
}

export interface MomentumSwing {
  minute: number;
  net: number;
  change: number;
  direction: 'home' | 'away';
  associated_events: {
    event_type: string;
    home_away: string | null;
    player_name: string | null;
    clock_display: string | null;
    text: string | null;
  }[];
  note: string;
}

export interface MomentumSummary {
  available: boolean;
  peak_home: number;
  peak_away: number;
  minutes_home_ahead: number;
  minutes_away_ahead: number;
  share_home: number | null;
  version: string;
}

export interface Momentum {
  available: boolean;
  reason?: string;
  parameters: Record<string, unknown>;
  series: MomentumPoint[];
  markers: MomentumMarker[];
  swings: MomentumSwing[];
  summary?: MomentumSummary;
  events_used?: number;
  note?: string;
}

export type ShotOutcome = 'goal' | 'on_target' | 'off_target' | 'blocked' | 'woodwork';

/** How each shot outcome is drawn and labelled. Shape carries the meaning as
    well as colour, so the map survives being read without colour. */
export const SHOT_LEGEND: {
  outcome: ShotOutcome;
  label: string;
  className: string;
  shape: 'filled' | 'ring' | 'cross' | 'square';
}[] = [
  { outcome: 'goal', label: 'Goal', className: 'shot-goal', shape: 'filled' },
  { outcome: 'on_target', label: 'On target', className: 'shot-on', shape: 'ring' },
  { outcome: 'off_target', label: 'Off target', className: 'shot-off', shape: 'cross' },
  { outcome: 'blocked', label: 'Blocked', className: 'shot-blocked', shape: 'square' },
  { outcome: 'woodwork', label: 'Woodwork', className: 'shot-woodwork', shape: 'ring' },
];

export interface ShotPoint {
  event_row_id: number | null;
  minute: number | null;
  clock_display: string | null;
  team_uid: string | null;
  home_away: 'home' | 'away' | null;
  player_name: string | null;
  event_type: string;
  outcome: ShotOutcome;
  x: number;
  y: number;
  end_x: number | null;
  end_y: number | null;
  distance_m: number;
  text: string | null;
}

export interface StatComparison {
  stat: string;
  label: string;
  kind: string;
  higher_is_better: boolean | null;
  home: number | null;
  away: number | null;
  home_display: string | null;
  away_display: string | null;
  home_share: number | null;
}

export interface QualityComponentState {
  name: string;
  label: string;
  state: 'present' | 'partial' | 'absent';
  detail: string;
  count: number | null;
  coverage: number | null;
}

export interface LineupEntry {
  player_uid: string | null;
  player_name: string | null;
  jersey: string | null;
  role: string | null;
  position_group: string | null;
  formation_place: string | null;
  subbed_in: boolean;
  subbed_out: boolean;
}

export interface LineupSide {
  team_uid: string | null;
  team_name: string | null;
  formation: string | null;
  observed_at: string | null;
  lineup_state: string | null;
  starters: LineupEntry[];
  bench: LineupEntry[];
  other: LineupEntry[];
}

export interface MatchSide {
  team_uid: string;
  name: string;
  score: number | null;
  color: string | null;
  logo: string | null;
  form: string | null;
  formation: string | null;
}

export interface PlayerLine {
  player_uid: string | null;
  player_name: string | null;
  team_uid: string | null;
  home_away: string | null;
  jersey: string | null;
  position: string | null;
  position_group: string;
  starter: boolean;
  subbed_in: boolean;
  subbed_out: boolean;
  stats: { stat: string; label: string; value: number | null; display: string | null }[];
  has_stats: boolean;
}

export interface EventDensity {
  kind: string;
  basis: string;
  not_tracking: boolean;
  note: string;
  columns: number;
  rows: number;
  pitch: { length: number; width: number };
  grid: number[][];
  peak: number;
  events_considered: number;
  events_located: number;
  coverage: number | null;
}

export interface PassingPanel {
  available: boolean;
  state: string;
  reason: string;
  requires: string[];
  passes: unknown[];
  aggregate_totals: Record<string, Record<string, number>>;
  note: string;
}

export interface MarketSnapshot {
  captured_at: string;
  phase: string;
  consensus: Record<string, number>;
  best: Record<string, number>;
  novig: Record<string, number> | null;
  books: number;
}

export interface MarketPanel {
  available: boolean;
  market: string;
  reason?: string;
  selections?: string[];
  series?: MarketSnapshot[];
  opening?: MarketSnapshot;
  closing?: MarketSnapshot | null;
  latest?: MarketSnapshot;
  snapshots: number;
  books?: number;
  sources?: string[];
  as_of?: string | null;
  note?: string;
}

export interface StandingsRow {
  position: number;
  team_uid: string;
  team_name: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
}

export interface ModelPrediction {
  prediction_id: number;
  created_at: string;
  market: string;
  selection: string;
  model_prob: number;
  market_prob: number | null;
  price_decimal: number | null;
  bookmaker: string | null;
  edge: number | null;
  stake: number | null;
  model_version: string | null;
  lineup_state: string | null;
  prediction_stage: string | null;
  superseded_at: string | null;
  mode: string;
  flags: string[] | null;
}

export interface MatchCentre {
  game_uid: string;
  generated_at: string;
  as_of: string | null;
  replay_minute: number | null;
  replay: {
    observation_as_of: string | null;
    information_as_of: string | null;
    replay_minute: number | null;
    events_basis: string;
    retrospective_events: boolean;
    note: string;
  };
  match: {
    game_uid: string;
    league_id: string;
    league_name: string | null;
    league_country: string | null;
    season: string;
    game_date: string;
    kickoff_utc: string | null;
    status: string;
    home: MatchSide;
    away: MatchSide;
    venue: string | null;
    venue_city: string | null;
    attendance: number | null;
    officials: string[];
  };
  state: {
    state: string;
    mode: 'LIVE' | 'REPLAY' | 'POST_MATCH' | 'PRE_MATCH';
    is_live: boolean;
    period: number | null;
    clock_display: string | null;
    status_detail: string | null;
    replay_minute: number | null;
  };
  events: MatchEvent[];
  statistics: {
    available: boolean;
    comparisons: StatComparison[];
    unmapped_stats: string[];
    basis?: string;
    unavailable?: string[];
    note?: string;
  };
  lineups: { home: LineupSide; away: LineupSide };
  players: { available: boolean; players: PlayerLine[]; rated: boolean; rating_note: string };
  contributions: {
    goals: { player_name: string; home_away: string; goals: number; own_goals: number; minutes: (string | number)[] }[];
    assists: { player_name: string; home_away: string; assists: number }[];
  };
  momentum: Momentum;
  shots: {
    available: boolean;
    points: ShotPoint[];
    total_shots: number;
    located: number;
    reason: string | null;
    pitch: { length: number; width: number; orientation: string };
  };
  heatmap: EventDensity;
  passing: PassingPanel;
  market: MarketPanel;
  model: {
    available: boolean;
    predictions: ModelPrediction[];
    latest: Record<string, ModelPrediction>;
    count: number;
    superseded: number;
    reason: string | null;
  };
  model_vs_market: {
    available: boolean;
    reason?: string;
    rows?: {
      selection: string;
      model_probability: number;
      market_probability: number;
      difference_points: number;
      price: number | null;
      bookmaker: string | null;
      stake: number | null;
    }[];
    note?: string;
  };
  standings: {
    available: boolean;
    reason?: string;
    table?: StandingsRow[];
    highlight?: (string | null)[];
    as_of_date?: string;
    note?: string;
  };
  quality: {
    components: QualityComponentState[];
    present: number;
    partial: number;
    absent: number;
    grade: string;
    note: string;
  };
  provenance: Record<string, unknown>;
}

export interface SoccerMatchRow {
  game_uid: string;
  league_id: string;
  season: string;
  game_date: string;
  kickoff_utc: string | null;
  status: string;
  home_name: string;
  away_name: string;
  home_score: number | null;
  away_score: number | null;
  status_name: string | null;
  venue: string | null;
  attendance: number | null;
  events: number;
  prices: number;
  starters: number;
  predictions: number;
}

const bound = (minute?: number | null) =>
  minute === null || minute === undefined ? '' : `&minute=${minute}`;

export const soccerApi = {
  matches: (params: {
    league_id?: string;
    season?: string;
    status?: string;
    withEvents?: boolean;
    limit?: number;
  } = {}) => {
    const query = new URLSearchParams();
    if (params.league_id) query.set('league_id', params.league_id);
    if (params.season) query.set('season', params.season);
    if (params.status) query.set('status', params.status);
    if (params.withEvents) query.set('with_events', 'true');
    query.set('limit', String(params.limit ?? 60));
    return request<{ matches: SoccerMatchRow[]; count: number }>(
      `/api/soccer/matches?${query.toString()}`,
    );
  },
  match: (gameUid: string, minute?: number | null) =>
    request<MatchCentre>(
      `/api/soccer/match/${encodeURIComponent(gameUid)}?market=1x2${bound(minute)}`,
    ),
  momentum: (gameUid: string, minute?: number | null) =>
    request<Momentum>(
      `/api/soccer/match/${encodeURIComponent(gameUid)}/momentum?market=1x2${bound(minute)}`,
    ),
  heatmap: (gameUid: string, side?: 'home' | 'away' | null, minute?: number | null) =>
    request<EventDensity>(
      `/api/soccer/match/${encodeURIComponent(gameUid)}/heatmap?market=1x2${
        side ? `&side=${side}` : ''
      }${bound(minute)}`,
    ),
  passes: (gameUid: string) =>
    request<PassingPanel>(`/api/soccer/match/${encodeURIComponent(gameUid)}/passes`),
  report: (gameUid: string, minute?: number | null) =>
    request<MatchReport>(
      `/api/soccer/match/${encodeURIComponent(gameUid)}/report?market=1x2${bound(minute)}`,
    ),
};

export interface MatchReport {
  game_uid: string;
  generated_at: string;
  replay: MatchCentre['replay'];
  result: {
    headline: string;
    state: string;
    competition: string | null;
    season: string;
    kickoff_utc: string | null;
    venue: string | null;
    attendance: number | null;
    officials: string[];
  };
  key_events: {
    minute: string | number | null;
    event_type: string;
    team: string | null;
    player: string | null;
    assist: string | null;
    score: string | null;
    text: string | null;
  }[];
  scorers: MatchCentre['contributions']['goals'];
  assists: MatchCentre['contributions']['assists'];
  momentum: { prose: string; summary?: MomentumSummary; largest_swings: MomentumSwing[] };
  statistics: { headline: StatComparison[]; basis?: string; unavailable: string[] };
  shooting: {
    total_shots: number;
    located: number;
    by_outcome: Record<string, Record<string, number>>;
    note: string;
  };
  passing: { state: string; reason: string; aggregate_totals: Record<string, Record<string, number>> };
  market: { prose: string; snapshots: number };
  model: { prose: string; predictions: number };
  data_quality: MatchCentre['quality'];
  limitations: string[];
}

/* ------------------------------------------------------------- formatting */

/** Event types the reader recognises, spelled out. */
export const EVENT_LABELS: Record<string, string> = {
  goal: 'Goal',
  own_goal: 'Own goal',
  penalty_scored: 'Penalty scored',
  penalty_missed: 'Penalty missed',
  penalty_saved: 'Penalty saved',
  penalty_woodwork: 'Penalty off the woodwork',
  shot_on_target: 'Shot on target',
  shot_off_target: 'Shot off target',
  shot_blocked: 'Shot blocked',
  shot_woodwork: 'Off the woodwork',
  corner: 'Corner',
  foul: 'Foul',
  handball: 'Handball',
  offside: 'Offside',
  free_kick: 'Free kick',
  yellow_card: 'Yellow card',
  red_card: 'Red card',
  substitution: 'Substitution',
  var_decision: 'VAR decision',
  save: 'Save',
  blocked_pass: 'Blocked pass',
  kickoff: 'Kick-off',
  halftime: 'Half time',
  second_half_start: 'Second half',
  full_time: 'Full time',
};

export const eventLabel = (type: string) =>
  EVENT_LABELS[type] ?? type.replace(/_/g, ' ');

/** Match state as a short human label plus the tone it should read in. */
export const STATE_LABELS: Record<string, { label: string; tone: 'live' | 'done' | 'idle' | 'off' }> = {
  SCHEDULED: { label: 'Scheduled', tone: 'idle' },
  LIVE_FIRST_HALF: { label: '1st half', tone: 'live' },
  HALFTIME: { label: 'Half time', tone: 'live' },
  LIVE_SECOND_HALF: { label: '2nd half', tone: 'live' },
  EXTRA_TIME: { label: 'Extra time', tone: 'live' },
  PENALTIES: { label: 'Penalties', tone: 'live' },
  FINISHED: { label: 'Full time', tone: 'done' },
  POSTPONED: { label: 'Postponed', tone: 'off' },
  CANCELLED: { label: 'Cancelled', tone: 'off' },
};
