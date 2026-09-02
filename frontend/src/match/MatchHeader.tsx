/* The scoreboard.

   It renders from the normalised match state, never from a guess: a fixture
   sitting in the database does not make a match live, and a replay is its own
   mode rather than a live match with things hidden. */

import { Badge } from '../ui';
import { STATE_LABELS, type MatchCentre } from './api';

const TONE_CLASS: Record<string, string> = {
  live: 'state-live',
  done: 'state-done',
  idle: 'state-idle',
  off: 'state-off',
};

function Crest({ name, logo, color }: { name: string; logo: string | null; color: string | null }) {
  if (logo) {
    return <img className="crest" src={logo} alt="" width={44} height={44} loading="lazy" />;
  }
  // No crest in the feed: initials on the club colour beat a broken image.
  const initials = name
    .split(/\s+/)
    .slice(0, 2)
    .map((word) => word[0])
    .join('')
    .toUpperCase();
  return (
    <span
      className="crest crest-fallback"
      style={{ background: color ? `#${color}22` : undefined }}
      aria-hidden="true"
    >
      {initials}
    </span>
  );
}

function Form({ form }: { form: string | null }) {
  if (!form) return null;
  return (
    <span className="form-run" title={`Recent form: ${form}`}>
      {form
        .slice(-5)
        .split('')
        .map((result, index) => (
          <span key={index} className={`form-chip form-${result.toLowerCase()}`}>
            {result}
          </span>
        ))}
    </span>
  );
}

export function MatchHeader({ data }: { data: MatchCentre }) {
  const { match, state } = data;
  const meta = STATE_LABELS[state.state] ?? { label: state.state, tone: 'idle' as const };
  const scored = match.home.score !== null && match.away.score !== null;

  return (
    <header className="scoreboard">
      <div className="scoreboard-top">
        <span className="scoreboard-comp">
          {match.league_name ?? match.league_id}
          <span className="faint"> · {match.season}</span>
        </span>
        <span className="row" style={{ gap: 6 }}>
          {state.mode === 'REPLAY' && (
            <Badge tone="accent">replay · {state.replay_minute}′</Badge>
          )}
          {state.mode === 'LIVE' && <Badge tone="neg">live</Badge>}
          <span className={`state-pill ${TONE_CLASS[meta.tone]}`}>
            {meta.label}
            {state.clock_display ? ` ${state.clock_display}` : ''}
          </span>
        </span>
      </div>

      <div className="scoreboard-main">
        <div className="scoreboard-team home">
          <div className="scoreboard-team-id">
            <Crest name={match.home.name} logo={match.home.logo} color={match.home.color} />
            <div>
              <div className="scoreboard-name">{match.home.name}</div>
              <div className="row" style={{ gap: 6 }}>
                {match.home.formation && (
                  <span className="faint mono">{match.home.formation}</span>
                )}
                <Form form={match.home.form} />
              </div>
            </div>
          </div>
        </div>

        <div className="scoreboard-score">
          {scored ? (
            <span className="num">
              {match.home.score}
              <span className="scoreboard-dash">–</span>
              {match.away.score}
            </span>
          ) : (
            <span className="scoreboard-kickoff">
              {match.kickoff_utc
                ? new Date(match.kickoff_utc).toLocaleTimeString(undefined, {
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : '—'}
            </span>
          )}
        </div>

        <div className="scoreboard-team away">
          <div className="scoreboard-team-id">
            <div className="right">
              <div className="scoreboard-name">{match.away.name}</div>
              <div className="row" style={{ gap: 6, justifyContent: 'flex-end' }}>
                <Form form={match.away.form} />
                {match.away.formation && (
                  <span className="faint mono">{match.away.formation}</span>
                )}
              </div>
            </div>
            <Crest name={match.away.name} logo={match.away.logo} color={match.away.color} />
          </div>
        </div>
      </div>

      <div className="scoreboard-foot">
        <span>{match.venue ?? 'Venue unknown'}</span>
        {match.venue_city && <span className="faint">{match.venue_city}</span>}
        {match.attendance != null && (
          <span className="faint">Att. {match.attendance.toLocaleString()}</span>
        )}
        {match.officials?.length > 0 && (
          <span className="faint">Ref. {match.officials.join(', ')}</span>
        )}
        <span className="faint">
          {new Date(match.kickoff_utc ?? `${match.game_date}T00:00:00Z`).toLocaleDateString(
            undefined,
            { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' },
          )}
        </span>
      </div>
    </header>
  );
}
