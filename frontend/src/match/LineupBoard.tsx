/* Lineups, with the thing the rest of the platform actually cares about:
   whether the XI was known before the prediction was made.

   `lineup_state = 'final'` means the XI was read after kick-off. It is a
   historical record, not information anyone had beforehand, and the board says
   so rather than letting a confirmed-looking teamsheet imply otherwise. */

import { Badge } from '../ui';
import { FormationBoard } from './Pitch';
import type { LineupEntry, LineupSide } from './api';

const STATE_NOTE: Record<string, { tone: 'pos' | 'warn' | 'neutral'; text: string }> = {
  confirmed: { tone: 'pos', text: 'Confirmed before kick-off — usable as live information.' },
  projected: { tone: 'warn', text: 'Projected XI; not yet confirmed by the source.' },
  final: {
    tone: 'neutral',
    text: 'Observed after kick-off. A historical record, not information available beforehand.',
  },
};

function PlayerRow({ entry }: { entry: LineupEntry }) {
  return (
    <li className="lineup-row">
      <span className="lineup-jersey mono">{entry.jersey ?? '–'}</span>
      <span className="lineup-name">{entry.player_name}</span>
      <span className="faint">{entry.role ?? entry.position_group ?? ''}</span>
      {entry.subbed_out && <Badge tone="neutral">off</Badge>}
      {entry.subbed_in && <Badge tone="accent">on</Badge>}
    </li>
  );
}

function Side({ side, which }: { side: LineupSide; which: 'home' | 'away' }) {
  const note = side.lineup_state ? STATE_NOTE[side.lineup_state] : undefined;
  return (
    <div className="lineup-side">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <strong>{side.team_name ?? '—'}</strong>
        <span className="row" style={{ gap: 6 }}>
          {side.formation && <span className="mono faint">{side.formation}</span>}
          {side.lineup_state && (
            <Badge tone={note?.tone ?? 'neutral'} title={note?.text}>
              {side.lineup_state}
            </Badge>
          )}
        </span>
      </div>

      {side.starters.length > 0 ? (
        <>
          <FormationBoard
            entries={side.starters}
            side={which}
            teamName={side.team_name ?? which}
          />
          <div className="lineup-caption faint">
            Positions come from the published formation slot, not from measured player
            locations.
          </div>
          <ul className="lineup-list">
            {side.starters.map((entry) => (
              <PlayerRow key={entry.player_uid ?? entry.player_name} entry={entry} />
            ))}
          </ul>
        </>
      ) : (
        <div className="empty">No starting XI observed.</div>
      )}

      {side.bench.length > 0 && (
        <>
          <div className="lineup-subhead">Bench</div>
          <ul className="lineup-list">
            {side.bench.map((entry) => (
              <PlayerRow key={entry.player_uid ?? entry.player_name} entry={entry} />
            ))}
          </ul>
        </>
      )}

      {side.observed_at && (
        <div className="faint" style={{ marginTop: 6, fontSize: '0.72rem' }}>
          Observed {new Date(side.observed_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

export function LineupBoard({ lineups }: { lineups: { home: LineupSide; away: LineupSide } }) {
  const state = lineups.home.lineup_state ?? lineups.away.lineup_state;
  const note = state ? STATE_NOTE[state] : undefined;

  return (
    <div className="stack">
      {note && (
        <div className={`banner banner-${note.tone === 'pos' ? 'info' : 'warn'}`}>{note.text}</div>
      )}
      <div className="lineup-grid">
        <Side side={lineups.home} which="home" />
        <Side side={lineups.away} which="away" />
      </div>
    </div>
  );
}
