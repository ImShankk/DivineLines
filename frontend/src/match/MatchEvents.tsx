/* The event timeline: what changed, in order.

   Selecting an event lifts it into the page's shared selection so the shot
   map and the momentum chart can point at the same moment. That link is what
   separates this from three charts that happen to share a page. */

import { useState } from 'react';
import { Badge } from '../ui';
import { eventLabel, type MatchEvent } from './api';

/** Events that always belong on the timeline, versus the long tail of fouls
    and corners that only appear when the reader asks for them. */
const HEADLINE = new Set([
  'goal', 'own_goal', 'penalty_scored', 'penalty_missed', 'penalty_saved',
  'penalty_woodwork', 'yellow_card', 'red_card', 'substitution', 'var_decision',
]);

const ICON: Record<string, string> = {
  goal: '⚽', own_goal: '⚽', penalty_scored: '⚽',
  penalty_missed: '✕', penalty_saved: '✕', penalty_woodwork: '▮',
  yellow_card: '▮', red_card: '▮', substitution: '⇄', var_decision: '⌗',
  shot_on_target: '◎', shot_off_target: '✕', shot_blocked: '▣', shot_woodwork: '▮',
  corner: '⌐', foul: '·', offside: '⚑', handball: '·', free_kick: '·', save: '◎',
};

function toneFor(type: string): 'pos' | 'neg' | 'warn' | 'neutral' {
  if (type === 'goal' || type === 'penalty_scored') return 'pos';
  if (type === 'red_card' || type === 'own_goal') return 'neg';
  if (type === 'yellow_card' || type === 'var_decision') return 'warn';
  return 'neutral';
}

export function MatchEvents({
  events,
  homeName,
  awayName,
  selectedId,
  onSelect,
}: {
  events: MatchEvent[];
  homeName: string;
  awayName: string;
  selectedId?: number | null;
  onSelect?: (event: MatchEvent | null) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? events : events.filter((event) => HEADLINE.has(event.event_type));

  if (!events.length) {
    return <div className="empty">No events recorded up to this point.</div>;
  }

  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <span className="faint">
          {visible.length} of {events.length} events
        </span>
        <button className="ghost" onClick={() => setShowAll((value) => !value)}>
          {showAll ? 'Key events only' : `Show all ${events.length}`}
        </button>
      </div>

      <ol className="timeline">
        {visible.map((event) => {
          const side = event.home_away;
          const selected = selectedId === event.event_row_id;
          return (
            <li
              key={event.event_row_id}
              className={`timeline-row timeline-${side ?? 'neutral'} ${selected ? 'selected' : ''}`}
            >
              <button
                className="timeline-button"
                aria-pressed={selected}
                onClick={() => onSelect?.(selected ? null : event)}
              >
                <span className="timeline-clock mono">
                  {event.clock_display ?? (event.minute != null ? `${Math.round(event.minute)}′` : '—')}
                </span>
                <span className="timeline-icon" aria-hidden="true">
                  {ICON[event.event_type] ?? '•'}
                </span>
                <span className="timeline-body">
                  <span className="row" style={{ gap: 6 }}>
                    <Badge tone={toneFor(event.event_type)}>{eventLabel(event.event_type)}</Badge>
                    {event.player_name && <strong>{event.player_name}</strong>}
                    {event.home_score != null && HEADLINE.has(event.event_type) &&
                      (event.event_type.startsWith('goal') || event.event_type === 'penalty_scored' ||
                       event.event_type === 'own_goal') && (
                        <span className="num faint">
                          {event.home_score}–{event.away_score}
                        </span>
                      )}
                  </span>
                  <span className="faint">
                    {event.team_name ?? (side === 'home' ? homeName : side === 'away' ? awayName : '')}
                    {event.assist_player_name ? ` · assist ${event.assist_player_name}` : ''}
                  </span>
                  {selected && event.text && <span className="timeline-detail">{event.text}</span>}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
