/* The written report: the same material as the panels, as prose.

   It is generated server-side from the Match Centre payload, so the report and
   the charts cannot drift apart. The limitations section is not an appendix —
   it is the part that stops the rest from being read as more than it is. */

import { ErrorState, Loading, useAsync } from '../ui';
import { Panel } from './Panel';
import { soccerApi } from './api';

export function MatchReportView({
  gameUid,
  minute,
}: {
  gameUid: string;
  minute: number | null;
}) {
  const { data, error, loading, refresh } = useAsync(
    () => soccerApi.report(gameUid, minute),
    [gameUid, minute],
  );

  if (loading) return <Loading label="Building match report" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  return (
    <div className="stack report">
      <Panel title="Result" state="DATA">
        <h2>{data.result.headline}</h2>
        <dl className="kv">
          <dt>Competition</dt>
          <dd>
            {data.result.competition} {data.result.season}
          </dd>
          <dt>State</dt>
          <dd>{data.result.state}</dd>
          <dt>Venue</dt>
          <dd>{data.result.venue ?? '—'}</dd>
          <dt>Attendance</dt>
          <dd>{data.result.attendance?.toLocaleString() ?? '—'}</dd>
          <dt>Officials</dt>
          <dd>{data.result.officials?.join(', ') || '—'}</dd>
        </dl>
      </Panel>

      <Panel
        title="Key events"
        state={data.key_events.length ? 'DATA' : 'NO_DATA'}
        reason="No goal, card or VAR decision was recorded up to this point."
      >
        <ul className="report-events">
          {data.key_events.map((event, index) => (
            <li key={index}>
              <span className="mono">{event.minute}</span>{' '}
              <strong>{event.event_type.replace(/_/g, ' ')}</strong>{' '}
              {event.player ? `— ${event.player}` : ''}{' '}
              {event.team ? <span className="faint">({event.team})</span> : null}{' '}
              {event.score && <span className="num">{event.score}</span>}
              {event.text && <div className="faint">{event.text}</div>}
            </li>
          ))}
        </ul>
      </Panel>

      <Panel title="Momentum" state="DATA">
        <p>{data.momentum.prose}</p>
        {data.momentum.largest_swings.length > 0 && (
          <ul className="report-events">
            {data.momentum.largest_swings.map((swing, index) => (
              <li key={index}>
                <span className="mono">{swing.minute}′</span>{' '}
                <span className={swing.change > 0 ? 'pos' : 'neg'}>
                  {swing.change > 0 ? '+' : ''}
                  {swing.change.toFixed(1)}
                </span>{' '}
                <span className="faint">
                  {swing.associated_events
                    .map((event) => event.event_type.replace(/_/g, ' '))
                    .join(', ') || 'no associated event'}{' '}
                  — {swing.note}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Shooting" state="DATA" note={data.shooting.note}>
        <p>
          {data.shooting.total_shots} shots recorded, {data.shooting.located} with a field
          position.
        </p>
        <dl className="kv">
          {Object.entries(data.shooting.by_outcome).map(([side, outcomes]) => (
            <div key={side} style={{ display: 'contents' }}>
              <dt>{side}</dt>
              <dd>
                {Object.entries(outcomes)
                  .map(([outcome, count]) => `${count} ${outcome.replace(/_/g, ' ')}`)
                  .join(', ') || '—'}
              </dd>
            </div>
          ))}
        </dl>
      </Panel>

      <Panel title="Passing" state="DATA">
        <p className="muted">{data.passing.reason}</p>
      </Panel>

      <Panel title="Market" state="DATA">
        <p>{data.market.prose}</p>
      </Panel>

      <Panel title="DivineLines view" state="DATA">
        <p>{data.model.prose}</p>
      </Panel>

      <Panel title="Limitations" state="DATA">
        <ul className="report-limits">
          {data.limitations.map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
