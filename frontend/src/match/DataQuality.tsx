/* What this match can actually support, component by component.

   Deliberately not a single percentage. A match with a complete event feed and
   no price history is a very different thing from one with prices and no
   events, and one number hides exactly that difference. */

import { Badge } from '../ui';
import type { MatchCentre, QualityComponentState } from './api';

const MARK: Record<QualityComponentState['state'], { glyph: string; tone: 'pos' | 'warn' | 'neutral' }> = {
  present: { glyph: '✓', tone: 'pos' },
  partial: { glyph: '~', tone: 'warn' },
  absent: { glyph: '–', tone: 'neutral' },
};

export function DataQuality({ quality, replay }: { quality: MatchCentre['quality']; replay: MatchCentre['replay'] }) {
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row" style={{ gap: 8 }}>
        <span className="faint">Data quality</span>
        <Badge tone={quality.grade === 'high' ? 'pos' : quality.grade === 'none' ? 'neg' : 'warn'}>
          {quality.grade}
        </Badge>
        <span className="faint">
          {quality.present} present · {quality.partial} partial · {quality.absent} absent
        </span>
      </div>

      <ul className="quality-list">
        {quality.components.map((component) => {
          const mark = MARK[component.state];
          return (
            <li key={component.name} className={`quality-row quality-${component.state}`}>
              <span className={`quality-mark ${mark.tone}`} aria-hidden="true">
                {mark.glyph}
              </span>
              <span className="quality-label">{component.label}</span>
              <span className="quality-detail faint">{component.detail}</span>
            </li>
          );
        })}
      </ul>

      {replay.retrospective_events && (
        <div className="banner banner-warn">
          The event stream for this match was ingested after full time. Replaying it
          reconstructs what happened by a given minute — not what the platform knew at that
          minute.
        </div>
      )}
    </div>
  );
}

export function StandingsPanel({ standings }: { standings: MatchCentre['standings'] }) {
  if (!standings.available) {
    return <div className="empty">{standings.reason ?? 'No table available.'}</div>;
  }
  const highlight = new Set((standings.highlight ?? []).filter(Boolean) as string[]);

  return (
    <div className="table-wrap">
      <table className="compact">
        <thead>
          <tr>
            <th>#</th>
            <th>Team</th>
            <th className="right">P</th>
            <th className="right">W</th>
            <th className="right">D</th>
            <th className="right">L</th>
            <th className="right">GD</th>
            <th className="right">Pts</th>
          </tr>
        </thead>
        <tbody>
          {(standings.table ?? []).map((row) => (
            <tr key={row.team_uid} className={highlight.has(row.team_uid) ? 'row-highlight' : ''}>
              <td className="num faint">{row.position}</td>
              <td>{row.team_name}</td>
              <td className="right num">{row.played}</td>
              <td className="right num">{row.won}</td>
              <td className="right num">{row.drawn}</td>
              <td className="right num">{row.lost}</td>
              <td className="right num">
                {row.goal_difference > 0 ? '+' : ''}
                {row.goal_difference}
              </td>
              <td className="right num" style={{ fontWeight: 600 }}>
                {row.points}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
