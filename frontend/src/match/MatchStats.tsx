/* Match statistics as a two-sided comparison.

   A bar per metric rather than two columns of large numbers: the interesting
   thing is the gap, and a shared bar shows it without the reader doing
   arithmetic. Values are printed at both ends so the chart is still readable
   when the bar is not.

   When the view is a replay, the panel says which metrics it has recounted
   from events and which it is withholding, because the stored box score is a
   full-time figure and showing it at minute 32 would be wrong in the most
   convincing way possible. */

import { Badge } from '../ui';
import type { StatComparison } from './api';

function format(value: number | null, kind: string) {
  if (value === null || value === undefined) return '—';
  if (kind === 'percent' || kind === 'ratio') return `${value.toFixed(1)}%`;
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function MatchStats({
  comparisons,
  homeName,
  awayName,
  basis,
  unavailable = [],
}: {
  comparisons: StatComparison[];
  homeName: string;
  awayName: string;
  basis?: string;
  unavailable?: string[];
}) {
  if (!comparisons.length) {
    return <div className="empty">No box-score statistics published for this match.</div>;
  }

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="stat-head">
        <span>{homeName}</span>
        {basis && <span className="faint">{basis}</span>}
        <span className="right">{awayName}</span>
      </div>

      {comparisons.map((row) => {
        const share = row.home_share ?? 0.5;
        const homeLeads = (row.home ?? 0) > (row.away ?? 0);
        return (
          <div className="stat-row" key={row.stat}>
            <span className={`stat-value num ${homeLeads ? 'stat-lead' : ''}`}>
              {row.home_display ?? format(row.home, row.kind)}
            </span>
            <span className="stat-track" role="img"
                  aria-label={`${row.label}: ${homeName} ${format(row.home, row.kind)}, ${awayName} ${format(row.away, row.kind)}`}>
              <span className="stat-label">{row.label}</span>
              <span className="stat-bar">
                <span className="stat-bar-home" style={{ width: `${share * 100}%` }} />
                <span className="stat-bar-away" style={{ width: `${(1 - share) * 100}%` }} />
              </span>
            </span>
            <span className={`stat-value num right ${!homeLeads && (row.away ?? 0) > (row.home ?? 0) ? 'stat-lead' : ''}`}>
              {row.away_display ?? format(row.away, row.kind)}
            </span>
          </div>
        );
      })}

      {unavailable.length > 0 && (
        <div className="row" style={{ gap: 6, marginTop: 4 }}>
          <span className="faint">Withheld at this replay position:</span>
          {unavailable.map((stat) => (
            <Badge key={stat} tone="neutral">
              {stat}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
