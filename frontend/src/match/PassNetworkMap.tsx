/* The passing network.

   There is nothing to draw. No source configured in this platform publishes
   pass events — ESPN's feed carries `totalPasses` and `accuratePasses` per
   team and nothing about who passed to whom or from where.

   So this component renders an empty pitch with the reason on it, plus the
   aggregate pass numbers that *do* exist and a list of exactly what a provider
   would have to supply. That is a more useful screen than a plausible-looking
   network built out of nothing, and it is the only honest one. */

import { Panel } from './Panel';
import { PitchSurface } from './Pitch';
import type { PassingPanel } from './api';

export function PassNetworkMap({
  passing,
  homeName,
  awayName,
}: {
  passing: PassingPanel;
  homeName: string;
  awayName: string;
}) {
  const totals = passing.aggregate_totals ?? {};
  const hasTotals = Object.keys(totals).length > 0;

  return (
    <Panel
      title="Passing network"
      state="DATA"
      note={passing.note}
    >
      <div className="pitch-host pass-empty">
        <PitchSurface ariaLabel="Passing network unavailable: no pass coordinates are published for this competition.">
          <g className="pass-empty-overlay">
            <rect x="0" y="0" width="105" height="68" />
          </g>
        </PitchSurface>
        <div className="pass-empty-message">
          <strong>No pass coordinates for this competition</strong>
          <p className="muted">{passing.reason}</p>
          <div className="pass-requires">
            <span className="faint">A provider would need to supply:</span>
            <ul>
              {passing.requires.map((requirement) => (
                <li key={requirement}>{requirement}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {hasTotals && (
        <div className="pass-totals">
          <div className="pass-total-head">
            <span className="faint">
              What the feed does publish — team totals only, no per-pass detail
            </span>
          </div>
          <table className="pass-total-table">
            <thead>
              <tr>
                <th />
                <th className="right">Passes</th>
                <th className="right">Accurate</th>
                <th className="right">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {(['home', 'away'] as const).map((side) => {
                const row = totals[side];
                if (!row) return null;
                const accuracy = row.passPct != null
                  ? row.passPct <= 1 ? row.passPct * 100 : row.passPct
                  : null;
                return (
                  <tr key={side}>
                    <td>{side === 'home' ? homeName : awayName}</td>
                    <td className="right num">{row.totalPasses ?? '—'}</td>
                    <td className="right num">{row.accuratePasses ?? '—'}</td>
                    <td className="right num">
                      {accuracy != null ? `${accuracy.toFixed(0)}%` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
