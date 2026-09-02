/* Market and model, side by side but never merged.

   Two things this panel refuses to blur:

   * best available price and consensus price are separate numbers. V3 proved
     how easily comparing a best-of-N entry against a consensus close
     manufactures positive CLV out of nothing.
   * a probability difference is a disagreement, not an edge. The staking line
     comes from the betting engine under model-health limits, and when it says
     0.00u the panel says 0.00u. */

import { Badge } from '../ui';
import { american, pct } from '../api';
import type { MarketPanel as MarketPanelData, MatchCentre } from './api';

const SELECTION_LABEL: Record<string, string> = {
  home: 'Home', draw: 'Draw', away: 'Away',
};

function PriceLine({
  label,
  snapshot,
  selections,
}: {
  label: string;
  snapshot: MarketPanelData['latest'];
  selections: string[];
}) {
  if (!snapshot) return null;
  return (
    <tr>
      <td className="faint">{label}</td>
      {selections.map((selection) => (
        <td key={selection} className="right num">
          {snapshot.consensus?.[selection]?.toFixed(2) ?? '—'}
        </td>
      ))}
      <td className="right faint num">{snapshot.books}</td>
    </tr>
  );
}

export function MarketPanel({ market }: { market: MarketPanelData }) {
  if (!market.available) {
    return <div className="empty">No market prices stored for this fixture.</div>;
  }
  const selections = market.selections ?? [];
  const latest = market.latest;

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th />
              {selections.map((selection) => (
                <th key={selection} className="right">
                  {SELECTION_LABEL[selection] ?? selection}
                </th>
              ))}
              <th className="right">Books</th>
            </tr>
          </thead>
          <tbody>
            <PriceLine label="Opening" snapshot={market.opening} selections={selections} />
            <PriceLine label="Latest" snapshot={latest} selections={selections} />
            {market.closing && (
              <PriceLine label="Closing" snapshot={market.closing} selections={selections} />
            )}
            <tr className="market-best">
              <td className="faint">Best available</td>
              {selections.map((selection) => (
                <td key={selection} className="right num">
                  {latest?.best?.[selection]?.toFixed(2) ?? '—'}
                </td>
              ))}
              <td />
            </tr>
            <tr>
              <td className="faint">No-vig</td>
              {selections.map((selection) => (
                <td key={selection} className="right num">
                  {latest?.novig?.[selection] != null
                    ? pct(latest.novig[selection], 1)
                    : '—'}
                </td>
              ))}
              <td />
            </tr>
          </tbody>
        </table>
      </div>

      <MovementChart market={market} />

      <div className="row" style={{ gap: 8 }}>
        <span className="faint">
          {market.snapshots} snapshots · {market.books} books ·{' '}
          {(market.sources ?? []).join(', ')}
        </span>
      </div>
    </div>
  );
}

/** Price movement over the snapshots we hold. Sparse by design: for most
    historical fixtures there is an open and a close and nothing in between,
    and pretending otherwise would draw a trend through two points. */
function MovementChart({ market }: { market: MarketPanelData }) {
  const series = market.series ?? [];
  const selections = market.selections ?? [];
  if (series.length < 2) {
    return (
      <div className="faint">
        {series.length} price observation{series.length === 1 ? '' : 's'} stored — not enough to
        draw a movement line.
      </div>
    );
  }

  const width = 620;
  const height = 130;
  const pad = { top: 10, right: 8, bottom: 18, left: 34 };
  const probs = series
    .flatMap((point) => selections.map((selection) => point.novig?.[selection]))
    .filter((value): value is number => value != null);
  if (!probs.length) return null;
  const min = Math.min(...probs) * 0.95;
  const max = Math.max(...probs) * 1.05;

  const x = (index: number) =>
    pad.left + (index / Math.max(series.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (value: number) =>
    pad.top + (1 - (value - min) / Math.max(max - min, 1e-6)) * (height - pad.top - pad.bottom);

  const colors = ['var(--series-1, var(--accent))', 'var(--neutral)', 'var(--warning)'];
  const dashes = ['none', '5 3', '2 3'];

  return (
    <div className="chart-host">
      <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img"
           aria-label="No-vig market probability across stored price snapshots">
        {selections.map((selection, index) => {
          const path = series
            .map((point, position) => {
              const value = point.novig?.[selection];
              if (value == null) return null;
              return `${position ? 'L' : 'M'}${x(position).toFixed(1)},${y(value).toFixed(1)}`;
            })
            .filter(Boolean)
            .join('');
          if (!path) return null;
          return (
            <path
              key={selection}
              d={path}
              fill="none"
              stroke={colors[index % colors.length]}
              strokeDasharray={dashes[index % dashes.length]}
              strokeWidth="1.8"
            />
          );
        })}
        {series.map((point, index) => (
          <text key={index} x={x(index)} y={height - 5} textAnchor="middle" className="axis-text">
            {point.phase}
          </text>
        ))}
      </svg>
      <div className="legend-row">
        {selections.map((selection, index) => (
          <span className="legend-key" key={selection}>
            <svg width="18" height="10" aria-hidden="true">
              <line
                x1="0" y1="5" x2="18" y2="5"
                stroke={colors[index % colors.length]}
                strokeDasharray={dashes[index % dashes.length]}
                strokeWidth="2"
              />
            </svg>
            <span>{SELECTION_LABEL[selection] ?? selection}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function ModelPanel({ data }: { data: MatchCentre }) {
  const { model, model_vs_market: comparison } = data;

  if (!model.available) {
    return (
      <div className="empty">
        <strong>No prediction recorded</strong>
        <span className="muted">{model.reason}</span>
      </div>
    );
  }

  const latest = Object.values(model.latest ?? {});
  const staked = latest.reduce((total, prediction) => total + (prediction.stake ?? 0), 0);

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="row" style={{ gap: 6 }}>
        <Badge tone="warn" title="This sport's model does not beat the market in its own walk-forward backtest.">
          model unproven
        </Badge>
        <span className="faint">
          {model.count} prediction{model.count === 1 ? '' : 's'} stored
          {model.superseded ? `, ${model.superseded} superseded` : ''}
        </span>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Selection</th>
              <th className="right">Model</th>
              <th className="right">Market</th>
              <th className="right">Difference</th>
              <th className="right">Price</th>
              <th className="right">Stake</th>
            </tr>
          </thead>
          <tbody>
            {(comparison.rows ?? []).map((row) => (
              <tr key={row.selection}>
                <td>{SELECTION_LABEL[row.selection] ?? row.selection}</td>
                <td className="right num">{pct(row.model_probability, 1)}</td>
                <td className="right num">{pct(row.market_probability, 1)}</td>
                <td className={`right num ${row.difference_points > 0 ? 'pos' : row.difference_points < 0 ? 'neg' : ''}`}>
                  {row.difference_points > 0 ? '+' : ''}
                  {row.difference_points.toFixed(1)}pp
                </td>
                <td className="right num">
                  {row.price ? `${row.price.toFixed(2)} (${american(row.price)})` : '—'}
                </td>
                <td className="right num">{(row.stake ?? 0).toFixed(2)}u</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {!comparison.available && <div className="faint">{comparison.reason}</div>}

      <div className={`banner ${staked > 0 ? 'banner-info' : 'banner-warn'}`}>
        {staked > 0
          ? `Total staked on this fixture: ${staked.toFixed(2)}u.`
          : 'No stake on this fixture. A probability difference is not an edge — staking is ' +
            'decided by the betting engine under model-health limits, and this model has not ' +
            'been shown to beat the market.'}
      </div>
    </div>
  );
}
