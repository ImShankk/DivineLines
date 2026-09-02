/* Momentum: home above the axis, away below, events marked on the timeline.

   Hand-rolled SVG, like the rest of the charts in this app — there is no
   charting dependency here and one chart is not a reason to add 200kB of one.

   The tooltip says "associated event", never "cause". The curve moved because
   that event entered a weighted sum; whether it changed the match is not
   something an event feed can establish. */

import { useMemo, useState } from 'react';
import type { Momentum, MomentumMarker } from './api';
import { eventLabel } from './api';

const WIDTH = 880;
const HEIGHT = 240;
const PAD = { top: 18, right: 16, bottom: 30, left: 44 };

/** Events worth a mark on the axis, and how each one is drawn. Shape is the
    primary encoding so the chart survives being read without colour. */
const MARKER_SHAPE: Record<string, 'goal' | 'card' | 'sub'> = {
  goal: 'goal',
  own_goal: 'goal',
  penalty_scored: 'goal',
  penalty_missed: 'goal',
  penalty_saved: 'goal',
  yellow_card: 'card',
  red_card: 'card',
  substitution: 'sub',
};

export function MomentumChart({
  momentum,
  homeName,
  awayName,
  onSelectMinute,
}: {
  momentum: Momentum;
  homeName: string;
  awayName: string;
  onSelectMinute?: (minute: number) => void;
}) {
  const [hover, setHover] = useState<{ x: number; y: number; minute: number } | null>(null);

  const geometry = useMemo(() => {
    const series = momentum.series ?? [];
    if (!series.length) return null;
    const maxMinute = Math.max(...series.map((point) => point.minute), 1);
    const extent = Math.max(
      10,
      ...series.map((point) => Math.abs(point.net)),
    );
    const innerWidth = WIDTH - PAD.left - PAD.right;
    const innerHeight = HEIGHT - PAD.top - PAD.bottom;
    const x = (minute: number) => PAD.left + (minute / maxMinute) * innerWidth;
    const y = (net: number) => PAD.top + innerHeight / 2 - (net / extent) * (innerHeight / 2);
    const zero = y(0);

    const path = series.map((p, i) => `${i ? 'L' : 'M'}${x(p.minute).toFixed(1)},${y(p.net).toFixed(1)}`).join('');
    const areaHome = `M${x(series[0].minute).toFixed(1)},${zero.toFixed(1)}` +
      series.map((p) => `L${x(p.minute).toFixed(1)},${y(Math.max(p.net, 0)).toFixed(1)}`).join('') +
      `L${x(series[series.length - 1].minute).toFixed(1)},${zero.toFixed(1)}Z`;
    const areaAway = `M${x(series[0].minute).toFixed(1)},${zero.toFixed(1)}` +
      series.map((p) => `L${x(p.minute).toFixed(1)},${y(Math.min(p.net, 0)).toFixed(1)}`).join('') +
      `L${x(series[series.length - 1].minute).toFixed(1)},${zero.toFixed(1)}Z`;

    return { series, maxMinute, extent, x, y, zero, path, areaHome, areaAway, innerWidth, innerHeight };
  }, [momentum.series]);

  if (!geometry) return null;

  const { series, maxMinute, x, y, zero, path, areaHome, areaAway } = geometry;
  const ticks = [0, 15, 30, 45, 60, 75, 90].filter((tick) => tick <= maxMinute + 5);

  const nearest = (clientMinute: number) =>
    series.reduce((best, point) =>
      Math.abs(point.minute - clientMinute) < Math.abs(best.minute - clientMinute) ? point : best,
    );

  const hovered = hover ? nearest(hover.minute) : null;
  const hoveredMarkers: MomentumMarker[] = hovered
    ? (momentum.markers ?? []).filter((marker) => Math.abs(marker.minute - hovered.minute) <= 0.75)
    : [];

  return (
    <div className="chart-host">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="chart"
        role="img"
        aria-label={`Momentum over time. ${homeName} above the axis, ${awayName} below.`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const px = ((event.clientX - rect.left) / rect.width) * WIDTH;
          const minute = ((px - PAD.left) / (WIDTH - PAD.left - PAD.right)) * maxMinute;
          if (minute < -2 || minute > maxMinute + 2) return setHover(null);
          setHover({ x: event.clientX - rect.left, y: event.clientY - rect.top, minute });
        }}
        onClick={() => {
          if (hovered && onSelectMinute) onSelectMinute(Math.round(hovered.minute));
        }}
      >
        <defs>
          <linearGradient id="momentum-home" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.42" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.04" />
          </linearGradient>
          <linearGradient id="momentum-away" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="var(--warning)" stopOpacity="0.42" />
            <stop offset="100%" stopColor="var(--warning)" stopOpacity="0.04" />
          </linearGradient>
        </defs>

        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={x(tick)} x2={x(tick)} y1={PAD.top} y2={HEIGHT - PAD.bottom}
              stroke="var(--border)" strokeDasharray="2 4"
            />
            <text x={x(tick)} y={HEIGHT - PAD.bottom + 15} textAnchor="middle" className="axis-text">
              {tick}′
            </text>
          </g>
        ))}

        <path d={areaHome} fill="url(#momentum-home)" />
        <path d={areaAway} fill="url(#momentum-away)" />
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={zero} y2={zero} stroke="var(--border-strong)" />
        <path d={path} fill="none" stroke="var(--text)" strokeWidth="1.6" strokeLinejoin="round" />

        <text x={PAD.left - 8} y={PAD.top + 10} textAnchor="end" className="axis-text">
          {homeName.slice(0, 12)}
        </text>
        <text x={PAD.left - 8} y={HEIGHT - PAD.bottom - 2} textAnchor="end" className="axis-text">
          {awayName.slice(0, 12)}
        </text>

        {(momentum.markers ?? []).map((marker, index) => {
          const shape = MARKER_SHAPE[marker.event_type];
          if (!shape) return null;
          const mx = x(marker.minute);
          const my = marker.home_away === 'home' ? PAD.top + 6 : HEIGHT - PAD.bottom - 6;
          return (
            <g key={`${marker.minute}-${index}`} className="momentum-marker">
              <line x1={mx} x2={mx} y1={my} y2={zero} stroke="var(--border-strong)" strokeWidth="1" />
              {shape === 'goal' && <circle cx={mx} cy={my} r="5" fill="var(--positive)" />}
              {shape === 'card' && (
                <rect
                  x={mx - 3} y={my - 5} width="6" height="10" rx="1"
                  fill={marker.event_type === 'red_card' ? 'var(--negative)' : 'var(--warning)'}
                />
              )}
              {shape === 'sub' && (
                <g stroke="var(--text-muted)" strokeWidth="1.6" fill="none">
                  <path d={`M${mx - 4},${my - 2}h8l-2.5,-2.5M${mx + 4},${my + 2}h-8l2.5,2.5`} />
                </g>
              )}
              <title>
                {`${marker.clock_display ?? `${marker.minute}′`} — ${eventLabel(marker.event_type)}${
                  marker.player_name ? `: ${marker.player_name}` : ''
                }`}
              </title>
            </g>
          );
        })}

        {hovered && (
          <line
            x1={x(hovered.minute)} x2={x(hovered.minute)} y1={PAD.top} y2={HEIGHT - PAD.bottom}
            stroke="var(--accent)" strokeWidth="1"
          />
        )}
        {hovered && <circle cx={x(hovered.minute)} cy={y(hovered.net)} r="3.5" fill="var(--accent)" />}
      </svg>

      {hover && hovered && (
        <div
          className="chart-tip"
          style={{ left: Math.min(hover.x + 12, WIDTH - 200), top: Math.max(hover.y - 10, 0) }}
        >
          <div className="mono">{Math.round(hovered.minute)}′</div>
          <div>
            <span className="muted">net </span>
            <span className="num">{hovered.net > 0 ? '+' : ''}{hovered.net.toFixed(1)}</span>
            <span className="faint"> ({hovered.net >= 0 ? homeName : awayName})</span>
          </div>
          {hoveredMarkers.map((marker, index) => (
            <div key={index} className="faint">
              Associated event: {eventLabel(marker.event_type)}
              {marker.player_name ? ` — ${marker.player_name}` : ''}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
