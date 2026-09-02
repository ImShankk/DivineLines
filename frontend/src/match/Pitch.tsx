/* Pitch rendering: one coordinate frame, three views built on it.

   The server hands over metres in a fixed frame — 105 × 68, home attacking
   right. This file scales that to SVG units and nothing else transforms
   anything, so a component can never invent its own idea of which way a team
   is playing. */

import { useState } from 'react';
import { SHOT_LEGEND, type EventDensity, type ShotPoint } from './api';

const LENGTH = 105;
const WIDTH = 68;
const MARGIN = 3;

/** The markings, drawn once. Everything else layers on top. */
export function PitchSurface({ children, ariaLabel }: { children?: React.ReactNode; ariaLabel: string }) {
  return (
    <svg
      viewBox={`${-MARGIN} ${-MARGIN} ${LENGTH + MARGIN * 2} ${WIDTH + MARGIN * 2}`}
      className="pitch"
      role="img"
      aria-label={ariaLabel}
    >
      <rect x={-MARGIN} y={-MARGIN} width={LENGTH + MARGIN * 2} height={WIDTH + MARGIN * 2}
            className="pitch-ground" />
      <g className="pitch-lines">
        <rect x="0" y="0" width={LENGTH} height={WIDTH} fill="none" />
        <line x1={LENGTH / 2} y1="0" x2={LENGTH / 2} y2={WIDTH} />
        <circle cx={LENGTH / 2} cy={WIDTH / 2} r="9.15" fill="none" />
        <circle cx={LENGTH / 2} cy={WIDTH / 2} r="0.6" className="pitch-spot" />
        {/* Penalty and six-yard boxes, both ends. */}
        <rect x="0" y={(WIDTH - 40.3) / 2} width="16.5" height="40.3" fill="none" />
        <rect x={LENGTH - 16.5} y={(WIDTH - 40.3) / 2} width="16.5" height="40.3" fill="none" />
        <rect x="0" y={(WIDTH - 18.3) / 2} width="5.5" height="18.3" fill="none" />
        <rect x={LENGTH - 5.5} y={(WIDTH - 18.3) / 2} width="5.5" height="18.3" fill="none" />
        <rect x={-1.8} y={(WIDTH - 7.32) / 2} width="1.8" height="7.32" fill="none" />
        <rect x={LENGTH} y={(WIDTH - 7.32) / 2} width="1.8" height="7.32" fill="none" />
        <circle cx="11" cy={WIDTH / 2} r="0.6" className="pitch-spot" />
        <circle cx={LENGTH - 11} cy={WIDTH / 2} r="0.6" className="pitch-spot" />
      </g>
      {children}
    </svg>
  );
}

/** Indexed form of the shared legend, so the map and its key cannot disagree. */
const OUTCOME_STYLE = Object.fromEntries(
  SHOT_LEGEND.map((entry) => [entry.outcome, entry]),
) as Record<ShotPoint['outcome'], (typeof SHOT_LEGEND)[number]>;

/** Shot map. Real coordinates from the event feed; unlocated shots are
    counted in the caption rather than placed somewhere convenient. */
export function ShotMap({
  points,
  totalShots,
  homeName,
  awayName,
  side,
}: {
  points: ShotPoint[];
  totalShots: number;
  homeName: string;
  awayName: string;
  side: 'both' | 'home' | 'away';
}) {
  const [hover, setHover] = useState<ShotPoint | null>(null);
  const visible = side === 'both' ? points : points.filter((p) => p.home_away === side);

  return (
    <div className="pitch-host">
      <PitchSurface
        ariaLabel={`Shot locations. ${homeName} attacks right, ${awayName} attacks left. ${visible.length} shots plotted.`}
      >
        <g>
          {visible.map((shot, index) => {
            const style = OUTCOME_STYLE[shot.outcome];
            const radius = shot.outcome === 'goal' ? 1.9 : 1.5;
            return (
              <g
                key={shot.event_row_id ?? index}
                className={`shot ${style.className} ${shot.home_away === 'home' ? 'shot-home' : 'shot-away'}`}
                onMouseEnter={() => setHover(shot)}
                onMouseLeave={() => setHover(null)}
                tabIndex={0}
                onFocus={() => setHover(shot)}
                onBlur={() => setHover(null)}
              >
                {style.shape === 'filled' && <circle cx={shot.x} cy={shot.y} r={radius} />}
                {style.shape === 'ring' && (
                  <circle cx={shot.x} cy={shot.y} r={radius} fill="none" strokeWidth="0.7" />
                )}
                {style.shape === 'square' && (
                  <rect x={shot.x - radius} y={shot.y - radius} width={radius * 2} height={radius * 2} />
                )}
                {style.shape === 'cross' && (
                  <g strokeWidth="0.7">
                    <line x1={shot.x - radius} y1={shot.y - radius} x2={shot.x + radius} y2={shot.y + radius} />
                    <line x1={shot.x + radius} y1={shot.y - radius} x2={shot.x - radius} y2={shot.y + radius} />
                  </g>
                )}
                <title>
                  {`${shot.clock_display ?? `${shot.minute}′`} ${shot.player_name ?? 'unknown'} — ${style.label}, ${shot.distance_m.toFixed(0)}m`}
                </title>
              </g>
            );
          })}
        </g>
        <text x="4" y={WIDTH + 1.6} className="pitch-label">
          ← {awayName.slice(0, 16)} attacks
        </text>
        <text x={LENGTH - 4} y={WIDTH + 1.6} textAnchor="end" className="pitch-label">
          {homeName.slice(0, 16)} attacks →
        </text>
      </PitchSurface>

      {hover && (
        <div className="pitch-tip">
          <strong>{hover.player_name ?? 'Unknown player'}</strong>
          <div className="muted">
            {hover.clock_display ?? `${hover.minute}′`} · {OUTCOME_STYLE[hover.outcome].label} ·{' '}
            {hover.distance_m.toFixed(0)}m
          </div>
          {hover.text && <div className="faint">{hover.text}</div>}
        </div>
      )}

      <p className="pitch-caption faint">
        {visible.length} of {totalShots} shots plotted.{' '}
        {totalShots > points.length
          ? `${totalShots - points.length} carry no field position in the feed and are counted in the statistics only.`
          : 'Every recorded shot carries a field position.'}
      </p>
    </div>
  );
}

/** Event-location density. Explicitly not tracking, and it says so. */
export function EventHeatmap({
  density,
  homeName,
  awayName,
}: {
  density: EventDensity;
  homeName: string;
  awayName: string;
}) {
  const cellWidth = LENGTH / density.columns;
  const cellHeight = WIDTH / density.rows;
  const peak = Math.max(density.peak, 1);

  return (
    <div className="pitch-host">
      <PitchSurface
        ariaLabel={`Density of recorded event locations. ${homeName} attacks right, ${awayName} attacks left.`}
      >
        <g className="heat-grid">
          {density.grid.map((row, rowIndex) =>
            row.map((count, columnIndex) => {
              if (!count) return null;
              const intensity = count / peak;
              return (
                <rect
                  key={`${rowIndex}-${columnIndex}`}
                  x={columnIndex * cellWidth}
                  y={rowIndex * cellHeight}
                  width={cellWidth}
                  height={cellHeight}
                  fill="var(--accent)"
                  opacity={0.08 + intensity * 0.62}
                >
                  <title>{`${count} event${count === 1 ? '' : 's'} recorded here`}</title>
                </rect>
              );
            }),
          )}
        </g>
      </PitchSurface>
      <p className="pitch-caption faint">
        {density.events_located} of {density.events_considered} events carry a position. Densest
        cell: {density.peak} events. {density.note}
      </p>
    </div>
  );
}

/** Average positions from the lineup's formation slots.

    This is a *formation diagram*, not measured positions: the feed publishes
    a formation and a slot number per player, and nothing about where anyone
    actually stood. Labelled accordingly rather than dressed up as a shape
    map. */
export function FormationBoard({
  entries,
  side,
  teamName,
}: {
  entries: { player_name: string | null; jersey: string | null; formation_place: string | null; position_group: string | null; subbed_out: boolean }[];
  side: 'home' | 'away';
  teamName: string;
}) {
  const slots = entries
    .filter((entry) => entry.formation_place)
    .sort((a, b) => Number(a.formation_place) - Number(b.formation_place));
  if (!slots.length) return null;

  // Lay players out by position group in bands from their own goal outward.
  const bands: Record<string, number> = {
    goalkeeper: 0.05, defender: 0.26, midfielder: 0.52, forward: 0.78,
  };
  const grouped = new Map<string, typeof slots>();
  for (const entry of slots) {
    const group = entry.position_group ?? 'midfielder';
    grouped.set(group, [...(grouped.get(group) ?? []), entry]);
  }

  return (
    <PitchSurface ariaLabel={`${teamName} formation`}>
      {[...grouped.entries()].map(([group, members]) =>
        members.map((entry, index) => {
          const depth = bands[group] ?? 0.5;
          const x = side === 'home' ? depth * LENGTH : (1 - depth) * LENGTH;
          const y = ((index + 1) / (members.length + 1)) * WIDTH;
          return (
            <g key={`${group}-${index}`} className={`formation-node formation-${side}`}>
              <circle cx={x} cy={y} r="3.1" />
              <text x={x} y={y + 1.1} textAnchor="middle" className="formation-jersey">
                {entry.jersey ?? ''}
              </text>
              <text x={x} y={y + 6.4} textAnchor="middle" className="formation-name">
                {(entry.player_name ?? '').split(' ').slice(-1)[0]}
              </text>
              <title>{`${entry.player_name ?? 'unknown'}${entry.subbed_out ? ' (substituted)' : ''}`}</title>
            </g>
          );
        }),
      )}
    </PitchSurface>
  );
}
