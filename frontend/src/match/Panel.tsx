/* Panel: the one place a visualisation decides what to render.

   Every panel in the Match Centre is in exactly one state and shows it. The
   important case is NO_DATA: it always carries the *reason*, because "no
   passing coordinates are published for this competition" is information a
   reader can act on and an empty pitch is not. */

import type { ReactNode } from 'react';
import { Badge, Card } from '../ui';
import type { PanelState } from './api';

export function Panel({
  title,
  state,
  reason,
  actions,
  note,
  legend,
  children,
  className = '',
}: {
  title: ReactNode;
  state: PanelState;
  reason?: string | null;
  actions?: ReactNode;
  note?: ReactNode;
  legend?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <Card
      className={className}
      title={
        <span className="row" style={{ gap: 8 }}>
          {title}
          {state === 'STALE' && <Badge tone="warn">stale</Badge>}
          {state === 'NO_DATA' && <Badge tone="neutral">no data</Badge>}
        </span>
      }
      actions={actions}
      note={note}
    >
      {state === 'LOADING' && <div className="panel-skeleton" aria-busy="true" />}
      {state === 'ERROR' && (
        <div className="empty">
          <strong>Could not load</strong>
          <span className="muted">{reason ?? 'Unknown error'}</span>
        </div>
      )}
      {state === 'NO_DATA' && (
        <div className="empty">
          <strong>Not available for this match</strong>
          <span className="muted">{reason ?? 'No source provides this data.'}</span>
        </div>
      )}
      {(state === 'DATA' || state === 'STALE') && (
        <>
          {children}
          {legend && <div className="legend-row">{legend}</div>}
        </>
      )}
    </Card>
  );
}

/** A legend entry. Shape carries the meaning as well as colour, because a
    reader who cannot separate red from green still has to read the chart. */
export function LegendKey({
  label,
  color,
  shape = 'dot',
}: {
  label: string;
  color: string;
  shape?: 'dot' | 'ring' | 'square' | 'line' | 'cross';
}) {
  return (
    <span className="legend-key">
      <svg width="13" height="13" aria-hidden="true">
        {shape === 'dot' && <circle cx="6.5" cy="6.5" r="4.5" fill={color} />}
        {shape === 'ring' && (
          <circle cx="6.5" cy="6.5" r="4" fill="none" stroke={color} strokeWidth="2" />
        )}
        {shape === 'square' && <rect x="2" y="2" width="9" height="9" fill={color} />}
        {shape === 'line' && (
          <line x1="0" y1="6.5" x2="13" y2="6.5" stroke={color} strokeWidth="2.5" />
        )}
        {shape === 'cross' && (
          <g stroke={color} strokeWidth="2">
            <line x1="2" y1="2" x2="11" y2="11" />
            <line x1="11" y1="2" x2="2" y2="11" />
          </g>
        )}
      </svg>
      <span>{label}</span>
    </span>
  );
}

/** Where a number came from, on demand rather than always on screen. */
export function Provenance({
  source,
  observed,
  derived,
}: {
  source?: string | null;
  observed?: string | null;
  derived?: string | null;
}) {
  const parts = [
    source ? `source: ${source}` : null,
    observed ? `observed: ${observed}` : null,
    derived ? `derived: ${derived}` : null,
  ].filter(Boolean);
  if (!parts.length) return null;
  return (
    <span className="faint" title={parts.join(' · ')} style={{ fontSize: '0.72rem' }}>
      {parts.join(' · ')}
    </span>
  );
}
