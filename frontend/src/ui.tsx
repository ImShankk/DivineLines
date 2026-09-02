/* Shared UI primitives. Small, composable, and deliberately plain: the data
   should carry the page, not the chrome. */

import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import { ApiError } from './api';

export function Card({
  title,
  actions,
  children,
  note,
  className = '',
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  note?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div className="card-title">{title}</div>
          {actions && <div className="row">{actions}</div>}
        </header>
      )}
      <div className="card-body">{children}</div>
      {note && <div className="card-note">{note}</div>}
    </section>
  );
}

export function Tile({
  label,
  value,
  sub,
  tone = 'neutral',
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: 'neutral' | 'pos' | 'neg' | 'warn';
}) {
  const toneClass = tone === 'pos' ? 'pos' : tone === 'neg' ? 'neg' : tone === 'warn' ? 'warn' : '';
  return (
    <div className="card tile">
      <div className="tile-label">{label}</div>
      <div className={`tile-value ${toneClass}`}>{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = 'neutral',
  title,
}: {
  children: ReactNode;
  tone?: 'neutral' | 'pos' | 'neg' | 'warn' | 'accent';
  title?: string;
}) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  );
}

/** Alert flags carry meaning, so each one gets a tone and an explanation. */
const FLAG_META: Record<string, { tone: 'pos' | 'neg' | 'warn' | 'accent' | 'neutral'; help: string }> = {
  HIGH_EV: { tone: 'pos', help: 'Model edge of 5 percentage points or more over the no-vig market.' },
  MODEL_OUTLIER: {
    tone: 'neg',
    help: 'The model disagrees with the market far more than usual. Treated as a likely model error, not an opportunity — this selection is excluded from staking.',
  },
  LOW_DATA_QUALITY: { tone: 'warn', help: 'Data quality below 60/100: inputs are stale or incomplete.' },
  STALE_ODDS: { tone: 'warn', help: 'The newest price snapshot is over an hour old.' },
  NO_MARKET_PRICE: { tone: 'neutral', help: 'No bookmaker price is stored for this game, so no edge can be computed.' },
  INJURY_UNCERTAINTY: { tone: 'warn', help: 'Uncertain availability materially moves the expected margin.' },
  MODEL_DISAGREEMENT: { tone: 'warn', help: 'Ensemble components disagree; confidence is reduced.' },
  EARLY_SEASON: { tone: 'neutral', help: 'Few games played this season — form estimates lean on priors.' },
  LINEUP_NOT_CONFIRMED: { tone: 'neutral', help: 'Starting lineups are not confirmed yet.' },
  EXCLUDED_BY_RISK_LIMITS: { tone: 'neutral', help: 'Portfolio limits scaled this stake to zero.' },
  MODEL_UNPROVEN: {
    tone: 'warn',
    help: "This sport's model does not beat the market in its own walk-forward backtest, so any edge shown is unproven.",
  },
  PROMOTED_TEAM: {
    tone: 'warn',
    help: 'A club with little or no history in this division is involved — the least reliable input the model has. Confidence is reduced accordingly.',
  },
};

export function FlagList({ flags }: { flags: string[] }) {
  if (!flags?.length) return <span className="faint">—</span>;
  return (
    <span className="row" style={{ gap: 4 }}>
      {flags.map((flag) => {
        const meta = FLAG_META[flag] ?? {
          tone: 'neutral' as const,
          help: flag.startsWith('game_cap') || flag.startsWith('team_cap') || flag.startsWith('slate_cap') || flag.startsWith('sport_cap')
            ? 'Stake was scaled down by a portfolio exposure limit.'
            : flag,
        };
        const label = flag.includes(':') ? flag.split(':')[0] : flag;
        return (
          <Badge key={flag} tone={meta.tone} title={meta.help}>
            {label.replace(/_/g, ' ').toLowerCase()}
          </Badge>
        );
      })}
    </span>
  );
}

export function StateDot({ state }: { state: string }) {
  const cls =
    state === 'fresh' || state === 'ok'
      ? 'dot-ok'
      : state === 'aging' || state === 'degraded'
        ? 'dot-warn'
        : state === 'missing' || state === 'stale' || state === 'error' || state === 'down'
          ? 'dot-bad'
          : 'dot-idle';
  return <span className={`dot ${cls}`} title={state} />;
}

export function Meter({ value, max = 1 }: { value: number; max?: number }) {
  const ratio = Math.max(0, Math.min(1, value / max));
  const tone = ratio >= 0.7 ? 'good' : ratio >= 0.4 ? 'mid' : 'bad';
  return (
    <span className={`meter ${tone}`} title={`${(ratio * 100).toFixed(0)}%`}>
      <span style={{ width: `${ratio * 100}%` }} />
    </span>
  );
}

/** Model probability against market probability, on one shared scale. */
export function ProbabilityBar({ model, market }: { model: number; market?: number | null }) {
  return (
    <span
      className="row"
      style={{ gap: 6, minWidth: 130 }}
      title={`model ${(model * 100).toFixed(1)}%${market != null ? ` · market ${(market * 100).toFixed(1)}%` : ''}`}
    >
      <span className="probbar" style={{ flexDirection: 'column', gap: 2, height: 'auto', background: 'none' }}>
        <span className="probbar">
          <span className="p-model" style={{ width: `${model * 100}%` }} />
        </span>
        {market != null && (
          <span className="probbar">
            <span className="p-market" style={{ width: `${market * 100}%` }} />
          </span>
        )}
      </span>
    </span>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export function Banner({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'warn' | 'bad';
  children: ReactNode;
}) {
  return <div className={`banner banner-${tone}`}>{children}</div>;
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="empty">
      <span className="spinner" /> <span style={{ marginLeft: 8 }}>{label}…</span>
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: Error | null; onRetry?: () => void }) {
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : 'Unexpected error';
  return (
    <Banner tone="bad">
      <div>
        <div style={{ fontWeight: 600, marginBottom: 3 }}>Could not load this view</div>
        <div className="muted">{message}</div>
        {onRetry && (
          <button className="ghost" style={{ marginTop: 6, paddingLeft: 0 }} onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    </Banner>
  );
}

/** Minimal async data hook with explicit loading/error/refresh states. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    loader()
      .then((value) => {
        if (!cancelled) setData(value);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, refresh: () => setNonce((n) => n + 1) };
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button
          key={option.value}
          className={option.value === value ? 'active' : ''}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      {label}
      {children}
    </label>
  );
}

/** Table rendered from a column spec, so every page formats numbers alike. */
export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  numeric?: boolean;
  sortValue?: (row: T) => number | string;
}

export function DataTable<T>({
  columns,
  rows,
  emptyLabel = 'No rows',
  onRowClick,
  initialSort,
}: {
  columns: Column<T>[];
  rows: T[];
  emptyLabel?: string;
  onRowClick?: (row: T) => void;
  initialSort?: { key: string; direction: 'asc' | 'desc' };
}) {
  const [sort, setSort] = useState(initialSort);

  if (!rows.length) return <Empty title={emptyLabel} />;

  const column = columns.find((c) => c.key === sort?.key);
  const sorted = column?.sortValue
    ? [...rows].sort((a, b) => {
        const left = column.sortValue!(a);
        const right = column.sortValue!(b);
        const cmp = typeof left === 'number' && typeof right === 'number'
          ? left - right
          : String(left).localeCompare(String(right));
        return sort?.direction === 'asc' ? cmp : -cmp;
      })
    : rows;

  return (
    <div className="table-wrap">
      <table className="dl">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className={col.numeric ? 'num' : ''}
                style={{ cursor: col.sortValue ? 'pointer' : 'default' }}
                onClick={() =>
                  col.sortValue &&
                  setSort((current) =>
                    current?.key === col.key
                      ? { key: col.key, direction: current.direction === 'asc' ? 'desc' : 'asc' }
                      : { key: col.key, direction: 'desc' },
                  )
                }
              >
                {col.header}
                {sort?.key === col.key ? (sort.direction === 'asc' ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, index) => (
            <tr
              key={index}
              className={onRowClick ? 'clickable' : ''}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((col) => (
                <td key={col.key} className={col.numeric ? 'num' : ''}>
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
