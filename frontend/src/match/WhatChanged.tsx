/* "What changed?" — the information timeline, reusing the V3 attribution
   endpoint rather than building a second one.

   The wording matters. The platform can say what changed and what arrived
   just before it. It cannot say one caused the other, so every attribution is
   labelled as co-occurrence and a movement with nothing before it is reported
   as unexplained rather than pinned on whatever happened to be nearby. */

import { api, shortDate, type TimelineEntry } from '../api';
import { Badge, Empty, ErrorState, Loading, useAsync } from '../ui';

/** One material prediction movement paired with whatever arrived before it. */
interface Attribution {
  timestamp: string;
  market: string | null;
  selection: string | null;
  move: number | null;
  candidate_causes: { timestamp: string; kind: string; label: string }[];
  explained: boolean;
  note: string;
}

const KIND_TONE: Record<string, 'accent' | 'warn' | 'neutral' | 'pos'> = {
  prediction: 'accent',
  market: 'neutral',
  lineup: 'warn',
  information: 'pos',
};

export function WhatChanged({ gameUid }: { gameUid: string }) {
  const { data, error, loading, refresh } = useAsync(() => api.timeline(gameUid), [gameUid]);

  if (loading) return <Loading label="Loading information timeline" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  const entries: TimelineEntry[] = data?.timeline ?? [];
  if (!entries.length) {
    return (
      <Empty title="Nothing recorded before this fixture">
        No prediction, price movement or lineup observation is stored for this match, so there
        is no belief change to describe.
      </Empty>
    );
  }

  const attributions: Attribution[] = data?.attributions ?? [];

  return (
    <div className="stack" style={{ gap: 12 }}>
      <ol className="change-list">
        {entries.map((entry, index) => (
          <li key={index} className="change-row">
            <span className="change-time mono">
              {entry.timestamp
                ? new Date(entry.timestamp).toLocaleString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                : '—'}
            </span>
            <Badge tone={KIND_TONE[entry.kind] ?? 'neutral'}>{entry.kind}</Badge>
            <span className="change-label">{entry.label}</span>
            {entry.detail?.move != null && (
              <span className={`num ${entry.detail.move > 0 ? 'pos' : 'neg'}`}>
                {entry.detail.move > 0 ? '+' : ''}
                {(entry.detail.move * 100).toFixed(1)}pp
              </span>
            )}
          </li>
        ))}
      </ol>

      {attributions.length > 0 && (
        <div className="stack" style={{ gap: 6 }}>
          <div className="lineup-subhead">Movements and what preceded them</div>
          {attributions.map((attribution, index) => (
            <div key={index} className="attribution">
              <div className="row" style={{ gap: 8 }}>
                <span className="mono faint">{shortDate(attribution.timestamp)}</span>
                <span>
                  {attribution.market}/{attribution.selection}
                </span>
                {attribution.move != null && (
                  <span className={`num ${attribution.move > 0 ? 'pos' : 'neg'}`}>
                    {attribution.move > 0 ? '+' : ''}
                    {(attribution.move * 100).toFixed(1)}pp
                  </span>
                )}
              </div>
              <div className="faint">{attribution.note}</div>
              {attribution.candidate_causes?.length > 0 && (
                <ul className="attribution-causes">
                  {attribution.candidate_causes.map((cause, position) => (
                    <li key={position} className="faint">
                      {cause.kind}: {cause.label}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}

      {data?.market_vs_model?.available && (
        <div className="banner banner-info">
          {data.market_vs_model.interpretation}. Model moved{' '}
          {(data.market_vs_model.model_move * 100).toFixed(1)}pp, market moved{' '}
          {(data.market_vs_model.market_move * 100).toFixed(1)}pp
          {data.market_vs_model.model_moved_first
            ? '; the model observation came first.'
            : '; the market observation came first.'}{' '}
          Order of observation, not evidence of leadership — the two feeds are sampled at
          different rates.
        </div>
      )}
    </div>
  );
}
