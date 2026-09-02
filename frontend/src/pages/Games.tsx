/* Games: schedule, results, and a per-game view with line movement. */

import { useState } from 'react';
import {
  api,
  american,
  num,
  pct,
  pctPoints,
  shortDate,
  timeAgo,
  type Game,
  type Sport,
  type TimelineEntry,
} from '../api';
import { LineChart, Legend } from '../charts';
import {
  Badge,
  Card,
  DataTable,
  Empty,
  ErrorState,
  Field,
  Loading,
  Segmented,
  Tile,
  useAsync,
  type Column,
} from '../ui';

export function GamesPage({
  onOpenGame,
  selected,
  onClose,
}: {
  onOpenGame: (uid: string) => void;
  selected: string | null;
  onClose: () => void;
}) {
  const [sport, setSport] = useState<Sport>('nba');
  const [status, setStatus] = useState<'scheduled' | 'final'>('final');
  const [days, setDays] = useState(14);
  const { data, error, loading, refresh } = useAsync(
    () => api.games(sport, status, days),
    [sport, status, days],
  );

  if (selected) return <GameDetail gameUid={selected} onClose={onClose} />;

  const columns: Column<Game>[] = [
    { key: 'date', header: 'Date', render: (row) => shortDate(row.game_date), sortValue: (row) => row.game_date },
    { key: 'league', header: 'League', render: (row) => <Badge>{row.league_id}</Badge> },
    {
      key: 'match',
      header: 'Match',
      render: (row) => (
        <span style={{ fontWeight: 600 }}>
          {row.away_name} @ {row.home_name}
        </span>
      ),
      sortValue: (row) => row.home_name,
    },
    {
      key: 'score',
      header: 'Score',
      numeric: true,
      render: (row) =>
        row.home_score != null ? (
          <span>
            {row.away_score} – {row.home_score}
          </span>
        ) : (
          <span className="faint">—</span>
        ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => (
        <Badge tone={row.status === 'final' ? 'neutral' : 'accent'}>{row.status}</Badge>
      ),
    },
    { key: 'season', header: 'Season', render: (row) => <span className="faint">{row.season}</span> },
  ];

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Games</h1>
          <p className="page-sub">
            Fixtures and results from the canonical store. Open a game for its stored
            predictions and how its market moved.
          </p>
        </div>
        <div className="controls">
          <Segmented
            value={sport}
            onChange={setSport}
            options={[
              { value: 'nba', label: 'NBA' },
              { value: 'soccer', label: 'Soccer' },
            ]}
          />
          <Segmented
            value={status}
            onChange={setStatus}
            options={[
              { value: 'scheduled', label: 'Upcoming' },
              { value: 'final', label: 'Results' },
            ]}
          />
          <Field label="Window (days)">
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {[7, 14, 30, 90, 365].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </Field>
          <button onClick={refresh}>Refresh</button>
        </div>
      </div>

      {loading && <Loading />}
      {error && <ErrorState error={error} onRetry={refresh} />}
      {data && (
        <Card title={`${data.count} games`}>
          <DataTable
            columns={columns}
            rows={data.games}
            onRowClick={(row) => onOpenGame(row.game_uid)}
            initialSort={{ key: 'date', direction: 'desc' }}
            emptyLabel="No games in this window"
          />
        </Card>
      )}
    </div>
  );
}

const KIND_TONE: Record<string, 'accent' | 'neutral' | 'warn' | 'pos'> = {
  prediction: 'accent',
  market: 'neutral',
  lineup: 'pos',
  information: 'warn',
};

function EventTimeline({ gameUid }: { gameUid: string }) {
  const { data, error, loading } = useAsync(() => api.timeline(gameUid), [gameUid]);

  if (loading) return <Card title="Timeline"><Loading /></Card>;
  if (error) return <Card title="Timeline"><ErrorState error={error} /></Card>;
  if (!data?.found || !data.timeline.length) {
    return (
      <Card title="Timeline">
        <Empty title="Nothing observed for this fixture yet">
          Timelines fill in as predictions, prices, lineups and information events are recorded.
        </Empty>
      </Card>
    );
  }

  const movement = data.market_vs_model;

  return (
    <Card
      title="Timeline"
      note="Attribution pairs a probability movement with information observed shortly before it. That is co-occurrence, not proven causation."
    >
      {movement?.available && (
        <div className="grid grid-3" style={{ marginBottom: 14 }}>
          <Tile label="Model move" value={pct(movement.model_move, 2)} />
          <Tile label="Market move" value={pct(movement.market_move, 2)} />
          <Tile
            label="Direction"
            value={movement.same_direction ? 'same' : 'opposite'}
            tone={movement.same_direction ? 'pos' : 'warn'}
            sub={movement.model_moved_first ? 'model moved first' : 'market moved first'}
          />
        </div>
      )}

      <div className="stack" style={{ gap: 6 }}>
        {data.timeline.map((entry: TimelineEntry, index: number) => (
          <div
            key={`${entry.timestamp}-${index}`}
            className="row"
            style={{ justifyContent: 'space-between', gap: 10, alignItems: 'flex-start' }}
          >
            <span className="row" style={{ gap: 8, minWidth: 0 }}>
              <span className="faint mono" style={{ fontSize: '0.72rem', minWidth: 132 }}>
                {entry.timestamp?.replace('T', ' ').slice(0, 16)}
              </span>
              <Badge tone={KIND_TONE[entry.kind] ?? 'neutral'}>{entry.kind}</Badge>
              <span style={{ fontSize: '0.82rem' }}>{entry.label}</span>
            </span>
            {entry.kind === 'prediction' && entry.detail?.move != null && (
              <span
                className={`num ${entry.detail.move > 0 ? 'pos' : 'neg'}`}
                style={{ fontSize: '0.8rem' }}
              >
                {pct(entry.detail.move, 2)}
              </span>
            )}
          </div>
        ))}
      </div>

      {data.attributions.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginBottom: 8 }}>Material movements</h3>
          {data.attributions.map((item, index) => (
            <div key={index} className="card" style={{ padding: 10, marginBottom: 6 }}>
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.82rem' }}>
                  {item.market}/{item.selection}{' '}
                  <span className={item.move && item.move > 0 ? 'pos' : 'neg'}>
                    {pct(item.move, 2)}
                  </span>
                </span>
                <Badge tone={item.explained ? 'warn' : 'neutral'}>
                  {item.explained ? 'candidate cause' : 'unexplained'}
                </Badge>
              </div>
              {item.candidate_causes.map((cause, causeIndex) => (
                <div key={causeIndex} className="faint" style={{ fontSize: '0.75rem', marginTop: 3 }}>
                  {cause.kind}: {cause.label}
                </div>
              ))}
              <div className="faint" style={{ fontSize: '0.72rem', marginTop: 4 }}>{item.note}</div>
            </div>
          ))}
        </div>
      )}

      {data.clv.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginBottom: 8 }}>Closing line value</h3>
          <div className="table-wrap">
            <table className="dl">
              <thead>
                <tr>
                  <th>Selection</th>
                  <th className="num">Entry</th>
                  <th>Book</th>
                  <th className="num">Close</th>
                  <th className="num">CLV</th>
                  <th className="num">Same book</th>
                  <th>Status</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {data.clv.map((row: any, index: number) => (
                  <tr key={index}>
                    <td>{row.selection}</td>
                    <td className="num">{num(row.entry_odds, 2)}</td>
                    <td className="faint" style={{ fontSize: '0.72rem' }}>{row.entry_book ?? '—'}</td>
                    <td className="num">{row.closing_odds ? num(row.closing_odds, 2) : '—'}</td>
                    <td className={`num ${(row.clv_price_pct ?? 0) > 0 ? 'pos' : 'neg'}`}>
                      {row.clv_price_pct != null ? pctPoints(row.clv_price_pct) : '—'}
                    </td>
                    <td className="num">
                      {row.clv_same_book_pct != null ? pctPoints(row.clv_same_book_pct) : '—'}
                    </td>
                    <td><Badge>{row.status}</Badge></td>
                    <td>{row.result ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}

function LineupPanel({ gameUid }: { gameUid: string }) {
  const { data, loading } = useAsync(() => api.lineups(gameUid).catch(() => null), [gameUid]);
  if (loading) return null;
  if (!data || !Object.keys(data.teams ?? {}).length) return null;

  const freshness = data.freshness ?? {};
  return (
    <Card
      title="Lineups"
      actions={
        <Badge
          tone={freshness.state === 'fresh' ? 'pos' : freshness.state === 'stale' ? 'neg' : 'warn'}
        >
          {data.lineup_state} · {freshness.state}
          {freshness.age_minutes != null ? ` (${Math.round(freshness.age_minutes)}m)` : ''}
        </Badge>
      }
      note={data.note}
    >
      <div className="grid grid-2">
        {Object.entries(data.teams).map(([team, entry]: [string, any]) => (
          <div key={team}>
            <h3 style={{ marginBottom: 6 }}>
              {team} <span className="faint">{entry.formation ?? ''}</span>
            </h3>
            <div className="row" style={{ gap: 4 }}>
              {entry.starters.map((player: any) => (
                <Badge
                  key={player.player}
                  tone={player.position_group === 'goalkeeper' ? 'accent' : 'neutral'}
                  title={player.position_group ?? player.role ?? ''}
                >
                  {player.player}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function GameDetail({ gameUid, onClose }: { gameUid: string; onClose: () => void }) {
  const { data, error, loading } = useAsync(() => api.game(gameUid), [gameUid]);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <button className="ghost" onClick={onClose} style={{ paddingLeft: 0 }}>
            ← Back to games
          </button>
          <h1 style={{ marginTop: 6 }}>
            {loading ? 'Loading…' : `${data?.game?.away_name} @ ${data?.game?.home_name}`}
          </h1>
          {data?.game && (
            <p className="page-sub">
              {shortDate(data.game.game_date)} · {data.game.league_id} · {data.game.status}
              {data.game.venue ? ` · ${data.game.venue}` : ''}
            </p>
          )}
        </div>
      </div>

      {loading && <Loading />}
      {error && <ErrorState error={error} />}

      {data && (
        <>
          <LineupPanel gameUid={gameUid} />
          <EventTimeline gameUid={gameUid} />
          {Object.entries(data.odds_movement ?? {}).map(([market, movement]: [string, any]) => {
            const selections = Object.keys(movement.selections ?? {});
            const byTime: Record<string, { name: string; points: any[] }> = {};
            (movement.series ?? []).forEach((point: any) => {
              byTime[point.selection] ??= { name: point.selection, points: [] };
              byTime[point.selection].points.push({
                x: new Date(point.captured_at).getTime(),
                y: point.price,
                label: shortDate(point.captured_at),
              });
            });
            const series = Object.values(byTime).filter((s) => s.points.length > 1);

            return (
              <Card
                key={market}
                title={`Line movement — ${market}`}
                actions={<Legend items={selections.map((name, index) => ({ name, index }))} />}
                note="Opening price is the earliest snapshot the platform recorded; closing is the price flagged at kick-off. Movement between them is what CLV is measured against."
              >
                <div className="table-wrap" style={{ marginBottom: 12 }}>
                  <table className="dl">
                    <thead>
                      <tr>
                        <th>Selection</th>
                        <th className="num">Opening</th>
                        <th className="num">Current</th>
                        <th className="num">Closing</th>
                        <th className="num">Move</th>
                        <th className="num">Snapshots</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selections.map((selection) => {
                        const row = movement.selections[selection];
                        return (
                          <tr key={selection}>
                            <td>{selection}</td>
                            <td className="num">{num(row.opening, 2)}</td>
                            <td className="num">
                              {num(row.current, 2)}{' '}
                              <span className="faint">({american(row.current)})</span>
                            </td>
                            <td className="num">{row.closing ? num(row.closing, 2) : '—'}</td>
                            <td className={`num ${(row.movement_pct ?? 0) >= 0 ? 'pos' : 'neg'}`}>
                              {row.movement_pct == null ? '—' : `${row.movement_pct.toFixed(2)}%`}
                            </td>
                            <td className="num">{row.snapshots}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {series.length > 0 && (
                  <LineChart series={series} valueFormat={(value) => value.toFixed(2)} />
                )}
              </Card>
            );
          })}

          <Card title="Stored predictions">
            {data.predictions?.length ? (
              <div className="table-wrap">
                <table className="dl">
                  <thead>
                    <tr>
                      <th>Made</th>
                      <th>Selection</th>
                      <th className="num">Model</th>
                      <th className="num">Market</th>
                      <th className="num">Edge</th>
                      <th className="num">Price</th>
                      <th className="num">Stake</th>
                      <th>Model version</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.predictions.map((row: any) => (
                      <tr key={row.prediction_id}>
                        <td>{timeAgo(row.created_at)}</td>
                        <td>{row.selection}</td>
                        <td className="num">{pct(row.model_prob)}</td>
                        <td className="num">{pct(row.market_prob)}</td>
                        <td className={`num ${(row.edge ?? 0) > 0 ? 'pos' : 'neg'}`}>
                          {row.edge == null ? '—' : pct(row.edge, 2)}
                        </td>
                        <td className="num">{num(row.price_decimal, 2)}</td>
                        <td className="num">{row.stake ? num(row.stake, 2) : '—'}</td>
                        <td className="faint" style={{ fontSize: '0.72rem' }}>
                          {row.model_version ?? '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty title="No predictions stored for this game" />
            )}
          </Card>
        </>
      )}
    </div>
  );
}
