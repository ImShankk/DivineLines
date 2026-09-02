/* Dashboard: the state of the system in one screen — what it likes right now,
   how healthy the data is, and how the bankroll is doing. */

import { api, money, num, shortDate, signedPct, timeAgo } from '../api';
import { LineChart } from '../charts';
import {
  Badge,
  Banner,
  Card,
  Empty,
  ErrorState,
  Loading,
  Meter,
  StateDot,
  Tile,
  useAsync,
} from '../ui';

export function DashboardPage({ onNavigate }: { onNavigate: (route: string) => void }) {
  const health = useAsync(() => api.health(), []);
  const performance = useAsync(() => api.performance(), []);
  const nba = useAsync(() => api.predictions('nba', 3).catch(() => null), []);
  const soccer = useAsync(() => api.predictions('soccer', 5).catch(() => null), []);

  const opportunities = [
    ...(nba.data?.opportunities ?? []),
    ...(soccer.data?.opportunities ?? []),
  ].sort((a, b) => b.edge_score - a.edge_score);

  const curve = performance.data?.bankroll_curve ?? [];
  const settled = performance.data?.overall?.[0];

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <p className="page-sub">
            Live state of the platform: current opportunities, data health, and realised
            performance from the prediction ledger.
          </p>
        </div>
        {health.data && (
          <div className="row">
            <StateDot state={health.data.status} />
            <span className="muted">
              {health.data.status} · {health.data.mode} mode
            </span>
          </div>
        )}
      </div>

      {health.error && <ErrorState error={health.error} onRetry={health.refresh} />}

      <div className="grid grid-4">
        <Tile
          label="Open opportunities"
          value={opportunities.length}
          sub={`${nba.data?.opportunities.length ?? 0} NBA · ${soccer.data?.opportunities.length ?? 0} soccer`}
        />
        <Tile
          label="Settled bets"
          value={settled ? settled.bets : 0}
          sub={settled ? `${money(settled.staked)} staked` : 'nothing graded yet'}
        />
        <Tile
          label="ROI"
          value={settled ? signedPct(settled.roi, 2) : '—'}
          tone={settled ? (settled.roi > 0 ? 'pos' : 'neg') : 'neutral'}
          sub={settled ? `${money(settled.profit)} profit` : 'needs settled bets'}
        />
        <Tile
          label="Mean CLV"
          value={
            performance.data?.clv?.mean_pct != null
              ? `${performance.data.clv.mean_pct >= 0 ? '+' : ''}${num(performance.data.clv.mean_pct, 2)}%`
              : '—'
          }
          tone={
            performance.data?.clv?.mean_pct == null
              ? 'neutral'
              : performance.data.clv.mean_pct > 0
                ? 'pos'
                : 'neg'
          }
          sub={
            performance.data?.clv?.n
              ? `${performance.data.clv.n} bets vs close`
              : 'closing lines needed'
          }
        />
      </div>

      <div className="grid grid-2">
        <Card
          title="Best current opportunities"
          actions={<button className="ghost" onClick={() => onNavigate('opportunities')}>Open scanner</button>}
        >
          {nba.loading || soccer.loading ? (
            <Loading label="Pricing slates" />
          ) : opportunities.length ? (
            <div className="stack" style={{ gap: 8 }}>
              {opportunities.slice(0, 6).map((row) => (
                <div
                  key={row.game_uid + row.selection}
                  className="row"
                  style={{ justifyContent: 'space-between', gap: 10 }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>
                      {row.away_name} @ {row.home_name}
                    </div>
                    <div className="faint" style={{ fontSize: '0.72rem' }}>
                      {shortDate(row.game_date)} · {row.league_id} · {row.selection}
                    </div>
                  </div>
                  <div className="row" style={{ gap: 10 }}>
                    <span className="num pos">{signedPct(row.edge, 1)}</span>
                    <span className="num muted">{num(row.price_decimal, 2)}</span>
                    <Meter value={row.edge_score} max={10} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Empty title="No qualifying opportunities">
              Either no fixtures are priced right now, or nothing clears the edge and
              risk thresholds. The system reporting "no bet" is a result, not an error.
            </Empty>
          )}
        </Card>

        <Card title="System status" actions={<button className="ghost" onClick={() => onNavigate('system')}>Details</button>}>
          {health.loading ? (
            <Loading />
          ) : health.data ? (
            <div className="stack" style={{ gap: 7 }}>
              {health.data.sources.slice(0, 7).map((source) => (
                <div key={`${source.source}:${source.dataset}`} className="row" style={{ justifyContent: 'space-between' }}>
                  <span className="row" style={{ gap: 7 }}>
                    <StateDot state={source.state} />
                    <span style={{ fontSize: '0.82rem' }}>{source.source}</span>
                    <span className="faint" style={{ fontSize: '0.72rem' }}>{source.dataset}</span>
                  </span>
                  <span className="faint num" style={{ fontSize: '0.75rem' }}>
                    {timeAgo(source.last_success)}
                  </span>
                </div>
              ))}
              {!health.data.validation.ok && (
                <Banner tone="bad">
                  {health.data.validation.critical.length} critical data issue(s) detected.
                </Banner>
              )}
            </div>
          ) : null}
        </Card>
      </div>

      <Card
        title="Bankroll"
        note="Paper-mode ledger. Every prediction is stored with the price available at decision time, then graded automatically once the game finishes."
      >
        {performance.loading ? (
          <Loading />
        ) : curve.length ? (
          <LineChart
            series={[
              {
                name: 'bankroll',
                points: curve.map((row: any, index: number) => ({
                  x: index,
                  y: row.bankroll,
                  label: shortDate(row.settled_at),
                })),
              },
            ]}
            valueFormat={(value) => `$${value.toFixed(0)}`}
          />
        ) : (
          <Empty title="No settled bets yet">
            {performance.data?.note ??
              'Run a scan in paper mode, then settle after the games finish.'}
          </Empty>
        )}
      </Card>

      {health.data?.models?.length ? (
        <Card title="Active models">
          <div className="row" style={{ gap: 8 }}>
            {health.data.models.slice(0, 4).map((model) => (
              <Badge key={model.model_id} tone="accent" title={model.model_id}>
                {model.sport} · {model.model_version} · {timeAgo(model.trained_at)}
              </Badge>
            ))}
          </div>
        </Card>
      ) : (
        <Banner tone="warn">
          No trained models registered. Run <code>divinelines train</code> to fit and register
          the NBA and soccer engines.
        </Banner>
      )}
    </div>
  );
}
