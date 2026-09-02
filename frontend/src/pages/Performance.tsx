/* Performance: realised results from the ledger, plus the backtest evidence.
   This page is where uncomfortable findings live, and they are shown as
   prominently as the good ones. */

import { useState } from 'react';
import { api, money, num, pct, shortDate, signedPct } from '../api';
import { BucketBars, CalibrationChart, LineChart } from '../charts';
import {
  Banner,
  Card,
  Empty,
  ErrorState,
  Loading,
  Segmented,
  Tile,
  useAsync,
} from '../ui';

export function PerformancePage() {
  const [view, setView] = useState<'live' | 'backtest'>('backtest');
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Performance</h1>
          <p className="page-sub">
            Realised results from the paper ledger, and the walk-forward evidence behind
            the models. Probability quality is reported alongside profit because over a
            few hundred bets profit is mostly noise.
          </p>
        </div>
        <Segmented
          value={view}
          onChange={setView}
          options={[
            { value: 'backtest', label: 'Backtests' },
            { value: 'live', label: 'Ledger' },
          ]}
        />
      </div>
      {view === 'live' ? <LedgerView /> : <BacktestView />}
    </div>
  );
}

function LedgerView() {
  const { data, error, loading, refresh } = useAsync(() => api.performance(), []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const overall = data.overall?.[0];
  const curve = data.bankroll_curve ?? [];

  return (
    <div className="stack">
      {data.note && <Banner tone="info">{data.note}</Banner>}

      <div className="grid grid-4">
        <Tile label="Settled bets" value={overall?.bets ?? 0} sub={`${data.open_bets} still open`} />
        <Tile
          label="Profit"
          value={overall ? money(overall.profit) : '—'}
          tone={overall ? (overall.profit > 0 ? 'pos' : 'neg') : 'neutral'}
          sub={overall ? `${money(overall.staked)} staked` : undefined}
        />
        <Tile
          label="ROI"
          value={overall ? signedPct(overall.roi, 2) : '—'}
          tone={overall ? (overall.roi > 0 ? 'pos' : 'neg') : 'neutral'}
        />
        <Tile
          label="Hit rate"
          value={overall ? pct(overall.hit_rate) : '—'}
          sub={overall ? `avg price ${num(overall.avg_price, 2)}` : undefined}
        />
      </div>

      <Card title="Bankroll trajectory">
        {curve.length ? (
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
          <Empty title="No settled bets yet" />
        )}
      </Card>

      <div className="grid grid-2">
        <Card
          title="ROI by edge bucket"
          note="If larger predicted edges do not produce better realised results, the model is overconfident in its tails."
        >
          {data.by_edge_bucket?.length ? (
            <BucketBars
              data={data.by_edge_bucket.map((row: any) => ({
                bucket: String(row.group),
                value: row.roi,
                count: row.bets,
              }))}
              label="ROI"
            />
          ) : (
            <Empty title="Needs settled bets" />
          )}
        </Card>

        <Card title="ROI by price bucket">
          {data.by_odds_bucket?.length ? (
            <BucketBars
              data={data.by_odds_bucket.map((row: any) => ({
                bucket: String(row.group),
                value: row.roi,
                count: row.bets,
              }))}
              label="ROI"
            />
          ) : (
            <Empty title="Needs settled bets" />
          )}
        </Card>
      </div>

      <Card
        title="Closing line value"
        note="CLV converges much faster than profit: beating the closing price consistently is the strongest short-horizon evidence that an edge is real."
      >
        <div className="grid grid-3">
          <Tile label="Bets with CLV" value={data.clv.n} />
          <Tile
            label="Mean CLV"
            value={data.clv.mean_pct == null ? '—' : `${num(data.clv.mean_pct, 2)}%`}
            tone={data.clv.mean_pct == null ? 'neutral' : data.clv.mean_pct > 0 ? 'pos' : 'neg'}
          />
          <Tile
            label="Beat the close"
            value={data.clv.beat_close_rate == null ? '—' : pct(data.clv.beat_close_rate)}
          />
        </div>
      </Card>
    </div>
  );
}

function BacktestView() {
  const { data, error, loading, refresh } = useAsync(() => api.backtests(), []);
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;

  const backtests = data?.backtests ?? {};
  if (!Object.keys(backtests).length) {
    return (
      <Empty title="No backtest results stored">
        {data?.note ?? 'Run `divinelines backtest --save` to generate them.'}
      </Empty>
    );
  }

  return (
    <div className="stack">
      {Object.entries(backtests).map(([name, result]: [string, any]) => (
        <BacktestCard key={name} name={name} result={result} />
      ))}
    </div>
  );
}

function BacktestCard({ name, result }: { name: string; result: any }) {
  const isSoccer = name.startsWith('soccer');

  if (isSoccer) {
    const metrics = result.metrics ?? {};
    const model = result.probability_metrics?.model;
    const market = result.probability_metrics?.market_novig;
    const beatsMarket = model && market && model.log_loss < market.log_loss;

    return (
      <Card
        title="Soccer — walk-forward against real historical prices"
        note="Bets are struck at the pre-match price a bettor could actually see; the closing line is used only to measure CLV afterwards."
      >
        {model && market && (
          <Banner tone={beatsMarket ? 'info' : 'warn'}>
            <div>
              <strong>
                {beatsMarket
                  ? 'The model edges the market on probability quality.'
                  : 'The model does not beat the market.'}
              </strong>
              <div className="muted" style={{ marginTop: 3 }}>
                Model log loss {num(model.log_loss, 4)} vs no-vig market {num(market.log_loss, 4)}{' '}
                over {model.n} matches. A model that cannot out-predict the closing market
                should not be staked, and the platform's default thresholds reflect that.
              </div>
            </div>
          </Banner>
        )}
        <div className="grid grid-4" style={{ marginTop: 12 }}>
          <Tile label="Bets" value={metrics.bets ?? 0} sub={`${money(metrics.staked)} staked`} />
          <Tile
            label="ROI"
            value={signedPct(metrics.roi, 2)}
            tone={metrics.roi > 0 ? 'pos' : 'neg'}
            sub={money(metrics.profit)}
          />
          <Tile label="Hit rate" value={pct(metrics.hit_rate)} sub={`avg price ${num(metrics.avg_price, 2)}`} />
          <Tile
            label="Mean CLV"
            value={metrics.clv_mean_pct == null ? '—' : `${num(metrics.clv_mean_pct, 2)}%`}
            tone={metrics.clv_mean_pct > 0 ? 'pos' : 'neg'}
          />
        </div>

        {result.by_price?.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginBottom: 8 }}>ROI by price bucket</h3>
            <BucketBars
              data={result.by_price.map((row: any) => ({
                bucket: String(row.bucket),
                value: row.roi,
                count: row.bets,
              }))}
              label="ROI"
            />
          </div>
        )}

        {result.totals_metrics && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginBottom: 8 }}>Over/under 2.5 goals</h3>
            <div className="grid grid-3">
              <Tile label="Bets" value={result.totals_metrics.bets} />
              <Tile
                label="ROI"
                value={signedPct(result.totals_metrics.roi, 2)}
                tone={result.totals_metrics.roi > 0 ? 'pos' : 'neg'}
              />
              <Tile label="Hit rate" value={pct(result.totals_metrics.hit_rate)} />
            </div>
          </div>
        )}
      </Card>
    );
  }

  const probability = result.probability_metrics ?? {};
  const breakEven = result.break_even ?? [];

  return (
    <Card title="NBA — walk-forward probability quality" note={result.note}>
      <div className="table-wrap">
        <table className="dl">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Log loss</th>
              <th className="num">Brier</th>
              <th className="num">Accuracy</th>
              <th className="num">ECE</th>
              <th className="num">Brier skill</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(probability).map(([key, metrics]: [string, any]) => (
              <tr key={key}>
                <td>{key.replace(/_/g, ' ')}</td>
                <td className="num">{num(metrics.log_loss, 4)}</td>
                <td className="num">{num(metrics.brier, 4)}</td>
                <td className="num">{pct(metrics.accuracy)}</td>
                <td className="num">{num(metrics.ece, 4)}</td>
                <td className={`num ${metrics.brier_skill > 0 ? 'pos' : ''}`}>
                  {num(metrics.brier_skill, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {result.reliability?.length > 0 && (
        <div className="grid grid-2" style={{ marginTop: 16 }}>
          <div>
            <h3 style={{ marginBottom: 8 }}>Calibration</h3>
            <CalibrationChart
              points={result.reliability.map((row: any) => ({
                predicted: row.predicted,
                observed: row.observed,
                count: row.count,
              }))}
            />
            <p className="faint" style={{ fontSize: '0.75rem' }}>
              Dot size is the number of games in the bin. Points below the diagonal mean
              the model was overconfident in that band.
            </p>
          </div>
          {breakEven.length > 0 && (
            <div>
              <h3 style={{ marginBottom: 8 }}>Break-even by probability band</h3>
              <div className="table-wrap">
                <table className="dl">
                  <thead>
                    <tr>
                      <th>Band</th>
                      <th className="num">Games</th>
                      <th className="num">Predicted</th>
                      <th className="num">Actual</th>
                      <th className="num">Gap</th>
                      <th className="num">Fair price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakEven.map((row: any) => (
                      <tr key={row.band}>
                        <td className="faint">{row.band}</td>
                        <td className="num">{row.games}</td>
                        <td className="num">{pct(row.predicted)}</td>
                        <td className="num">{pct(row.realised)}</td>
                        <td className={`num ${row.calibration_gap > 0 ? 'pos' : 'neg'}`}>
                          {signedPct(row.calibration_gap, 1)}
                        </td>
                        <td className="num">{num(row.fair_price_realised, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
