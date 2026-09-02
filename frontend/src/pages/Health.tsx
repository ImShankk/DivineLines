/* Model Health & CLV.

   The page has one job: answer "is DivineLines getting better?" in seconds,
   without letting a small sample masquerade as an edge. So every number carries
   its sample size, CLV is never styled as profit, and a +0.3% CLV on six bets
   renders as "insufficient sample" rather than a green edge badge. */

import { useState } from 'react';
import {
  api,
  money,
  num,
  pct,
  pctPoints,
  signedPct,
  type ClvSummary,
  type ModelHealth,
  type Sport,
} from '../api';
import { BucketBars, CalibrationChart, LineChart } from '../charts';
import {
  Badge,
  Banner,
  Card,
  DataTable,
  Empty,
  ErrorState,
  Loading,
  Meter,
  Segmented,
  Tile,
  useAsync,
} from '../ui';

const STATUS_META: Record<string, { tone: 'pos' | 'neg' | 'warn' | 'accent' | 'neutral'; help: string }> = {
  MARKET_BEATING: {
    tone: 'pos',
    help: 'Beats the no-vig market on log loss over a sample large enough to mean something.',
  },
  VALIDATED_FOR_PREDICTION: {
    tone: 'accent',
    help: 'Probabilities carry real information versus the base rate, but the market is still at least as good.',
  },
  UNPROVEN: { tone: 'warn', help: 'No measurable skill over the base rate yet.' },
  DEGRADED: { tone: 'neg', help: 'Worse than always predicting the base rate.' },
  INSUFFICIENT_SAMPLE: { tone: 'neutral', help: 'Not enough graded predictions to make any claim.' },
};

export function HealthPage() {
  const [sport, setSport] = useState<Sport>('nba');
  const [basis, setBasis] = useState<'consensus' | 'same_book'>('consensus');

  const health = useAsync(() => api.modelHealthAll(sport), [sport]);
  const clv = useAsync(() => api.clv(sport, basis), [sport, basis]);
  const skill = useAsync(() => api.clvSkill(sport).catch(() => null), [sport]);
  const coverage = useAsync(() => api.clvCoverage(sport).catch(() => null), [sport]);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Model Health &amp; CLV</h1>
          <p className="page-sub">
            Predictive health asks whether the probabilities are good. Betting health asks
            whether acting on them beat the market. A model can pass one and fail the other,
            so they are never merged into a single score.
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
          <button onClick={() => { health.refresh(); clv.refresh(); }}>Refresh</button>
        </div>
      </div>

      {health.loading && <Loading label="Scoring the prediction ledger" />}
      {health.error && <ErrorState error={health.error} onRetry={health.refresh} />}

      {health.data && <HealthSummary data={health.data} />}

      <Card
        title="Closing line value"
        actions={
          <Segmented
            value={basis}
            onChange={setBasis}
            options={[
              { value: 'consensus', label: 'vs consensus close' },
              { value: 'same_book', label: 'vs same book' },
            ]}
          />
        }
        note={clv.data?.disclaimer}
      >
        {clv.loading ? (
          <Loading />
        ) : clv.error ? (
          <ErrorState error={clv.error} onRetry={clv.refresh} />
        ) : clv.data ? (
          <ClvSection report={clv.data} />
        ) : null}
      </Card>

      {skill.data && <SkillSection skill={skill.data} />}

      {clv.data && Object.keys(clv.data.cohorts ?? {}).length > 0 && (
        <CohortSection cohorts={clv.data.cohorts} />
      )}

      {coverage.data && <CoverageSection coverage={coverage.data} />}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const meta = STATUS_META[status] ?? { tone: 'neutral' as const, help: status };
  return (
    <Badge tone={meta.tone} title={meta.help}>
      {status.replace(/_/g, ' ').toLowerCase()}
    </Badge>
  );
}

function HealthSummary({ data }: { data: { windows: Record<string, ModelHealth>; regression: any; stability: any } }) {
  const windows = Object.entries(data.windows);
  const allTime = data.windows['all_time'];

  return (
    <div className="stack">
      {data.regression?.regression && (
        <Banner tone="bad">
          <div>
            <strong>Model regression detected</strong>
            <div className="muted" style={{ marginTop: 3 }}>
              {data.regression.reason} — recent log loss {num(data.regression.recent_log_loss, 4)} vs
              lifetime {num(data.regression.lifetime_log_loss, 4)}.
            </div>
          </div>
        </Banner>
      )}

      {allTime && allTime.sample_size === 0 && (
        <Banner tone="info">
          No graded predictions yet for this sport. Health populates once predictions are
          recorded and their events finish — run <code>divinelines scan --paper</code>, then
          <code> divinelines settle</code>.
        </Banner>
      )}

      <div className="grid grid-4">
        <Tile
          label="Status"
          value={allTime ? <StatusBadge status={allTime.status} /> : '—'}
          sub={allTime?.status_reason}
        />
        <Tile
          label="Graded predictions"
          value={allTime?.sample_size ?? 0}
          sub={allTime?.betting?.settled_bets ? `${allTime.betting.settled_bets} settled` : undefined}
        />
        <Tile
          label="Brier skill"
          value={allTime?.predictive?.brier_skill != null ? signedPct(allTime.predictive.brier_skill, 2) : '—'}
          tone={(allTime?.predictive?.brier_skill ?? 0) > 0 ? 'pos' : 'neutral'}
          sub="vs always predicting the base rate"
        />
        <Tile
          label="vs market"
          value={
            allTime?.market_comparison?.skill_vs_market != null
              ? num(allTime.market_comparison.skill_vs_market, 4)
              : '—'
          }
          tone={allTime?.market_comparison?.beats_market ? 'pos' : 'neg'}
          sub={
            allTime?.market_comparison?.n
              ? `log loss gap over ${allTime.market_comparison.n} priced`
              : 'no market comparison'
          }
        />
      </div>

      <div className="grid grid-2">
        <Card title="By window" note="Small recent samples are not comparable to the lifetime record.">
          <div className="table-wrap">
            <table className="dl">
              <thead>
                <tr>
                  <th>Window</th>
                  <th className="num">n</th>
                  <th className="num">Brier</th>
                  <th className="num">Log loss</th>
                  <th className="num">ECE</th>
                  <th className="num">Skill</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {windows.map(([label, entry]) => (
                  <tr key={label}>
                    <td>{label.replace(/_/g, ' ')}</td>
                    <td className="num">{entry.sample_size}</td>
                    <td className="num">{num(entry.predictive?.brier, 4)}</td>
                    <td className="num">{num(entry.predictive?.log_loss, 4)}</td>
                    <td className="num">{num(entry.predictive?.ece, 4)}</td>
                    <td className={`num ${(entry.predictive?.brier_skill ?? 0) > 0 ? 'pos' : ''}`}>
                      {entry.predictive?.brier_skill != null
                        ? num(entry.predictive.brier_skill, 4)
                        : '—'}
                    </td>
                    <td><StatusBadge status={entry.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card
          title="Calibration"
          note="Dot size is the number of predictions in the bin. Points below the diagonal mean the model was overconfident."
        >
          {allTime?.calibration_curve?.length ? (
            <CalibrationChart points={allTime.calibration_curve} />
          ) : (
            <Empty title="No calibration data yet" />
          )}
        </Card>
      </div>

      {allTime?.stability && allTime.stability.n_series > 0 && (
        <Card
          title="Prediction stability"
          note="How much a fixture's probability moves between prediction versions. Large swings on small information updates are a stability problem, not responsiveness."
        >
          <div className="grid grid-4">
            <Tile label="Fixtures with revisions" value={allTime.stability.n_series} />
            <Tile label="Mean move" value={pct(allTime.stability.mean_abs_move, 2)} />
            <Tile label="Largest move" value={pct(allTime.stability.max_move, 2)} />
            <Tile label="95th pct range" value={pct(allTime.stability.p95_range, 2)} />
          </div>
        </Card>
      )}
    </div>
  );
}

function ClvVerdict({ summary }: { summary: ClvSummary }) {
  if (!summary.n) return <Badge tone="neutral">no data</Badge>;
  if (!summary.significant) return <Badge tone="neutral" title={summary.interpretation}>uncertain</Badge>;
  return (
    <Badge tone={summary.mean_clv_price_pct > 0 ? 'pos' : 'neg'} title={summary.interpretation}>
      {summary.mean_clv_price_pct > 0 ? 'positive CLV' : 'negative CLV'}
    </Badge>
  );
}

function ClvSection({ report }: { report: any }) {
  const overall: ClvSummary = report.overall;

  if (!overall.n) {
    return (
      <Empty title="No CLV records yet">
        CLV appears once predictions have been settled against a closing line. Run{' '}
        <code>divinelines settle</code> after events finish.
      </Empty>
    );
  }

  return (
    <div className="stack">
      {!report.sufficient_sample && (
        <Banner tone="warn">
          Sample of {overall.n} is below the {report.min_sample_for_inference} needed before a CLV
          figure means anything. Treat the numbers below as descriptive only.
        </Banner>
      )}

      <div className="grid grid-4">
        <Tile label="Sample" value={overall.n} sub={report.basis_description} />
        <Tile
          label="Mean CLV"
          value={pctPoints(overall.mean_clv_price_pct)}
          tone={overall.significant ? (overall.mean_clv_price_pct > 0 ? 'pos' : 'neg') : 'neutral'}
          sub={
            overall.ci_low != null
              ? `95% CI ${pctPoints(overall.ci_low)} … ${pctPoints(overall.ci_high)}`
              : 'interval needs a larger sample'
          }
        />
        <Tile label="Median CLV" value={pctPoints(overall.median_clv_price_pct)} />
        <Tile
          label="Beat the close"
          value={pct(overall.beat_close_rate)}
          sub={<ClvVerdict summary={overall} />}
        />
      </div>

      {report.vs_profit && (
        <div className="grid grid-3">
          <Tile
            label="Paper ROI"
            value={report.vs_profit.roi != null ? signedPct(report.vs_profit.roi, 2) : '—'}
            /* Only a statistically significant return gets coloured. A green
               +4.8% that is consistent with break-even is exactly the kind of
               decorative KPI this page exists to avoid. */
            tone={
              report.vs_profit.roi_interval?.significant
                ? (report.vs_profit.roi! > 0 ? 'pos' : 'neg')
                : 'neutral'
            }
            sub={
              report.vs_profit.roi_interval?.ci_low != null
                ? `95% CI ${signedPct(report.vs_profit.roi_interval.ci_low, 1)} … ${signedPct(
                    report.vs_profit.roi_interval.ci_high,
                    1,
                  )} — ${report.vs_profit.roi_interval.interpretation}`
                : `${report.vs_profit.settled_bets} settled · ${money(report.vs_profit.staked)} staked`
            }
          />
          <Tile label="Profit" value={money(report.vs_profit.profit)} />
          <Tile
            label="CLV sample"
            value={report.vs_profit.clv_sample ?? overall.n}
            sub="CLV and ROI are different questions"
          />
        </div>
      )}

      <div className="grid grid-2">
        <div>
          <h3 style={{ marginBottom: 8 }}>Cumulative mean CLV</h3>
          {report.cumulative?.length > 1 ? (
            <LineChart
              series={[
                {
                  name: 'mean CLV %',
                  points: report.cumulative.map((row: any) => ({
                    x: row.n,
                    y: row.cumulative_mean,
                    label: `n=${row.n}`,
                  })),
                },
              ]}
              valueFormat={(value) => `${value.toFixed(1)}%`}
              zeroLine
            />
          ) : (
            <Empty title="Needs more records" />
          )}
          <p className="faint" style={{ fontSize: '0.75rem' }}>
            A running average settles down as the sample grows; early swings are noise.
          </p>
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Distribution</h3>
          {report.distribution?.length ? (
            <BucketBars
              data={report.distribution.map((row: any) => ({
                bucket: row.bucket,
                value: row.count,
              }))}
              valueFormat={(value) => value.toFixed(0)}
              label="predictions"
            />
          ) : (
            <Empty title="No distribution yet" />
          )}
          <p className="faint" style={{ fontSize: '0.75rem' }}>
            A handful of large observations can drag the mean positive while the typical bet
            loses to the close — which is why the shape is shown, not just the average.
          </p>
        </div>
      </div>
    </div>
  );
}

function SkillSection({ skill }: { skill: any }) {
  if (!skill?.available) {
    return (
      <Card title="Is the CLV coming from the model?">
        <Empty title="Skill decomposition unavailable">{skill?.reason}</Empty>
      </Card>
    );
  }

  const components = skill.components;
  const control = skill.both_sides_control;

  return (
    <Card
      title="Is the CLV coming from the model?"
      note={components.note}
    >
      <div className="grid grid-4">
        <Tile label="Reported CLV" value={pctPoints(components.reported_clv_mean_pct)} />
        <Tile
          label="Line shopping"
          value={pctPoints(components.line_shopping_pct)}
          sub="best of many books — real money, no model"
        />
        <Tile
          label="Market drift"
          value={pctPoints(components.market_drift_pct)}
          sub="consensus open → close, applies to every selection"
        />
        <Tile
          label="Both-sides control"
          value={pctPoints(control?.mean_of_both_sides_pct)}
          sub={`${control?.n_games ?? 0} games — should sit near zero`}
        />
      </div>

      <div className="stack" style={{ marginTop: 14 }}>
        {Object.entries(skill.skill_tests ?? {}).map(([name, test]: [string, any]) => (
          <div key={name} className="card" style={{ padding: 12 }}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div>
                <strong style={{ fontSize: '0.85rem' }}>{name.replace(/_/g, ' ')}</strong>
                <div className="faint" style={{ fontSize: '0.72rem' }}>{test.basis}</div>
              </div>
              <Badge tone={test.significant ? (test.difference > 0 ? 'pos' : 'neg') : 'neutral'}>
                {test.significant ? 'significant' : 'not significant'}
              </Badge>
            </div>
            <div className="grid grid-3" style={{ marginTop: 10 }}>
              <Tile
                label="Recommended"
                value={pctPoints(test.recommended.mean)}
                sub={`n = ${test.recommended.n}`}
              />
              <Tile
                label="Rejected"
                value={pctPoints(test.rejected.mean)}
                sub={`n = ${test.rejected.n}`}
              />
              <Tile
                label="Difference"
                value={`${test.difference >= 0 ? '+' : ''}${num(test.difference, 2)}pp`}
                tone={test.significant ? (test.difference > 0 ? 'pos' : 'neg') : 'neutral'}
                sub={
                  test.ci_low != null
                    ? `95% CI ${num(test.ci_low, 2)} … ${num(test.ci_high, 2)}`
                    : 'interval needs a larger sample'
                }
              />
            </div>
            <p className="muted" style={{ fontSize: '0.8rem', marginTop: 8, marginBottom: 0 }}>
              {test.interpretation}
            </p>
          </div>
        ))}
      </div>
    </Card>
  );
}

function CohortSection({ cohorts }: { cohorts: Record<string, any[]> }) {
  const dimensions = Object.keys(cohorts);
  const [dimension, setDimension] = useState(dimensions[0]);
  const rows = cohorts[dimension] ?? [];

  return (
    <Card
      title="CLV by cohort"
      actions={
        <select value={dimension} onChange={(event) => setDimension(event.target.value)}>
          {dimensions.map((key) => (
            <option key={key} value={key}>
              {key.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      }
      note="Cohorts below the inference threshold report 'insufficient sample' rather than a number."
    >
      <DataTable
        columns={[
          { key: 'cohort', header: dimension.replace(/_/g, ' '), render: (row: any) => row.cohort },
          { key: 'n', header: 'n', numeric: true, render: (row: any) => row.n, sortValue: (row: any) => row.n },
          {
            key: 'mean',
            header: 'Mean CLV',
            numeric: true,
            render: (row: any) => (
              <span className={row.significant ? (row.mean_clv > 0 ? 'pos' : 'neg') : ''}>
                {pctPoints(row.mean_clv)}
              </span>
            ),
            sortValue: (row: any) => row.mean_clv,
          },
          {
            key: 'median',
            header: 'Median',
            numeric: true,
            render: (row: any) => pctPoints(row.median_clv),
          },
          {
            key: 'positive',
            header: 'Positive',
            numeric: true,
            render: (row: any) => (
              <span className="row" style={{ justifyContent: 'flex-end', gap: 6 }}>
                {pct(row.positive_rate)}
                <Meter value={row.positive_rate} />
              </span>
            ),
            sortValue: (row: any) => row.positive_rate,
          },
          {
            key: 'ci',
            header: '95% CI',
            render: (row: any) =>
              row.ci_low != null ? (
                <span className="mono faint" style={{ fontSize: '0.72rem' }}>
                  {num(row.ci_low, 2)} … {num(row.ci_high, 2)}
                </span>
              ) : (
                <span className="faint">insufficient</span>
              ),
          },
          {
            key: 'verdict',
            header: 'Reading',
            render: (row: any) => (
              <span className="faint" style={{ fontSize: '0.72rem' }}>{row.interpretation}</span>
            ),
          },
        ]}
        rows={rows}
        initialSort={{ key: 'n', direction: 'desc' }}
        emptyLabel="No cohort data"
      />
    </Card>
  );
}

function CoverageSection({ coverage }: { coverage: any }) {
  return (
    <Card
      title="Closing-line coverage"
      note="A CLV average computed over a small slice of games is not a portfolio-level statement, so coverage sits next to the number."
    >
      <div className="grid grid-2">
        <div>
          <h3 style={{ marginBottom: 8 }}>Games with a resolvable close</h3>
          <DataTable
            columns={[
              { key: 'sport', header: 'Sport', render: (row: any) => row.sport },
              { key: 'league', header: 'League', render: (row: any) => row.league_id ?? '—' },
              { key: 'final', header: 'Final', numeric: true, render: (row: any) => row.final_games },
              {
                key: 'declared',
                header: 'Declared close',
                numeric: true,
                render: (row: any) => row.declared_close,
              },
              {
                key: 'share',
                header: 'Coverage',
                numeric: true,
                render: (row: any) =>
                  row.final_games ? pct(row.declared_close / row.final_games) : '—',
              },
            ]}
            rows={coverage.coverage ?? []}
            emptyLabel="No coverage data"
          />
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Settlement state</h3>
          <dl className="kv">
            {Object.entries(coverage.settlement_state ?? {}).map(([state, count]) => (
              <div key={state} style={{ display: 'contents' }}>
                <dt>{state.replace(/_/g, ' ').toLowerCase()}</dt>
                <dd>{String(count)}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </Card>
  );
}
