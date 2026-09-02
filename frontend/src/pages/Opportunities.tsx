/* +EV scanner: the page that answers "why does the system like this bet?" */

import { useMemo, useState } from 'react';
import {
  api,
  american,
  money,
  num,
  pct,
  shortDate,
  signedPct,
  timeAgo,
  type Opportunity,
  type Sport,
} from '../api';
import {
  Badge,
  Banner,
  Card,
  DataTable,
  Empty,
  ErrorState,
  Field,
  FlagList,
  Loading,
  Meter,
  ProbabilityBar,
  Segmented,
  Tile,
  useAsync,
  type Column,
} from '../ui';

const SELECTION_LABEL: Record<string, string> = {
  home: 'Home',
  away: 'Away',
  draw: 'Draw',
};

export function OpportunitiesPage({ onOpenGame }: { onOpenGame: (uid: string) => void }) {
  const [sport, setSport] = useState<Sport>('nba');
  const [days, setDays] = useState(3);
  const [minEdge, setMinEdge] = useState(0);
  const [minScore, setMinScore] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, error, loading, refresh } = useAsync(
    () => api.predictions(sport, days),
    [sport, days],
  );

  const rows = useMemo(() => {
    if (!data) return [];
    const source = showAll ? data.predictions : data.opportunities;
    return source.filter(
      (row) =>
        (row.edge ?? -1) >= minEdge / 100 &&
        row.edge_score >= minScore,
    );
  }, [data, showAll, minEdge, minScore]);

  const columns: Column<Opportunity>[] = [
    {
      key: 'game',
      header: 'Match',
      render: (row) => (
        <div>
          <div style={{ fontWeight: 600 }}>
            {row.away_name} @ {row.home_name}
          </div>
          <div className="faint" style={{ fontSize: '0.72rem' }}>
            {shortDate(row.game_date)} · {row.league_id} · {row.market}
          </div>
        </div>
      ),
      sortValue: (row) => row.game_date,
    },
    {
      key: 'selection',
      header: 'Pick',
      render: (row) => <Badge tone="accent">{SELECTION_LABEL[row.selection] ?? row.selection}</Badge>,
      sortValue: (row) => row.selection,
    },
    {
      key: 'prob',
      header: 'Model / Market',
      render: (row) => (
        <div className="row" style={{ gap: 8 }}>
          <span className="num">{pct(row.model_probability)}</span>
          <span className="faint num">/ {pct(row.market_probability)}</span>
          <ProbabilityBar model={row.model_probability} market={row.market_probability} />
        </div>
      ),
      sortValue: (row) => row.model_probability,
    },
    {
      key: 'edge',
      header: 'Edge',
      numeric: true,
      render: (row) => (
        <span className={(row.edge ?? 0) > 0 ? 'pos' : 'neg'}>{signedPct(row.edge, 2)}</span>
      ),
      sortValue: (row) => row.edge ?? -1,
    },
    {
      key: 'price',
      header: 'Price',
      numeric: true,
      render: (row) =>
        row.price_decimal ? (
          <span title={`${row.bookmaker ?? 'best price'} · ${row.n_bookmakers} books`}>
            {num(row.price_decimal, 2)}{' '}
            <span className="faint">({american(row.price_decimal)})</span>
          </span>
        ) : (
          <span className="faint">—</span>
        ),
      sortValue: (row) => row.price_decimal ?? 0,
    },
    {
      key: 'ev',
      header: 'EV/unit',
      numeric: true,
      render: (row) => (
        <span className={(row.ev_per_unit ?? 0) > 0 ? 'pos' : 'muted'}>
          {row.ev_per_unit == null ? '—' : signedPct(row.ev_per_unit, 2)}
        </span>
      ),
      sortValue: (row) => row.ev_per_unit ?? -1,
    },
    {
      key: 'score',
      header: 'Edge score',
      numeric: true,
      render: (row) => (
        <div className="row" style={{ justifyContent: 'flex-end', gap: 6 }}>
          <span>{num(row.edge_score, 1)}</span>
          <Meter value={row.edge_score} max={10} />
        </div>
      ),
      sortValue: (row) => row.edge_score,
    },
    {
      key: 'conf',
      header: 'Conf.',
      numeric: true,
      render: (row) => num(row.confidence, 2),
      sortValue: (row) => row.confidence,
    },
    {
      key: 'quality',
      header: 'Data',
      numeric: true,
      render: (row) => (
        <span title={row.quality_detail?.components?.map((c) => `${c.name}: ${(c.score * 100).toFixed(0)}%`).join('\n')}>
          {num(row.data_quality, 0)}
        </span>
      ),
      sortValue: (row) => row.data_quality,
    },
    {
      key: 'stake',
      header: 'Stake',
      numeric: true,
      render: (row) => (row.stake > 0 ? money(row.stake) : <span className="faint">—</span>),
      sortValue: (row) => row.stake,
    },
    {
      key: 'flags',
      header: 'Flags',
      render: (row) => <FlagList flags={row.flags} />,
    },
  ];

  const totalStake = rows.reduce((sum, row) => sum + row.stake, 0);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>+EV Scanner</h1>
          <p className="page-sub">
            Model probability against de-vigged multi-book consensus. A selection only
            qualifies when the edge, the edge-quality score and the risk limits all
            agree — everything else is shown but not staked.
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
          <Field label="Days">
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {[1, 3, 5, 7, 14].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Min edge %">
            <input
              type="number"
              step={0.5}
              value={minEdge}
              style={{ width: 78 }}
              onChange={(e) => setMinEdge(Number(e.target.value))}
            />
          </Field>
          <Field label="Min score">
            <input
              type="number"
              step={0.5}
              value={minScore}
              style={{ width: 78 }}
              onChange={(e) => setMinScore(Number(e.target.value))}
            />
          </Field>
          <button onClick={refresh}>Refresh</button>
        </div>
      </div>

      {loading && <Loading label="Building features and pricing the slate" />}
      {error && <ErrorState error={error} onRetry={refresh} />}

      {data && (
        <>
          <div className="grid grid-4">
            <Tile label="Qualifying" value={data.opportunities.length} sub={`of ${data.predictions.length} priced selections`} />
            <Tile label="Total stake" value={money(totalStake)} sub={`${pct(data.portfolio?.exposure_pct ?? 0)} of bankroll`} />
            <Tile
              label="Best edge"
              value={signedPct(Math.max(...data.opportunities.map((o) => o.edge ?? 0), 0), 2)}
              tone={data.opportunities.length ? 'pos' : 'neutral'}
            />
            <Tile label="Generated" value={timeAgo(data.generated_at)} sub={data.cached ? 'from cache' : 'fresh'} />
          </div>

          {data.notes.map((note) => (
            <Banner key={note} tone="info">
              {note}
            </Banner>
          ))}
          {data.warnings.map((warning) => (
            <Banner key={warning} tone="warn">
              {warning}
            </Banner>
          ))}

          <Card
            title={showAll ? 'All priced selections' : 'Qualifying opportunities'}
            actions={
              <button className="ghost" onClick={() => setShowAll((v) => !v)}>
                {showAll ? 'Show qualifying only' : 'Show all selections'}
              </button>
            }
            note="Edge is measured against the no-vig consensus, not the raw price — comparing against raw implied probability would manufacture an edge the size of the bookmaker's margin."
          >
            {rows.length ? (
              <DataTable
                columns={columns}
                rows={rows}
                initialSort={{ key: 'score', direction: 'desc' }}
                onRowClick={(row) =>
                  setExpanded(expanded === row.game_uid + row.selection ? null : row.game_uid + row.selection)
                }
              />
            ) : (
              <Empty title="No selections match these filters">
                {showAll
                  ? 'No fixtures are priced in this window.'
                  : 'The scanner found no bet clearing the edge, quality and risk thresholds. That is a valid answer, not a failure.'}
              </Empty>
            )}
          </Card>

          {expanded && (
            <OpportunityDetail
              opportunity={
                [...data.predictions, ...data.opportunities].find(
                  (o) => o.game_uid + o.selection === expanded,
                )!
              }
              onOpenGame={onOpenGame}
            />
          )}
        </>
      )}
    </div>
  );
}

function OpportunityDetail({
  opportunity,
  onOpenGame,
}: {
  opportunity: Opportunity;
  onOpenGame: (uid: string) => void;
}) {
  if (!opportunity) return null;
  const availability = opportunity.availability;

  return (
    <Card
      title={`Why: ${opportunity.away_name} @ ${opportunity.home_name} — ${opportunity.selection}`}
      actions={<button onClick={() => onOpenGame(opportunity.game_uid)}>Open game</button>}
    >
      <div className="grid grid-3">
        <div>
          <h3 style={{ marginBottom: 8 }}>Model agreement</h3>
          <dl className="kv">
            {Object.entries(opportunity.components).map(([name, value]) => (
              <div key={name} style={{ display: 'contents' }}>
                <dt>{name}</dt>
                <dd>{pct(value)}</dd>
              </div>
            ))}
            <dt>agreement</dt>
            <dd>{num(opportunity.agreement, 2)}</dd>
            <dt>model version</dt>
            <dd style={{ fontSize: '0.72rem' }}>{opportunity.model_version ?? '—'}</dd>
          </dl>
        </div>

        <div>
          <h3 style={{ marginBottom: 8 }}>Edge score components</h3>
          <dl className="kv">
            {opportunity.edge_detail?.components?.map((component) => (
              <div key={component.name} style={{ display: 'contents' }}>
                <dt title={component.note ?? undefined}>
                  {component.name.replace(/_/g, ' ')} <span className="faint">×{component.weight}</span>
                </dt>
                <dd>{num(component.score, 2)}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div>
          <h3 style={{ marginBottom: 8 }}>Data quality</h3>
          <dl className="kv">
            {opportunity.quality_detail?.components?.map((component) => (
              <div key={component.name} style={{ display: 'contents' }}>
                <dt title={component.note ?? undefined}>{component.name.replace(/_/g, ' ')}</dt>
                <dd>{num(component.score * 100, 0)}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {opportunity.explanation?.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginBottom: 8 }}>Top feature contributions</h3>
          <div className="row" style={{ gap: 6 }}>
            {opportunity.explanation.map((item) => (
              <Badge key={item.feature} tone={item.direction === 'home' ? 'pos' : 'neg'}>
                {item.feature.replace(/_/g, ' ')} {item.contribution > 0 ? '+' : ''}
                {item.contribution.toFixed(3)}
              </Badge>
            ))}
          </div>
          <p className="faint" style={{ fontSize: '0.75rem', marginTop: 6 }}>
            Exact SHAP contributions from the boosted trees, in log-odds toward the home side.
          </p>
        </div>
      )}

      {availability && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ marginBottom: 8 }}>Availability adjustment</h3>
          <dl className="kv">
            <dt>base probability</dt>
            <dd>{pct(availability.base_probability)}</dd>
            <dt>adjusted</dt>
            <dd>{pct(availability.adjusted_probability)}</dd>
            <dt>margin change</dt>
            <dd>{num(availability.margin_delta, 2)} pts</dd>
            <dt>uncertainty</dt>
            <dd>± {num(availability.uncertainty_margin, 2)} pts</dd>
          </dl>
          {['home', 'away'].map((side) => {
            const team = availability[side];
            if (!team?.missing_players?.length && !team?.uncertain_players?.length) return null;
            return (
              <div key={side} style={{ marginTop: 8 }}>
                <div className="muted" style={{ fontSize: '0.75rem', marginBottom: 4 }}>
                  {side} — {num(team.expected_margin_delta, 2)} pts
                </div>
                <div className="row" style={{ gap: 5 }}>
                  {team.missing_players?.map((player: any) => (
                    <Badge key={player.player} tone="neg" title={`${player.status}, ${player.minutes} mpg`}>
                      {player.player} −{num(player.margin_impact, 1)}
                    </Badge>
                  ))}
                  {team.uncertain_players?.map((player: any) => (
                    <Badge key={player.player} tone="warn" title={`${player.status}, ${(player.play_probability * 100).toFixed(0)}% to play`}>
                      {player.player} ?{num(player.margin_impact, 1)}
                    </Badge>
                  ))}
                </div>
              </div>
            );
          })}
          <p className="faint" style={{ fontSize: '0.75rem', marginTop: 6 }}>
            Player impact is PIE-minutes over replacement; the margin-to-probability
            conversion is fitted on this platform's own games. Injury features are not
            in the trained model — no historical injury data exists to validate them —
            so this is a transparent post-model adjustment.
          </p>
        </div>
      )}
    </Card>
  );
}
