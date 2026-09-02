/* Teams, models and system pages. Grouped because each is a single view over
   one part of the platform's state. */

import { useState } from 'react';
import { api, num, pct, shortDate, timeAgo, type Sport } from '../api';
import {
  Badge,
  Banner,
  Card,
  DataTable,
  ErrorState,
  Field,
  Loading,
  Segmented,
  StateDot,
  Tile,
  useAsync,
  type Column,
} from '../ui';

/* --------------------------------------------------------------- teams */

export function TeamsPage() {
  const [sport, setSport] = useState<Sport>('nba');
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const { data, error, loading, refresh } = useAsync(() => api.teams(sport), [sport]);

  const teams = (data?.teams ?? []).filter((team: any) =>
    team.canonical_name.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Teams</h1>
          <p className="page-sub">
            Canonical entities. Every source name resolves to one of these, so
            "LA Lakers" and "Los Angeles Lakers" can never become two teams.
          </p>
        </div>
        <div className="controls">
          <Segmented
            value={sport}
            onChange={(value) => {
              setSport(value);
              setSelected(null);
            }}
            options={[
              { value: 'nba', label: 'NBA' },
              { value: 'soccer', label: 'Soccer' },
            ]}
          />
          <Field label="Filter">
            <input
              type="search"
              value={query}
              placeholder="Team name"
              onChange={(e) => setQuery(e.target.value)}
            />
          </Field>
        </div>
      </div>

      {loading && <Loading />}
      {error && <ErrorState error={error} onRetry={refresh} />}

      <div className="grid grid-2">
        {data && (
          <Card title={`${teams.length} teams`}>
            <DataTable
              columns={[
                {
                  key: 'name',
                  header: 'Team',
                  render: (row: any) => row.canonical_name,
                  sortValue: (row: any) => row.canonical_name,
                },
                { key: 'abbr', header: 'Abbr', render: (row: any) => row.abbr ?? '—' },
                { key: 'uid', header: 'UID', render: (row: any) => <span className="faint mono" style={{ fontSize: '0.7rem' }}>{row.team_uid}</span> },
              ]}
              rows={teams}
              onRowClick={(row: any) => setSelected(row.team_uid)}
              emptyLabel="No teams match"
            />
          </Card>
        )}
        {selected && <TeamDetail teamUid={selected} />}
      </div>
    </div>
  );
}

function TeamDetail({ teamUid }: { teamUid: string }) {
  const { data, error, loading } = useAsync(() => api.team(teamUid), [teamUid]);
  if (loading) return <Card title="Team"><Loading /></Card>;
  if (error) return <Card title="Team"><ErrorState error={error} /></Card>;
  if (!data) return null;

  return (
    <Card title={data.team.canonical_name}>
      <h3 style={{ marginBottom: 8 }}>Recent results</h3>
      <div className="table-wrap">
        <table className="dl">
          <thead>
            <tr>
              <th>Date</th>
              <th>Match</th>
              <th className="num">Score</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {data.recent_games.map((game: any) => {
              const isHome = game.home_team_uid === teamUid;
              const own = isHome ? game.home_score : game.away_score;
              const opponent = isHome ? game.away_score : game.home_score;
              const result = own > opponent ? 'W' : own === opponent ? 'D' : 'L';
              return (
                <tr key={game.game_uid}>
                  <td>{shortDate(game.game_date)}</td>
                  <td>
                    {isHome ? 'vs' : '@'} {isHome ? game.away_name : game.home_name}
                  </td>
                  <td className="num">
                    {own} – {opponent}
                  </td>
                  <td>
                    <Badge tone={result === 'W' ? 'pos' : result === 'L' ? 'neg' : 'neutral'}>
                      {result}
                    </Badge>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {data.availability?.length > 0 && (
        <>
          <h3 style={{ margin: '16px 0 8px' }}>Availability</h3>
          <div className="row" style={{ gap: 5 }}>
            {data.availability.map((player: any) => (
              <Badge
                key={player.full_name}
                tone={player.status === 'out' || player.status === 'suspended' ? 'neg' : 'warn'}
                title={`${player.detail ?? ''}${player.expected_return ? ` · back ${player.expected_return}` : ''}`}
              >
                {player.full_name} · {player.status}
              </Badge>
            ))}
          </div>
        </>
      )}
    </Card>
  );
}

/* -------------------------------------------------------------- models */

export function ModelsPage() {
  const { data, error, loading, refresh } = useAsync(() => api.models(), []);
  const experiments = useAsync(() => api.experiments(), []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Models</h1>
          <p className="page-sub">
            Every registered model with the data it saw, the features it used and its
            validation metrics — so results are reproducible and version-to-version
            comparisons are possible.
          </p>
        </div>
      </div>

      {(data?.models ?? []).map((model) => {
        const validation = model.metrics?.validation ?? model.metrics?.test;
        const weights = model.metrics?.blend_weights ?? {};
        let features: string[] = [];
        try {
          features = JSON.parse(model.feature_set);
        } catch { /* stored as text */ }

        return (
          <Card
            key={model.model_id}
            title={`${model.sport.toUpperCase()} · ${model.model_version}`}
            actions={<span className="faint">{timeAgo(model.trained_at)}</span>}
          >
            <div className="grid grid-4">
              <Tile label="Log loss" value={num(validation?.log_loss, 4)} sub="validation" />
              <Tile label="Brier" value={num(validation?.brier, 4)} />
              <Tile
                label="Accuracy"
                value={validation?.accuracy ? pct(validation.accuracy) : '—'}
              />
              <Tile label="Features" value={features.length || '—'} sub={model.feature_set_version} />
            </div>

            <div className="grid grid-2" style={{ marginTop: 14 }}>
              <div>
                <h3 style={{ marginBottom: 8 }}>Provenance</h3>
                <dl className="kv">
                  <dt>model id</dt>
                  <dd style={{ fontSize: '0.7rem' }}>{model.model_id}</dd>
                  <dt>train window</dt>
                  <dd>
                    {model.train_start} → {model.train_end}
                  </dd>
                  <dt>validation</dt>
                  <dd>
                    {model.valid_start} → {model.valid_end}
                  </dd>
                  <dt>data version</dt>
                  <dd>{model.data_version ?? '—'}</dd>
                  <dt>seed</dt>
                  <dd>{model.random_seed ?? '—'}</dd>
                  <dt>notes</dt>
                  <dd style={{ fontSize: '0.72rem' }}>{model.notes ?? '—'}</dd>
                </dl>
              </div>
              <div>
                <h3 style={{ marginBottom: 8 }}>Ensemble weights</h3>
                {Object.keys(weights).length ? (
                  <dl className="kv">
                    {Object.entries(weights).map(([name, weight]: [string, any]) => (
                      <div key={name} style={{ display: 'contents' }}>
                        <dt>{name}</dt>
                        <dd>{pct(weight)}</dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <span className="faint">—</span>
                )}
                {features.length > 0 && (
                  <details className="explain" style={{ marginTop: 10 }}>
                    <summary>{features.length} features</summary>
                    <div className="faint mono" style={{ fontSize: '0.7rem', marginTop: 6 }}>
                      {features.join(', ')}
                    </div>
                  </details>
                )}
              </div>
            </div>
          </Card>
        );
      })}

      {experiments.data?.experiments?.length > 0 && (
        <Card
          title="Feature ablation"
          note="A feature stays only if it improves out-of-sample performance. Variants are re-tested with `divinelines ablate`."
        >
          <DataTable
            columns={[
              { key: 'variant', header: 'Variant', render: (row: any) => row.variant },
              { key: 'sport', header: 'Sport', render: (row: any) => row.sport },
              {
                key: 'logloss',
                header: 'Log loss',
                numeric: true,
                render: (row: any) => {
                  try {
                    return num(JSON.parse(row.metrics).log_loss, 5);
                  } catch {
                    return '—';
                  }
                },
                sortValue: (row: any) => {
                  try {
                    return JSON.parse(row.metrics).log_loss;
                  } catch {
                    return 99;
                  }
                },
              },
              {
                key: 'n',
                header: 'Games',
                numeric: true,
                render: (row: any) => row.n_valid ?? '—',
              },
              { key: 'when', header: 'Run', render: (row: any) => timeAgo(row.created_at) },
            ]}
            rows={experiments.data.experiments}
            initialSort={{ key: 'logloss', direction: 'asc' }}
          />
        </Card>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- system */

export function SystemPage() {
  const { data, error, loading, refresh } = useAsync(() => api.health(), []);
  const config = useAsync(() => api.config(), []);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const columns: Column<any>[] = [
    {
      key: 'state',
      header: '',
      render: (row) => <StateDot state={row.state} />,
    },
    { key: 'source', header: 'Source', render: (row) => row.source },
    { key: 'dataset', header: 'Dataset', render: (row) => <span className="faint">{row.dataset}</span> },
    { key: 'status', header: 'Status', render: (row) => row.status ?? '—' },
    {
      key: 'age',
      header: 'Last success',
      numeric: true,
      render: (row) => timeAgo(row.last_success),
      sortValue: (row) => row.age_minutes ?? 1e9,
    },
    {
      key: 'message',
      header: 'Message',
      render: (row) => <span className="faint" style={{ fontSize: '0.72rem' }}>{row.message ?? ''}</span>,
    },
  ];

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>System</h1>
          <p className="page-sub">
            Source health, data freshness and the risk policy currently in force.
            Stale data is surfaced rather than silently served as if it were live.
          </p>
        </div>
        <div className="row">
          <StateDot state={data.status} />
          <span className="muted">
            {data.status} · checked {timeAgo(data.checked_at)}
          </span>
          <button onClick={refresh}>Refresh</button>
        </div>
      </div>

      {!data.validation.ok && (
        <Banner tone="bad">
          <div>
            <strong>Data validation failing</strong>
            <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
              {data.validation.critical.map((issue: any) => (
                <li key={issue.code}>
                  {issue.code}: {issue.detail}
                </li>
              ))}
            </ul>
          </div>
        </Banner>
      )}

      <div className="grid grid-4">
        {(data.database.games ?? []).map((row: any) => (
          <Tile
            key={row.sport}
            label={`${row.sport} games`}
            value={row.games?.toLocaleString?.() ?? row.games}
            sub={`latest ${shortDate(row.latest)}`}
          />
        ))}
        <Tile label="Odds snapshots" value={data.database.odds_snapshots?.toLocaleString?.() ?? '—'} />
        <Tile label="Predictions" value={data.database.predictions?.toLocaleString?.() ?? '—'} />
      </div>

      <Card title="Sources">
        <DataTable columns={columns} rows={data.sources} emptyLabel="No source activity recorded" />
      </Card>

      {config.data && (
        <div className="grid grid-2">
          <Card title="Risk policy">
            <dl className="kv">
              {Object.entries(config.data.betting).map(([key, value]: [string, any]) => (
                <div key={key} style={{ display: 'contents' }}>
                  <dt>{key.replace(/_/g, ' ')}</dt>
                  <dd>{typeof value === 'number' && value < 1 && value > 0 ? value : String(value)}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card title="Environment">
            <dl className="kv">
              <dt>mode</dt>
              <dd>{config.data.mode}</dd>
              <dt>calibration</dt>
              <dd>{config.data.model.calibration}</dd>
              <dt>random seed</dt>
              <dd>{config.data.model.random_seed}</dd>
              <dt>odds API key</dt>
              <dd>{config.data.odds_api_configured ? 'configured' : 'missing'}</dd>
              <dt>soccer leagues</dt>
              <dd style={{ fontSize: '0.72rem' }}>
                {config.data.leagues.soccer.map((league: any) => league.id).join(', ')}
              </dd>
            </dl>
          </Card>
        </div>
      )}
    </div>
  );
}
