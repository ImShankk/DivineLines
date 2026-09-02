/* Soccer match index.

   The counts on each row are the point of this page. A list that does not say
   which matches have an event feed sends the reader into an empty Match Centre
   to find out — so coverage is shown before the click, not after it. */

import { useState } from 'react';
import { shortDate } from '../api';
import { MatchCentrePage } from '../match/MatchCentre';
import { soccerApi, type SoccerMatchRow } from '../match/api';
import { Badge, Card, DataTable, ErrorState, Field, Loading, Segmented, useAsync, type Column } from '../ui';

const LEAGUES = [
  { value: '', label: 'All' },
  { value: 'ENG_PL', label: 'Premier League' },
  { value: 'ESP_LL', label: 'La Liga' },
  { value: 'ITA_SA', label: 'Serie A' },
  { value: 'GER_BL', label: 'Bundesliga' },
  { value: 'FRA_L1', label: 'Ligue 1' },
  { value: 'NED_ED', label: 'Eredivisie' },
  { value: 'POR_PL', label: 'Primeira Liga' },
  { value: 'ENG_CH', label: 'Championship' },
];

function Coverage({ count, label }: { count: number; label: string }) {
  if (!count) return <span className="faint" title={`No ${label}`}>—</span>;
  return (
    <span className="num" title={`${count} ${label}`}>
      {count}
    </span>
  );
}

export function MatchesPage({
  selected,
  onOpenMatch,
  onClose,
}: {
  selected: string | null;
  onOpenMatch: (gameUid: string) => void;
  onClose: () => void;
}) {
  const [league, setLeague] = useState('ENG_PL');
  const [status, setStatus] = useState<'final' | 'scheduled'>('final');
  const [onlyWithEvents, setOnlyWithEvents] = useState(true);

  const { data, error, loading, refresh } = useAsync(
    () =>
      soccerApi.matches({
        league_id: league || undefined,
        status,
        withEvents: onlyWithEvents,
        limit: 120,
      }),
    [league, status, onlyWithEvents],
  );

  if (selected) return <MatchCentrePage gameUid={selected} onClose={onClose} />;

  const columns: Column<SoccerMatchRow>[] = [
    {
      key: 'date',
      header: 'Date',
      render: (row) => shortDate(row.game_date),
      sortValue: (row) => row.game_date,
    },
    { key: 'league', header: 'League', render: (row) => <Badge>{row.league_id}</Badge> },
    {
      key: 'match',
      header: 'Match',
      render: (row) => (
        <span style={{ fontWeight: 600 }}>
          {row.home_name} <span className="faint">v</span> {row.away_name}
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
          <span className="num">
            {row.home_score}–{row.away_score}
          </span>
        ) : (
          <span className="faint">—</span>
        ),
    },
    { key: 'events', header: 'Events', numeric: true, render: (row) => <Coverage count={row.events} label="events" />, sortValue: (row) => row.events },
    { key: 'starters', header: 'Lineup', numeric: true, render: (row) => <Coverage count={row.starters} label="starters" />, sortValue: (row) => row.starters },
    { key: 'prices', header: 'Prices', numeric: true, render: (row) => <Coverage count={row.prices} label="price snapshots" />, sortValue: (row) => row.prices },
    { key: 'predictions', header: 'Model', numeric: true, render: (row) => <Coverage count={row.predictions} label="predictions" />, sortValue: (row) => row.predictions },
  ];

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Match Centre</h1>
          <p className="page-sub">
            Soccer fixtures with the event feed, lineups, prices and predictions the platform
            holds for each. Open a match for its timeline, momentum, shot map, statistics and
            the model-versus-market view.
          </p>
        </div>
        <div className="controls">
          <Field label="League">
            <select value={league} onChange={(event) => setLeague(event.target.value)}>
              {LEAGUES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Field>
          <Segmented
            value={status}
            onChange={setStatus}
            options={[
              { value: 'final', label: 'Results' },
              { value: 'scheduled', label: 'Upcoming' },
            ]}
          />
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={onlyWithEvents}
              onChange={(event) => setOnlyWithEvents(event.target.checked)}
            />
            With event data
          </label>
        </div>
      </div>

      {error && <ErrorState error={error} onRetry={refresh} />}
      {loading && !data && <Loading label="Loading matches" />}

      {data && (
        <Card
          title={`${data.count} match${data.count === 1 ? '' : 'es'}`}
          note="Event coverage comes from ESPN's play-by-play feed and is currently ingested for the Premier League only. Other competitions show fixtures, prices and results but no in-match events."
        >
          <DataTable
            columns={columns}
            rows={data.matches}
            emptyLabel="No matches match these filters"
            onRowClick={(row) => onOpenMatch(row.game_uid)}
            initialSort={{ key: 'date', direction: 'desc' }}
          />
        </Card>
      )}
    </div>
  );
}
