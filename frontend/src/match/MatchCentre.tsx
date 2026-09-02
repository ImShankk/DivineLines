/* The Match Centre.

   One request builds the page; the replay slider re-fetches it with a server
   side bound. Tabs unlock from what the payload says is actually available, so
   a competition with no event feed does not get a Momentum tab that opens onto
   an apology.

   The whole page shares one selected event. Clicking a goal in the timeline
   marks it on the shot map and moves the replay to that minute — which is the
   difference between an intelligence tool and a page of unrelated charts. */

import { useEffect, useMemo, useState } from 'react';
import { ErrorState, Loading, useAsync } from '../ui';
import { DataQuality, StandingsPanel } from './DataQuality';
import { LineupBoard } from './LineupBoard';
import { MarketPanel, ModelPanel } from './MarketPanel';
import { MatchEvents } from './MatchEvents';
import { MatchHeader } from './MatchHeader';
import { MatchReportView } from './MatchReport';
import { MatchStats } from './MatchStats';
import { MomentumChart } from './MomentumChart';
import { Panel, LegendKey } from './Panel';
import { PassNetworkMap } from './PassNetworkMap';
import { EventHeatmap, ShotMap } from './Pitch';
import { PlayerStats } from './PlayerStats';
import { ReplayControls } from './ReplayControls';
import { WhatChanged } from './WhatChanged';
import {
  SHOT_LEGEND,
  soccerApi,
  type MatchCentre as MatchCentreData,
  type MatchEvent,
} from './api';

type Tab =
  | 'overview'
  | 'statistics'
  | 'lineups'
  | 'events'
  | 'players'
  | 'momentum'
  | 'spatial'
  | 'markets'
  | 'model'
  | 'report';

interface TabSpec {
  id: Tab;
  label: string;
  /** A tab appears only when the payload can fill it. */
  enabled: (data: MatchCentreData) => boolean;
}

const TABS: TabSpec[] = [
  { id: 'overview', label: 'Overview', enabled: () => true },
  { id: 'statistics', label: 'Statistics', enabled: (d) => d.statistics.comparisons.length > 0 },
  { id: 'lineups', label: 'Lineups', enabled: (d) => d.lineups.home.starters.length > 0 || d.lineups.away.starters.length > 0 },
  { id: 'events', label: 'Events', enabled: (d) => d.events.length > 0 },
  { id: 'players', label: 'Players', enabled: (d) => d.players.players.length > 0 },
  { id: 'momentum', label: 'Momentum', enabled: (d) => d.momentum.available },
  { id: 'spatial', label: 'Spatial', enabled: (d) => d.shots.located > 0 || d.heatmap.events_located > 0 },
  { id: 'markets', label: 'Markets', enabled: (d) => d.market.available },
  { id: 'model', label: 'Model', enabled: () => true },
  { id: 'report', label: 'Report', enabled: () => true },
];

export function MatchCentrePage({
  gameUid,
  onClose,
}: {
  gameUid: string;
  onClose: () => void;
}) {
  const [minute, setMinute] = useState<number | null>(null);
  const [requestedTab, setTab] = useState<Tab>('overview');
  const [selectedEvent, setSelectedEvent] = useState<MatchEvent | null>(null);
  const [shotSide, setShotSide] = useState<'both' | 'home' | 'away'>('both');

  const { data, error, loading, refresh } = useAsync(
    () => soccerApi.match(gameUid, minute),
    [gameUid, minute],
  );

  // A live match keeps itself current; a finished one has nothing to poll for.
  useEffect(() => {
    if (!data?.state.is_live || minute !== null) return;
    const timer = setInterval(refresh, 30_000);
    return () => clearInterval(timer);
  }, [data?.state.is_live, minute, refresh]);

  const available = useMemo(() => (data ? TABS.filter((spec) => spec.enabled(data)) : []), [data]);
  // Derived, not corrected after the fact: moving to a match whose feed cannot
  // fill the open tab should render Overview immediately, not render an empty
  // tab and then fix itself.
  const tab = available.some((spec) => spec.id === requestedTab) ? requestedTab : 'overview';

  if (loading && !data) return <Loading label="Loading match centre" />;
  if (error) return <ErrorState error={error} onRetry={refresh} />;
  if (!data) return null;

  const homeName = data.match.home.name;
  const awayName = data.match.away.name;
  const maxMinute = Math.max(
    90,
    ...data.events.map((event) => Math.ceil(event.minute ?? 0)),
  );

  const selectEvent = (event: MatchEvent | null) => {
    setSelectedEvent(event);
  };

  return (
    <div className="stack match-centre">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <button className="ghost" onClick={onClose}>
          ← All matches
        </button>
        <span className="faint mono">{data.match.game_uid}</span>
      </div>

      <MatchHeader data={data} />

      <ReplayControls
        minute={minute}
        onChange={setMinute}
        maxMinute={maxMinute}
        retrospective={data.replay.retrospective_events}
      />

      <nav className="match-tabs" role="tablist">
        {available.map((spec) => (
          <button
            key={spec.id}
            role="tab"
            aria-selected={tab === spec.id}
            className={tab === spec.id ? 'active' : ''}
            onClick={() => setTab(spec.id)}
          >
            {spec.label}
          </button>
        ))}
      </nav>

      {tab === 'overview' && (
        <div className="overview-grid">
          <Panel
            className="span-2"
            title="Momentum"
            state={data.momentum.available ? 'DATA' : 'NO_DATA'}
            reason={data.momentum.reason}
            note={data.momentum.note}
            legend={
              <>
                <LegendKey label={`${homeName} (above axis)`} color="var(--accent)" shape="square" />
                <LegendKey label={`${awayName} (below axis)`} color="var(--warning)" shape="square" />
                <LegendKey label="Goal" color="var(--positive)" shape="dot" />
                <LegendKey label="Card" color="var(--warning)" shape="square" />
              </>
            }
          >
            <MomentumChart
              momentum={data.momentum}
              homeName={homeName}
              awayName={awayName}
              onSelectMinute={(value) => setMinute(value)}
            />
          </Panel>

          <Panel title="Match data" state="DATA">
            <DataQuality quality={data.quality} replay={data.replay} />
          </Panel>

          <Panel
            title="Key events"
            state={data.events.length ? 'DATA' : 'NO_DATA'}
            reason="No play-by-play recorded for this match."
          >
            <MatchEvents
              events={data.events}
              homeName={homeName}
              awayName={awayName}
              selectedId={selectedEvent?.event_row_id ?? null}
              onSelect={selectEvent}
            />
          </Panel>

          <Panel
            title="Match statistics"
            state={data.statistics.comparisons.length ? 'DATA' : 'NO_DATA'}
            reason="No box score published for this match."
            note={data.statistics.note}
          >
            <MatchStats
              comparisons={data.statistics.comparisons}
              homeName={homeName}
              awayName={awayName}
              basis={data.statistics.basis}
              unavailable={data.statistics.unavailable}
            />
          </Panel>

          <Panel
            title="Shot map"
            state={data.shots.located ? 'DATA' : 'NO_DATA'}
            reason={data.shots.reason}
            legend={SHOT_LEGEND.map((entry) => (
              <LegendKey
                key={entry.outcome}
                label={entry.label}
                color={
                  entry.outcome === 'goal'
                    ? 'var(--positive)'
                    : entry.outcome === 'blocked'
                      ? 'var(--neutral)'
                      : 'var(--accent)'
                }
                shape={entry.shape === 'filled' ? 'dot' : entry.shape === 'cross' ? 'cross' : entry.shape === 'square' ? 'square' : 'ring'}
              />
            ))}
          >
            <ShotMap
              points={data.shots.points}
              totalShots={data.shots.total_shots}
              homeName={homeName}
              awayName={awayName}
              side={shotSide}
            />
          </Panel>

          <Panel title="Model and market" state="DATA">
            <ModelPanel data={data} />
          </Panel>

          <Panel
            className="span-2"
            title={`League table before this fixture${data.standings.as_of_date ? ` (${data.standings.as_of_date})` : ''}`}
            state={data.standings.available ? 'DATA' : 'NO_DATA'}
            reason={data.standings.reason}
            note={data.standings.note}
          >
            <StandingsPanel standings={data.standings} />
          </Panel>
        </div>
      )}

      {tab === 'statistics' && (
        <Panel title="Match statistics" state="DATA" note={data.statistics.note}>
          <MatchStats
            comparisons={data.statistics.comparisons}
            homeName={homeName}
            awayName={awayName}
            basis={data.statistics.basis}
            unavailable={data.statistics.unavailable}
          />
        </Panel>
      )}

      {tab === 'lineups' && (
        <Panel title="Lineups" state="DATA">
          <LineupBoard lineups={data.lineups} />
        </Panel>
      )}

      {tab === 'events' && (
        <Panel title="Event timeline" state="DATA">
          <MatchEvents
            events={data.events}
            homeName={homeName}
            awayName={awayName}
            selectedId={selectedEvent?.event_row_id ?? null}
            onSelect={selectEvent}
          />
        </Panel>
      )}

      {tab === 'players' && (
        <Panel title="Player statistics" state="DATA" note={data.players.rating_note}>
          <PlayerStats
            players={data.players.players}
            homeName={homeName}
            awayName={awayName}
            ratingNote={data.players.rating_note}
          />
        </Panel>
      )}

      {tab === 'momentum' && (
        <div className="stack">
          <Panel
            title="Momentum"
            state="DATA"
            note={`${data.momentum.note} Version ${data.momentum.summary?.version}.`}
          >
            <MomentumChart
              momentum={data.momentum}
              homeName={homeName}
              awayName={awayName}
              onSelectMinute={setMinute}
            />
          </Panel>
          <Panel title="Largest swings" state={data.momentum.swings.length ? 'DATA' : 'NO_DATA'}
                 reason="No swing exceeded the reporting threshold.">
            <ul className="swing-list">
              {data.momentum.swings.slice(0, 12).map((swing, index) => (
                <li key={index} className="swing-row">
                  <span className="mono">{swing.minute}′</span>
                  <span className={`num ${swing.change > 0 ? 'pos' : 'neg'}`}>
                    {swing.change > 0 ? '+' : ''}
                    {swing.change.toFixed(1)}
                  </span>
                  <span className="faint">
                    toward {swing.direction === 'home' ? homeName : awayName}
                  </span>
                  <span>
                    {swing.associated_events
                      .map((event) => `${event.event_type.replace(/_/g, ' ')}${event.player_name ? ` (${event.player_name})` : ''}`)
                      .join(', ') || 'no event recorded in this minute'}
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Methodology" state="DATA">
            <pre className="method-block">{JSON.stringify(data.momentum.parameters, null, 2)}</pre>
          </Panel>
        </div>
      )}

      {tab === 'spatial' && (
        <div className="stack">
          <Panel
            title="Shot map"
            state={data.shots.located ? 'DATA' : 'NO_DATA'}
            reason={data.shots.reason}
            actions={
              <div className="segmented">
                {(['both', 'home', 'away'] as const).map((option) => (
                  <button
                    key={option}
                    className={shotSide === option ? 'active' : ''}
                    onClick={() => setShotSide(option)}
                  >
                    {option === 'both' ? 'Both' : option === 'home' ? homeName : awayName}
                  </button>
                ))}
              </div>
            }
            legend={SHOT_LEGEND.map((entry) => (
              <LegendKey
                key={entry.outcome}
                label={entry.label}
                color={entry.outcome === 'goal' ? 'var(--positive)' : entry.outcome === 'blocked' ? 'var(--neutral)' : 'var(--accent)'}
                shape={entry.shape === 'filled' ? 'dot' : entry.shape === 'cross' ? 'cross' : entry.shape === 'square' ? 'square' : 'ring'}
              />
            ))}
          >
            <ShotMap
              points={data.shots.points}
              totalShots={data.shots.total_shots}
              homeName={homeName}
              awayName={awayName}
              side={shotSide}
            />
          </Panel>

          <Panel
            title="Event location density"
            state={data.heatmap.events_located ? 'DATA' : 'NO_DATA'}
            reason="No event in this match carries a field position."
            note={data.heatmap.note}
          >
            <EventHeatmap density={data.heatmap} homeName={homeName} awayName={awayName} />
          </Panel>

          <PassNetworkMap passing={data.passing} homeName={homeName} awayName={awayName} />
        </div>
      )}

      {tab === 'markets' && (
        <div className="stack">
          <Panel title="Market" state="DATA" note={data.market.note}>
            <MarketPanel market={data.market} />
          </Panel>
          <Panel title="What changed?" state="DATA">
            <WhatChanged gameUid={gameUid} />
          </Panel>
        </div>
      )}

      {tab === 'model' && (
        <div className="stack">
          <Panel title="DivineLines view" state="DATA">
            <ModelPanel data={data} />
          </Panel>
          <Panel title="Data behind this view" state="DATA" note={data.quality.note}>
            <DataQuality quality={data.quality} replay={data.replay} />
          </Panel>
        </div>
      )}

      {tab === 'report' && <MatchReportView gameUid={gameUid} minute={minute} />}
    </div>
  );
}
