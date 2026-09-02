/* Player lines, grouped by side and shown with the metrics their position
   calls for. A goalkeeper's row is about shot stopping; an outfielder's is
   about what they did with the ball.

   There is no rating column. No source publishes one for this competition and
   DivineLines does not compute one, so inventing a number out of five box-score
   counters and calling it a rating would be exactly the kind of confident
   fiction this platform is meant to avoid. */

import { useState } from 'react';
import { Badge, Segmented } from '../ui';
import type { PlayerLine } from './api';

function Row({ player }: { player: PlayerLine }) {
  return (
    <tr className={player.starter ? '' : 'player-sub'}>
      <td className="mono faint">{player.jersey ?? '–'}</td>
      <td>
        <span className="row" style={{ gap: 6 }}>
          <span style={{ fontWeight: player.starter ? 600 : 400 }}>{player.player_name}</span>
          {player.subbed_out && <Badge tone="neutral">off</Badge>}
          {player.subbed_in && <Badge tone="accent">on</Badge>}
        </span>
      </td>
      <td className="faint">{player.position ?? player.position_group}</td>
      {player.stats.map((stat) => (
        <td key={stat.stat} className="right num">
          {stat.value === null ? <span className="faint">—</span> : stat.display ?? stat.value}
        </td>
      ))}
    </tr>
  );
}

function SideTable({ players, title }: { players: PlayerLine[]; title: string }) {
  if (!players.length) return null;
  const keepers = players.filter((player) => player.position_group === 'goalkeeper');
  const outfield = players.filter((player) => player.position_group !== 'goalkeeper');

  const table = (rows: PlayerLine[], label: string) => {
    if (!rows.length) return null;
    const headers = rows[0].stats.map((stat) => stat.label);
    return (
      <div className="table-wrap" key={label}>
        <div className="lineup-subhead">{label}</div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Player</th>
              <th>Pos</th>
              {headers.map((header) => (
                <th key={header} className="right">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((player) => (
              <Row key={player.player_uid ?? player.player_name} player={player} />
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="stack" style={{ gap: 10 }}>
      <strong>{title}</strong>
      {table(keepers, 'Goalkeeper')}
      {table(outfield, 'Outfield')}
    </div>
  );
}

export function PlayerStats({
  players,
  homeName,
  awayName,
  ratingNote,
}: {
  players: PlayerLine[];
  homeName: string;
  awayName: string;
  ratingNote: string;
}) {
  const [side, setSide] = useState<'home' | 'away'>('home');
  if (!players.length) {
    return <div className="empty">No per-player statistics published for this match.</div>;
  }

  const selected = players.filter((player) => player.home_away === side);

  return (
    <div className="stack">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <Segmented
          value={side}
          onChange={setSide}
          options={[
            { value: 'home', label: homeName },
            { value: 'away', label: awayName },
          ]}
        />
        <span className="faint">{ratingNote}</span>
      </div>
      <SideTable players={selected} title={side === 'home' ? homeName : awayName} />
    </div>
  );
}
