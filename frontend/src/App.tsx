/* Shell: hash routing, navigation, theme toggle and global search.
   Routing is hash-based on purpose — the app is a single static bundle served
   next to the API, and a router dependency would buy nothing here. */

import { useEffect, useState } from 'react';
import { api } from './api';
import { DashboardPage } from './pages/Dashboard';
import { GamesPage } from './pages/Games';
import { MatchesPage } from './pages/Matches';
import { OpportunitiesPage } from './pages/Opportunities';
import { HealthPage } from './pages/Health';
import { PerformancePage } from './pages/Performance';
import { ModelsPage, SystemPage, TeamsPage } from './pages/Reference';
import { StateDot } from './ui';

type Route =
  | 'dashboard'
  | 'opportunities'
  | 'games'
  | 'matches'
  | 'health'
  | 'performance'
  | 'teams'
  | 'models'
  | 'system';

const NAV: { group: string; items: { route: Route; label: string }[] }[] = [
  {
    group: 'Trading',
    items: [
      { route: 'dashboard', label: 'Dashboard' },
      { route: 'opportunities', label: '+EV Scanner' },
      { route: 'games', label: 'Games' },
      { route: 'matches', label: 'Match Centre' },
    ],
  },
  {
    group: 'Research',
    items: [
      { route: 'health', label: 'Model Health & CLV' },
      { route: 'performance', label: 'Performance' },
      { route: 'models', label: 'Models' },
    ],
  },
  {
    group: 'Reference',
    items: [
      { route: 'teams', label: 'Teams' },
      { route: 'system', label: 'System' },
    ],
  },
];

function readRoute(): { route: Route; param: string | null } {
  const hash = window.location.hash.replace(/^#\/?/, '');
  const [route, ...rest] = hash.split('/');
  const known = NAV.flatMap((group) => group.items).map((item) => item.route);
  return {
    route: (known.includes(route as Route) ? route : 'dashboard') as Route,
    param: rest.length ? decodeURIComponent(rest.join('/')) : null,
  };
}

export default function App() {
  const [{ route, param }, setLocation] = useState(readRoute);
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>(
    () => (localStorage.getItem('dl-theme') as any) ?? 'system',
  );
  const [health, setHealth] = useState<string>('unknown');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any>(null);

  useEffect(() => {
    const onHashChange = () => setLocation(readRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (theme === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('dl-theme', theme);
  }, [theme]);

  useEffect(() => {
    api.health()
      .then((response) => setHealth(response.status))
      .catch(() => setHealth('down'));
  }, []);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults(null);
      return;
    }
    const timer = setTimeout(() => {
      api.search(query).then(setResults).catch(() => setResults(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [query]);

  const navigate = (next: string) => {
    window.location.hash = `#/${next}`;
  };

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">
          <span className="brand-mark">DivineLines</span>
          <span className="brand-version">v5</span>
        </div>

        <div style={{ padding: '0 4px 14px' }}>
          <input
            type="search"
            placeholder="Search teams, games…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            style={{ width: '100%' }}
          />
          {results && (
            <div className="card" style={{ marginTop: 6, padding: 8, fontSize: '0.78rem' }}>
              {results.teams?.slice(0, 4).map((team: any) => (
                <div
                  key={team.team_uid}
                  className="nav-item"
                  onClick={() => {
                    navigate('teams');
                    setQuery('');
                  }}
                >
                  {team.canonical_name}
                  <span className="nav-count">{team.sport}</span>
                </div>
              ))}
              {results.games?.slice(0, 4).map((game: any) => (
                <div
                  key={game.game_uid}
                  className="nav-item"
                  onClick={() => {
                    navigate(`games/${encodeURIComponent(game.game_uid)}`);
                    setQuery('');
                  }}
                >
                  {game.away_name} @ {game.home_name}
                </div>
              ))}
              {!results.teams?.length && !results.games?.length && (
                <div className="faint" style={{ padding: 6 }}>
                  No matches
                </div>
              )}
            </div>
          )}
        </div>

        {NAV.map((group) => (
          <div className="nav-group" key={group.group}>
            <div className="nav-label">{group.group}</div>
            {group.items.map((item) => (
              <a
                key={item.route}
                className={`nav-item ${route === item.route ? 'active' : ''}`}
                href={`#/${item.route}`}
              >
                {item.label}
                {item.route === 'system' && <StateDot state={health} />}
              </a>
            ))}
          </div>
        ))}

        <div className="nav-group">
          <div className="nav-label">Theme</div>
          <div className="segmented" style={{ width: '100%' }}>
            {(['light', 'system', 'dark'] as const).map((option) => (
              <button
                key={option}
                className={theme === option ? 'active' : ''}
                onClick={() => setTheme(option)}
                style={{ flex: 1 }}
              >
                {option[0].toUpperCase() + option.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="main">
        {route === 'dashboard' && <DashboardPage onNavigate={navigate} />}
        {route === 'opportunities' && (
          <OpportunitiesPage onOpenGame={(uid) => navigate(`games/${encodeURIComponent(uid)}`)} />
        )}
        {route === 'games' && (
          <GamesPage
            selected={param}
            onOpenGame={(uid) => navigate(`games/${encodeURIComponent(uid)}`)}
            onClose={() => navigate('games')}
          />
        )}
        {route === 'matches' && (
          <MatchesPage
            selected={param}
            onOpenMatch={(uid) => navigate(`matches/${encodeURIComponent(uid)}`)}
            onClose={() => navigate('matches')}
          />
        )}
        {route === 'health' && <HealthPage />}
        {route === 'performance' && <PerformancePage />}
        {route === 'teams' && <TeamsPage />}
        {route === 'models' && <ModelsPage />}
        {route === 'system' && <SystemPage />}
      </main>
    </div>
  );
}
