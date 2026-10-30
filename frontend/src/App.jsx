import { useEffect, useState } from 'react';

import DashboardPage from './components/DashboardPage';
import EditorPage from './components/EditorPage';
import FleetPage from './components/FleetPage';
import InfoPage from './components/InfoPage';
import { checkBackend } from './lib/api';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'fleet', label: 'Live floor' },
  { id: 'editor', label: 'Editor' },
  { id: 'info', label: 'Info' },
];

const THEME_KEY = 'warehouse-rl-theme';

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  // Light stays the default; dark is one click for a projector that washes out.
  const [theme, setTheme] = useState(savedTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Private window or blocked storage: the toggle still works this visit.
    }
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    checkBackend().then((payload) => !cancelled && setHealth(payload));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Warehouse RL Dashboard</h1>
          <p className="subtitle">
            AI-based warehouse automation: reinforcement learning for intelligent robot navigation
          </p>
        </div>
        <div className="header-right">
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-pressed={theme === 'dark'}
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
          <span className={`badge ${health ? 'ok' : 'neutral'}`}>
            {health ? 'Backend connected' : 'Replaying recorded episodes'}
          </span>
        </div>
      </header>

      <nav className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab-button ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === 'dashboard' && <DashboardPage health={health} />}
      {tab === 'fleet' && <FleetPage />}
      {tab === 'editor' && <EditorPage backendHealth={health} />}
      {tab === 'info' && <InfoPage />}
    </div>
  );
}
