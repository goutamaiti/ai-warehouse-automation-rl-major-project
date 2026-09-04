import { useEffect, useState } from 'react';

import DashboardPage from './components/DashboardPage';
import EditorPage from './components/EditorPage';
import InfoPage from './components/InfoPage';
import { checkBackend } from './lib/api';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'editor', label: 'Editor' },
  { id: 'info', label: 'Info' },
];

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [health, setHealth] = useState(null);

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
        <span className={`badge ${health ? 'ok' : 'neutral'}`}>
          {health ? 'Backend connected' : 'Replaying recorded episodes'}
        </span>
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
      {tab === 'editor' && <EditorPage backendHealth={health} />}
      {tab === 'info' && <InfoPage />}
    </div>
  );
}
