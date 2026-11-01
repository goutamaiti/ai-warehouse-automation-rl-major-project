import { useEffect, useState } from 'react';

import EditorPage from './components/EditorPage';
import FleetPage from './components/FleetPage';
import HomePage from './components/HomePage';
import ResearchPage from './components/ResearchPage';
import { checkBackend } from './lib/api';
import { BRAND, PAGES, parseHash } from './lib/site';

const THEME_KEY = 'warehouse-rl-theme';

function savedTheme() {
  try {
    return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

/** The page named in the address bar, so links and the back button both work. */
function useRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => {
      setRoute(parseHash(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}

function Logo() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--accent)" />
      <path
        d="M9 16.5l5 5 9-11"
        fill="none"
        stroke="var(--accent-ink)"
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function App() {
  const [page, view] = useRoute();
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
    <div className="site">
      <header className="site-header">
        <div className="site-bar">
          <a className="brand" href="#/">
            <Logo />
            {BRAND}
          </a>
          <nav className="site-nav" aria-label="Main">
            {PAGES.map((item) => (
              <a
                key={item.id}
                href={item.href}
                className={page === item.id ? 'active' : ''}
                aria-current={page === item.id ? 'page' : undefined}
              >
                {item.label}
              </a>
            ))}
          </nav>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-pressed={theme === 'dark'}
          >
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      </header>

      {page === 'home' ? (
        <HomePage />
      ) : (
        <div className="app">
          {page === 'test' && <EditorPage backendHealth={health} />}
          {page === 'watch' && <FleetPage />}
          {page === 'research' && <ResearchPage health={health} view={view} />}
        </div>
      )}

      <footer className="site-footer">
        <p>
          {BRAND} is the working name of a final-year project: AI-Based Warehouse Automation
          System Using Reinforcement Learning for Intelligent Robot Navigation, Basaveshwar
          Engineering College, Bagalkote.
        </p>
        <p>Software simulation only. Results have not yet been compared with a real warehouse.</p>
      </footer>
    </div>
  );
}
