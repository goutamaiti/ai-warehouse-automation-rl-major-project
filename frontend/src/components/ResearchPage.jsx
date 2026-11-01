import DashboardPage from './DashboardPage';
import InfoPage from './InfoPage';

const VIEWS = [
  { id: 'findings', label: 'Findings', href: '#/research' },
  { id: 'method', label: 'How it was built', href: '#/research/method' },
];

/** The academic project: the question, what was measured, and how. */
export default function ResearchPage({ health, view }) {
  const current = view === 'method' ? 'method' : 'findings';
  return (
    <div className="research">
      <header className="page-head">
        <h1>Can AI learn to drive a warehouse robot?</h1>
        <p>
          The research behind the project. A robot learned by trial and error (reinforcement
          learning) in a simulated warehouse, then ran against a classic route planner on identical
          runs.
        </p>
        <nav className="sub-nav" aria-label="Research">
          {VIEWS.map((item) => (
            <a key={item.id} href={item.href} className={current === item.id ? 'active' : ''}>
              {item.label}
            </a>
          ))}
        </nav>
      </header>
      {current === 'findings' ? <DashboardPage health={health} /> : <InfoPage />}
    </div>
  );
}
