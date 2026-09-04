import { useEffect, useState } from 'react';

import MetricsPanel from './MetricsPanel';
import PlaybackControls from './PlaybackControls';
import ResultsTable from './ResultsTable';
import RewardPanel from './RewardPanel';
import WarehouseCanvas from './WarehouseCanvas';
import {
  backendConfigured,
  loadEpisode,
  loadIndex,
  loadResult,
  loadScenarios,
  runLiveEpisode,
} from '../lib/api';
import { controllerLabel, titleCase } from '../lib/format';
import { useEpisodePlayer } from '../lib/useEpisodePlayer';

/**
 * Which episode to open on: the learned policy on the baseline scenario, then
 * any learned-policy episode, then whatever exists. Every episode - including
 * the ones where a controller fails - stays available in the picker.
 */
function pickDefaultEpisode(episodes) {
  return (
    episodes.find((e) => e.controller === 'ppo' && e.scenario === 'default') ??
    episodes.find((e) => e.controller === 'ppo') ??
    episodes[0] ??
    null
  );
}

export default function DashboardPage({ health }) {
  const [index, setIndex] = useState({ episodes: [], results: [] });
  const [results, setResults] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [episode, setEpisode] = useState(null);
  const [error, setError] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [liveForm, setLiveForm] = useState({ scenario: 'default', controller: 'astar', seed: 0 });
  const [running, setRunning] = useState(false);

  const player = useEpisodePlayer(episode);

  useEffect(() => {
    let cancelled = false;
    loadIndex()
      .then(async (data) => {
        if (cancelled) return;
        setIndex(data);
        const first = pickDefaultEpisode(data.episodes ?? []);
        if (first) setSelectedId(first.id);
        const loaded = await Promise.all(
          (data.results ?? []).map(async (entry) => {
            const full = await loadResult(entry.id).catch(() => null);
            return full ? { ...entry, summaries: full.summaries ?? [] } : entry;
          })
        );
        if (!cancelled) setResults(loaded);
      })
      .catch((err) => !cancelled && setError(`Could not load bundled data: ${err.message}`));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!health) return;
    let cancelled = false;
    loadScenarios()
      .then((available) => !cancelled && setScenarios(available))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [health]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    loadEpisode(selectedId)
      .then((data) => {
        if (cancelled) return;
        setEpisode(data);
        setError(null);
      })
      .catch((err) => !cancelled && setError(`Could not load episode: ${err.message}`));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const runLive = async (event) => {
    event.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const recording = await runLiveEpisode(liveForm);
      setEpisode(recording);
      setSelectedId(null);
    } catch (err) {
      setError(`Live run failed: ${err.message}`);
    } finally {
      setRunning(false);
    }
  };

  const meta = {
    controller: episode?.controller ?? 'unknown',
    scenario: episode?.scenario ?? 'unknown',
    seed: episode?.seed ?? 0,
  };
  const availableControllers = (health?.controllers ?? []).filter((c) => c.available);

  return (
    <>
      {error && <p className="error">{error}</p>}

      <div className="layout">
        <main className="stage">
          <div className="stage-header">
            <label>
              Episode
              <select value={selectedId ?? ''} onChange={(event) => setSelectedId(event.target.value)}>
                {selectedId === null && <option value="">live run</option>}
                {(index.episodes ?? []).map((item) => (
                  <option key={item.id} value={item.id}>
                    {titleCase(item.scenario)} · {controllerLabel(item.controller)} · seed {item.seed}
                  </option>
                ))}
              </select>
            </label>
            {episode?.summary && (
              <span className="muted small">
                {episode.summary.steps} steps · {episode.summary.moves} moves ·{' '}
                {episode.summary.collisions} collisions
              </span>
            )}
          </div>

          {episode ? (
            <>
              <WarehouseCanvas layout={episode.layout} frame={player.frame} trail={player.trail} />
              <PlaybackControls
                playing={player.playing}
                onTogglePlay={player.togglePlay}
                frameIndex={player.frameIndex}
                totalFrames={player.totalFrames}
                onSeek={player.seek}
                speed={player.speed}
                onSpeedChange={player.setSpeed}
              />
            </>
          ) : (
            <p className="muted">Loading episode…</p>
          )}

          {health ? (
            <form className="live-run" onSubmit={runLive}>
              <h2>Run a new episode</h2>
              <div className="live-controls">
                <label>
                  Scenario
                  <select
                    value={liveForm.scenario}
                    onChange={(e) => setLiveForm({ ...liveForm, scenario: e.target.value })}
                  >
                    {(scenarios.length ? scenarios.map((s) => s.name) : ['default']).map((name) => (
                      <option key={name} value={name}>
                        {titleCase(name)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Controller
                  <select
                    value={liveForm.controller}
                    onChange={(e) => setLiveForm({ ...liveForm, controller: e.target.value })}
                  >
                    {availableControllers.map((controller) => (
                      <option key={controller.name} value={controller.name}>
                        {controllerLabel(controller.name)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Seed
                  <input
                    type="number"
                    min="0"
                    value={liveForm.seed}
                    onChange={(e) => setLiveForm({ ...liveForm, seed: Number(e.target.value) })}
                  />
                </label>
                <button type="submit" className="primary" disabled={running}>
                  {running ? 'Running…' : 'Run episode'}
                </button>
              </div>
            </form>
          ) : (
            backendConfigured && (
              <p className="muted small">
                Backend configured but unreachable — showing recorded episodes only.
              </p>
            )
          )}
        </main>

        <MetricsPanel
          meta={meta}
          frame={player.frame}
          summary={episode?.summary}
          frameIndex={player.frameIndex}
          totalFrames={player.totalFrames}
        />
      </div>

      <RewardPanel
        rewardConfig={episode?.reward_config}
        components={player.frame?.reward_components}
        reward={player.frame?.reward}
        cumulative={player.cumulativeReward}
        stepIndex={player.frameIndex}
      />

      <ResultsTable results={results} />

      <footer className="app-footer">
        <p>
          Every animation and number on this page is read from files produced by an actual
          simulation run (<code>data/episodes</code>, <code>data/results</code>). Nothing here is
          mocked or hand-written.
        </p>
      </footer>
    </>
  );
}
