import { useEffect, useState } from 'react';
import WarehouseEditor, { blankGrid } from './WarehouseEditor';
import WarehouseCanvas from './WarehouseCanvas';
import PlaybackControls from './PlaybackControls';
import MetricsPanel from './MetricsPanel';
import RewardPanel from './RewardPanel';
import { validateGrid } from '../lib/grid';
import { RULESETS, DEFAULT_RULESET } from '../lib/rulesets';
import { OFFLINE_CONTROLLERS, runOfflineEpisode } from '../lib/offlineSimulation';
import { runLiveEpisode, loadEpisode } from '../lib/api';
import { useEpisodePlayer } from '../lib/useEpisodePlayer';
import { controllerLabel, titleCase } from '../lib/format';

const RULESET_OPTIONS = [
  { name: 'default', label: RULESETS.default.label, backendOnly: false },
  { name: 'dynamic_obstacles', label: 'Dynamic obstacles', backendOnly: true },
  { name: 'battery_constrained', label: RULESETS.battery_constrained.label, backendOnly: false },
];

const CONTROLLER_OPTIONS = [
  { name: 'astar', backendOnly: false },
  { name: 'bfs', backendOnly: false },
  { name: 'random', backendOnly: false },
  { name: 'ppo', backendOnly: true },
];

const TEMPLATES = [
  { id: 'default_astar_seed1000', label: 'Default warehouse' },
  { id: 'battery_constrained_astar_seed1000', label: 'Battery-constrained warehouse' },
  { id: 'dynamic_obstacles_astar_seed1000', label: 'Dynamic-obstacles warehouse (layout only)' },
  { id: 'complex_static_astar_seed1000', label: 'Complex static warehouse' },
  { id: 'simple_static_astar_seed1000', label: 'Simple static warehouse' },
];

export default function EditorPage({ backendHealth }) {
  const [grid, setGrid] = useState(() => blankGrid(13, 19));
  const [templateLoaded, setTemplateLoaded] = useState(false);
  const [rulesetName, setRulesetName] = useState(DEFAULT_RULESET);
  const [controller, setController] = useState('astar');
  const [seed, setSeed] = useState(0);
  const [replay, setReplay] = useState(null);
  const [source, setSource] = useState(null); // 'offline' | 'backend'
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const player = useEpisodePlayer(replay);
  const validation = validateGrid(grid);
  const backendReachable = Boolean(backendHealth);
  const ppoAvailable = (backendHealth?.controllers ?? []).some(
    (c) => c.name === 'ppo' && c.available
  );

  // Start from a real warehouse layout instead of a blank grid, so there is
  // something meaningful to edit and run immediately.
  useEffect(() => {
    if (templateLoaded) return;
    setTemplateLoaded(true);
    loadEpisode('default_astar_seed1000')
      .then((data) => setGrid(data.layout.grid))
      .catch(() => {
        /* keep the blank starter grid if the bundled template is unavailable */
      });
  }, [templateLoaded]);

  useEffect(() => {
    if (rulesetName === 'dynamic_obstacles' && !backendReachable) setRulesetName(DEFAULT_RULESET);
    if (controller === 'ppo' && !ppoAvailable) setController('astar');
  }, [backendReachable, ppoAvailable, rulesetName, controller]);

  async function loadTemplate(templateId) {
    setError(null);
    try {
      const data = await loadEpisode(templateId);
      setGrid(data.layout.grid);
    } catch (err) {
      setError(`Could not load template: ${err.message}`);
    }
  }

  async function run() {
    if (!validation.valid) return;
    setRunning(true);
    setError(null);
    try {
      if (backendReachable) {
        const data = await runLiveEpisode({ scenario: rulesetName, controller, seed, layout: grid });
        setReplay(data);
        setSource('backend');
      } else {
        const data = runOfflineEpisode({ grid, rulesetName, controllerName: controller, seed });
        setReplay(data);
        setSource('offline');
      }
    } catch (err) {
      setError(`Run failed: ${err.message}`);
    } finally {
      setRunning(false);
    }
  }

  const meta = {
    controller: replay?.controller ?? controller,
    scenario: replay?.scenario ?? rulesetName,
    seed: replay?.seed ?? seed,
  };

  return (
    <div className="editor-page">
      <section className="panel">
        <h2>Draw a warehouse</h2>
        <div className="template-row">
          <span className="muted small">Start from:</span>
          {TEMPLATES.map((t) => (
            <button key={t.id} type="button" onClick={() => loadTemplate(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <WarehouseEditor grid={grid} onChange={setGrid} />
      </section>

      <section className="panel run-panel">
        <h2>Run it</h2>
        <div className="live-controls">
          <label>
            Ruleset
            <select value={rulesetName} onChange={(e) => setRulesetName(e.target.value)}>
              {RULESET_OPTIONS.filter((r) => !r.backendOnly || backendReachable).map((r) => (
                <option key={r.name} value={r.name}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Controller
            <select value={controller} onChange={(e) => setController(e.target.value)}>
              {CONTROLLER_OPTIONS.filter((c) => c.name !== 'ppo' || ppoAvailable).map((c) => (
                <option key={c.name} value={c.name}>
                  {controllerLabel(c.name)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Seed
            <input type="number" min="0" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <button type="button" className="primary" onClick={run} disabled={!validation.valid || running}>
            {running ? 'Running…' : 'Run episode'}
          </button>
        </div>
        {!validation.valid && <p className="validation-status fail">{validation.error}</p>}
        <p className="muted small">
          {backendReachable
            ? 'Running through the connected backend — real physics and reward, same code as every reported result.'
            : 'No backend connected: running A*, BFS or Random in an in-browser sandbox. PPO and moving obstacles need a backend — see the Info page.'}
        </p>
        {error && <p className="error">{error}</p>}
      </section>

      {replay && (
        <>
          <section className="panel stage">
            <div className="stage-header">
              <h2>
                {titleCase(meta.scenario)} · {controllerLabel(meta.controller)} · seed {meta.seed}
              </h2>
              <span className={`badge ${source === 'backend' ? 'ok' : 'neutral'}`}>
                {source === 'backend' ? 'Backend run' : 'Offline sandbox'}
              </span>
            </div>
            <WarehouseCanvas layout={replay.layout} frame={player.frame} trail={player.trail} />
            <PlaybackControls
              playing={player.playing}
              onTogglePlay={player.togglePlay}
              frameIndex={player.frameIndex}
              totalFrames={player.totalFrames}
              onSeek={player.seek}
              speed={player.speed}
              onSpeedChange={player.setSpeed}
            />
          </section>

          <div className="layout editor-results">
            <RewardPanel
              rewardConfig={replay.reward_config}
              components={player.frame?.reward_components}
              reward={player.frame?.reward}
              cumulative={player.cumulativeReward}
              stepIndex={player.frameIndex}
            />
            <MetricsPanel
              meta={meta}
              frame={player.frame}
              summary={replay.summary}
              frameIndex={player.frameIndex}
              totalFrames={player.totalFrames}
            />
          </div>
        </>
      )}
    </div>
  );
}
