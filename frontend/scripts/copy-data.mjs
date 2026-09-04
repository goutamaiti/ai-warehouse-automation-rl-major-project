/**
 * Copy real experiment output into the static assets of the dashboard.
 *
 * The dashboard must never contain hand-written numbers: every episode it
 * replays and every metric it shows comes from ../data, which is written by
 * baselines/evaluate_baselines.py and rl_agent/evaluate.py. This script runs
 * automatically before `npm run dev` and `npm run build`.
 */
import { mkdir, readdir, readFile, writeFile, rm } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const sourceDir = join(repoRoot, 'data');
const targetDir = resolve(here, '..', 'public', 'data');

async function readJsonFiles(directory) {
  let names = [];
  try {
    names = await readdir(directory);
  } catch {
    return [];
  }
  const files = [];
  for (const name of names.filter((n) => n.endsWith('.json')).sort()) {
    try {
      const text = await readFile(join(directory, name), 'utf8');
      files.push({ id: name.replace(/\.json$/, ''), text, data: JSON.parse(text) });
    } catch (error) {
      console.warn(`[copy-data] skipping ${name}: ${error.message}`);
    }
  }
  return files;
}

async function main() {
  await rm(targetDir, { recursive: true, force: true });
  await mkdir(join(targetDir, 'episodes'), { recursive: true });
  await mkdir(join(targetDir, 'results'), { recursive: true });

  const episodes = await readJsonFiles(join(sourceDir, 'episodes'));
  const results = await readJsonFiles(join(sourceDir, 'results'));

  for (const episode of episodes) {
    await writeFile(join(targetDir, 'episodes', `${episode.id}.json`), episode.text);
  }
  for (const result of results) {
    await writeFile(join(targetDir, 'results', `${result.id}.json`), result.text);
  }

  const index = {
    generatedAt: new Date().toISOString(),
    episodes: episodes.map(({ id, data }) => ({
      id,
      scenario: data.scenario ?? 'unknown',
      controller: data.controller ?? 'unknown',
      seed: data.seed ?? 0,
      frames: Array.isArray(data.frames) ? data.frames.length : 0,
      summary: data.summary ?? {},
    })),
    results: results.map(({ id, data }) => ({
      id,
      scenario: data.scenario ?? 'unknown',
      generatedAt: data.generated_at ?? null,
      summaries: data.summaries ?? [],
    })),
  };
  await writeFile(join(targetDir, 'index.json'), JSON.stringify(index, null, 1));

  console.log(
    `[copy-data] ${episodes.length} episode replay(s), ${results.length} result file(s) -> frontend/public/data`
  );
  if (episodes.length === 0) {
    console.warn(
      '[copy-data] no replays found. Generate some with:\n' +
        '            python -m baselines.evaluate_baselines --scenario default --record 1'
    );
  }
}

main().catch((error) => {
  console.error('[copy-data] failed:', error);
  process.exit(1);
});
