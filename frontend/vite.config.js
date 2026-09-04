import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The dashboard is a static site: it replays episode recordings copied out of
// ../data by scripts/copy-data.mjs. When VITE_API_BASE is set at build time it
// additionally talks to the FastAPI backend for live runs.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: { outDir: 'dist', sourcemap: false },
});
