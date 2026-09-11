import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// GUARDIAN_PLAN.md section 11 rule 6: a dev front-end is *told* the backend
// port at launch and proxies to it. `scripts/dev.mjs` picks both ports and puts
// them in the environment; the numbers below are only the preferred defaults
// for a bare `npx vite`, and are not assumed to be free.
const backendPort = Number.parseInt(process.env.BUZZEDIT_PORT || '', 10) || 8099;
const vitePort = Number.parseInt(process.env.BUZZEDIT_VITE_PORT || '', 10) || 5173;
const backend = process.env.BUZZEDIT_URL || `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  base: './',
  server: {
    // strictPort so the dev server never silently drifts off the port Electron
    // was told to open; dev.mjs has already stepped forward if it had to.
    port: vitePort,
    strictPort: true,
    // With these in place the renderer's API base can stay relative in dev —
    // same origin, proxied — instead of carrying a port of its own.
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/output': { target: backend, changeOrigin: true },
    },
  },
});
