// `npm run dev` — the development launcher.
//
// It exists so the dev path obeys GUARDIAN_PLAN.md section 11 as well as the
// packaged one does. Vite used to sit on a hardcoded 5173 with `strictPort`,
// which meant a second checkout, a stale node, or any other project on 5173
// simply broke the launch. Now both ports are picked, and each child is told
// which one it got:
//
//   vite       --port <picked> --strictPort   and the proxy target by env
//   electron   BUZZEDIT_DEV_URL, BUZZEDIT_PORT
//
// Nothing is killed to make room; whatever holds a port keeps it.

import { spawn } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { pickPort } from './buzzcaf-ports.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const VITE_PREFERRED = Number.parseInt(process.env.BUZZEDIT_VITE_PORT || '', 10) || 5173;
const BACKEND_PREFERRED = Number.parseInt(process.env.BUZZEDIT_PORT || '', 10) || 8099;
const SPAN = 20;

const vitePort = await pickPort(VITE_PREFERRED, SPAN);
// The backend is spawned by electron/main.cjs, which picks its own port. All we
// do here is choose the *preferred* one it should start from, so that Vite's
// proxy and Electron agree. main.cjs still steps forward if this one goes.
const backendPort = await pickPort(BACKEND_PREFERRED, SPAN);
const devUrl = `http://127.0.0.1:${vitePort}`;

console.log(`BuzzEdit dev: vite ${vitePort}, backend ${backendPort}.`);

const env = {
  ...process.env,
  NODE_ENV: 'development',
  BUZZEDIT_VITE_PORT: String(vitePort),
  BUZZEDIT_PORT: String(backendPort),
  BUZZEDIT_DEV_URL: devUrl,
};

const children = [];
function run(command, args, name) {
  const child = spawn(command, args, { cwd: root, env, stdio: 'inherit', shell: true });
  child.on('exit', (code) => {
    console.log(`${name} exited with ${code}`);
    stopAll();
    process.exit(code ?? 0);
  });
  children.push(child);
  return child;
}

function stopAll() {
  for (const child of children) {
    if (child.exitCode === null && child.pid) {
      try { child.kill(); } catch { /* already gone */ }
    }
  }
}
process.on('SIGINT', () => { stopAll(); process.exit(0); });
process.on('SIGTERM', () => { stopAll(); process.exit(0); });

run('npx', ['vite', '--port', String(vitePort), '--strictPort'], 'vite');
run('npx', ['wait-on', devUrl, '&&', 'npx', 'electron', '.'], 'electron');
