// Unit tests for BuzzEdit's launcher decisions (electron/port-plan.cjs).
//
//   node --test scripts/port-plan.test.mjs
//
// These cover the part of the Electron launcher that decides *where* the
// backend and ComfyUI go — without launching Electron, and therefore without
// launching ComfyUI and the GPU with it. Every test runs against a throwaway
// ledger and fake servers on OS-assigned ports.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import net from 'node:net';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { pickPort, publish, forget } from './buzzcaf-ports.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const require_ = createRequire(import.meta.url);
const plan = require_(path.join(here, '..', 'electron', 'port-plan.cjs'));

function tempLedger() {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'buzzedit-plan-')), 'ports.json');
}

function jsonServer(routes) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const body = routes[req.url.split('?')[0]];
      if (body === undefined) { res.writeHead(404).end('no'); return; }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    });
    server.listen({ port: 0, host: '127.0.0.1' }, () => resolve(server));
  });
}

function occupy(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer((socket) => {
      socket.resume();
      socket.on('error', () => { /* expected */ });
    });
    server.once('error', reject);
    server.listen({ port, host: '127.0.0.1', exclusive: true }, () => resolve(server));
  });
}

const close = (s) => new Promise((r) => s.close(() => r()));

/** Run `fn` with these environment variables set, then put the env back. */
async function withEnv(vars, fn) {
  const before = {};
  for (const [k, v] of Object.entries(vars)) {
    before[k] = process.env[k];
    if (v === null) delete process.env[k];
    else process.env[k] = String(v);
  }
  try {
    return await fn();
  } finally {
    for (const [k, v] of Object.entries(before)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
}

// ─────────────────────────── preferred ports ───────────────────────────

test('the preferred ports are today\'s numbers, and the env overrides them', async () => {
  await withEnv({ BUZZEDIT_PORT: null, COMFYUI_PORT: null, BUZZEDIT_VITE_PORT: null }, () => {
    assert.equal(plan.preferredBackendPort(), 8099);
    assert.equal(plan.preferredComfyPort(), 8188);
    assert.equal(plan.preferredVitePort(), 5173);
  });
  await withEnv({ BUZZEDIT_PORT: 9001, COMFYUI_PORT: 9002, BUZZEDIT_VITE_PORT: 9003 }, () => {
    assert.equal(plan.preferredBackendPort(), 9001);
    assert.equal(plan.preferredComfyPort(), 9002);
    assert.equal(plan.preferredVitePort(), 9003);
  });
  // Nonsense falls back rather than producing NaN in a URL.
  await withEnv({ BUZZEDIT_PORT: 'later' }, () => {
    assert.equal(plan.preferredBackendPort(), 8099);
  });
});

// ───────────────────────────── the backend ─────────────────────────────

test('resolveBackend picks the preferred port when nothing is there', async () => {
  forget();
  const free = await pickPort(0);
  await withEnv({ BUZZEDIT_PORT: free, BUZZEDIT_URL: null, BUZZCAF_PORTS_FILE: tempLedger() }, async () => {
    const got = await plan.resolveBackend();
    assert.deepEqual({ port: got.port, adopt: got.adopt }, { port: free, adopt: false });
  });
});

test('resolveBackend steps forward instead of evicting a squatter', async () => {
  forget();
  const base = await pickPort(0);
  const squatter = await occupy(base);
  try {
    await withEnv({ BUZZEDIT_PORT: base, BUZZEDIT_URL: null, BUZZCAF_PORTS_FILE: tempLedger() }, async () => {
      const got = await plan.resolveBackend();
      assert.equal(got.adopt, false, 'a stranger on the port is not our backend');
      assert.notEqual(got.port, base);
      assert.ok(got.port > base && got.port <= base + plan.PORT_SPAN);
    });
    assert.ok(squatter.listening, 'the squatter was left alone');
  } finally {
    await close(squatter);
  }
});

test('resolveBackend adopts a backend already published in the ledger', async () => {
  forget();
  const ledger = tempLedger();
  const server = await jsonServer({ '/api/health': { app: 'buzzedit', status: 'ok' } });
  const port = server.address().port;
  try {
    await publish('buzzedit', port, `http://127.0.0.1:${port}/api/health`, { comfyui: 1 }, ledger);
    await withEnv({ BUZZEDIT_PORT: 1, BUZZEDIT_URL: null, BUZZCAF_PORTS_FILE: ledger }, async () => {
      const got = await plan.resolveBackend();
      assert.deepEqual({ port: got.port, adopt: got.adopt }, { port, adopt: true });
    });
  } finally {
    await close(server);
  }
});

test('resolveBackend adopts a backend found only by the identity scan', async () => {
  forget();
  const server = await jsonServer({ '/api/health': { app: 'buzzedit', status: 'ok' } });
  const port = server.address().port;
  try {
    await withEnv({
      BUZZEDIT_PORT: Math.max(1, port - 4), BUZZEDIT_URL: null, BUZZCAF_PORTS_FILE: tempLedger(),
    }, async () => {
      const got = await plan.resolveBackend();
      assert.deepEqual({ port: got.port, adopt: got.adopt }, { port, adopt: true });
    });
  } finally {
    await close(server);
  }
});

test('resolveBackend refuses to adopt a stranger that answers /api/health', async () => {
  forget();
  // This is the CaliberAI-mistaken-for-the-Studio bug, in miniature: a live
  // JSON health route on the port we wanted, belonging to something else.
  const impostor = await jsonServer({ '/api/health': { app: 'caliberai', status: 'ok' } });
  const port = impostor.address().port;
  try {
    await withEnv({ BUZZEDIT_PORT: port, BUZZEDIT_URL: null, BUZZCAF_PORTS_FILE: tempLedger() }, async () => {
      const got = await plan.resolveBackend();
      assert.equal(got.adopt, false);
      assert.notEqual(got.port, port, 'we step past it rather than take it over');
    });
  } finally {
    await close(impostor);
  }
});

test('resolveBackend honours BUZZEDIT_URL above everything', async () => {
  forget();
  await withEnv({
    BUZZEDIT_PORT: 8099, BUZZEDIT_URL: 'http://127.0.0.1:12345', BUZZCAF_PORTS_FILE: tempLedger(),
  }, async () => {
    const got = await plan.resolveBackend();
    assert.deepEqual({ port: got.port, adopt: got.adopt }, { port: 12345, adopt: true });
  });
});

// ────────────────────────────── ComfyUI ──────────────────────────────

test('resolveComfy picks a free port when no ComfyUI is running', async () => {
  const free = await pickPort(0);
  await withEnv({ COMFYUI_PORT: free, COMFYUI_URL: null }, async () => {
    const got = await plan.resolveComfy(async () => null);
    assert.deepEqual({ port: got.port, adopt: got.adopt }, { port: free, adopt: false });
  });
});

test('resolveComfy adopts a ComfyUI that answers /system_stats', async () => {
  const server = await jsonServer({ '/system_stats': { system: { os: 'nt' } } });
  const port = server.address().port;
  try {
    await withEnv({ COMFYUI_PORT: Math.max(1, port - 3), COMFYUI_URL: null }, async () => {
      const got = await plan.resolveComfy();
      assert.deepEqual({ port: got.port, adopt: got.adopt }, { port, adopt: true });
    });
  } finally {
    await close(server);
  }
});

test('resolveComfy does not mistake a stranger for ComfyUI', async () => {
  const stranger = await jsonServer({ '/system_stats': { hello: 'not comfy' } });
  const port = stranger.address().port;
  try {
    await withEnv({ COMFYUI_PORT: port, COMFYUI_URL: null }, async () => {
      const got = await plan.resolveComfy();
      assert.equal(got.adopt, false);
      assert.notEqual(got.port, port);
    });
  } finally {
    await close(stranger);
  }
});

test('resolveComfy takes COMFYUI_URL as given', async () => {
  await withEnv({ COMFYUI_PORT: 8188, COMFYUI_URL: 'http://127.0.0.1:8001' }, async () => {
    const got = await plan.resolveComfy(async () => { throw new Error('must not probe'); });
    assert.deepEqual({ port: got.port, adopt: got.adopt }, { port: 8001, adopt: true });
  });
});

test('resolveComfy steps past a squatter rather than killing it', async () => {
  const base = await pickPort(0);
  const squatter = await occupy(base);
  try {
    await withEnv({ COMFYUI_PORT: base, COMFYUI_URL: null }, async () => {
      const got = await plan.resolveComfy();
      assert.equal(got.adopt, false);
      assert.notEqual(got.port, base);
    });
    assert.ok(squatter.listening);
  } finally {
    await close(squatter);
  }
});

// ───────────────────── nothing hardcoded, still ─────────────────────

test('no literal port survives in the launcher, the preload or the renderer', () => {
  const root = path.resolve(here, '..');
  const files = [
    'electron/main.cjs',
    'electron/preload.cjs',
    'src/hooks/api.ts',
    'src/hooks/useAutoSave.ts',
    'src/components/PreviewPlayer.tsx',
    'src/components/AgentPanel.tsx',
    'src/components/Preferences.tsx',
  ];
  // The four numbers this app used to hardcode. Anything else four digits long
  // in these files is a window size or a timeout.
  const oldPorts = /\b(8099|8188|8000|5173)\b/;
  for (const rel of files) {
    const text = fs.readFileSync(path.join(root, rel), 'utf8');
    const offenders = text.split('\n')
      .map((line, i) => [i + 1, line])
      .filter(([, line]) => oldPorts.test(line) && !line.trimStart().startsWith('//')
        && !line.trimStart().startsWith('*'));
    assert.deepEqual(offenders, [], `${rel} still names a port`);
  }
  // The preferred defaults live in exactly one place each.
  const planText = fs.readFileSync(path.join(root, 'electron/port-plan.cjs'), 'utf8');
  for (const n of ['8099', '8188', '5173']) {
    const hits = planText.split(n).length - 1;
    assert.equal(hits, 1, `${n} should appear once in port-plan.cjs, found ${hits}`);
  }
});
