// Unit tests for the shared Node port helper (GUARDIAN_PLAN.md section 11).
//
//   node --test scripts/
//
// Every test points at a throwaway ledger, so running these never touches the
// real one at %LOCALAPPDATA%\Buzzcaf\ports.json.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import net from 'node:net';
import http from 'node:http';
import path from 'node:path';

import {
  ledgerPath, pickPort, publish, withdraw, entries, entry,
  discover, identifies, baseOf, forget,
} from './buzzcaf-ports.mjs';

function tempLedger() {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'buzzcaf-ports-')), 'ports.json');
}

/** A listener that holds a port and answers nothing. */
function occupy(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer((socket) => {
      // Read and discard. A socket with no reader stays paused, never sees the
      // peer's FIN, and so never closes — which leaves `server.close()` in the
      // teardown waiting forever.
      socket.resume();
      socket.on('error', () => { /* the client hung up; that is the point */ });
    });
    server.once('error', reject);
    server.listen({ port, host: '127.0.0.1', exclusive: true }, () => resolve(server));
  });
}

/** A health endpoint that answers with the given identity payload. */
function healthServer(payload, { status = 200, healthPath = '/health' } = {}) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.url.split('?')[0] !== healthPath) { res.writeHead(404).end('no'); return; }
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(typeof payload === 'string' ? payload : JSON.stringify(payload));
    });
    server.listen({ port: 0, host: '127.0.0.1' }, () => resolve(server));
  });
}

const close = (s) => new Promise((r) => s.close(() => r()));

// ───────────────────────────── ledgerPath() ─────────────────────────────

test('ledgerPath honours BUZZCAF_PORTS_FILE', () => {
  const before = process.env.BUZZCAF_PORTS_FILE;
  try {
    process.env.BUZZCAF_PORTS_FILE = path.join(os.tmpdir(), 'x', 'ports.json');
    assert.equal(ledgerPath(), path.resolve(process.env.BUZZCAF_PORTS_FILE));
  } finally {
    if (before === undefined) delete process.env.BUZZCAF_PORTS_FILE;
    else process.env.BUZZCAF_PORTS_FILE = before;
  }
});

test('ledgerPath defaults to Buzzcaf/ports.json', () => {
  const before = process.env.BUZZCAF_PORTS_FILE;
  delete process.env.BUZZCAF_PORTS_FILE;
  try {
    const p = ledgerPath();
    assert.equal(path.basename(p), 'ports.json');
    assert.equal(path.basename(path.dirname(p)), 'Buzzcaf');
  } finally {
    if (before !== undefined) process.env.BUZZCAF_PORTS_FILE = before;
  }
});

// ────────────────────────────── pickPort() ──────────────────────────────

test('pickPort returns the preferred port when it is free', async () => {
  const free = await pickPort(0);                 // ask the OS for a spare number
  assert.equal(await pickPort(free, 5), free);
});

test('pickPort steps forward over a busy port, and never evicts it', async () => {
  const base = await pickPort(0);
  const holder = await occupy(base);
  try {
    const chosen = await pickPort(base, 5);
    assert.notEqual(chosen, base);
    assert.ok(chosen > base && chosen <= base + 5, `expected base+1..+5, got ${chosen}`);
    assert.ok(holder.listening, 'the squatter is still listening');
  } finally {
    await close(holder);
  }
});

test('pickPort steps over a whole busy run', async () => {
  const base = await pickPort(0);
  const holders = [];
  try {
    for (let p = base; p <= base + 2; p++) {
      try { holders.push(await occupy(p)); } catch { /* someone else has it */ }
    }
    const chosen = await pickPort(base, 20);
    assert.ok(!holders.some((h) => h.address().port === chosen));
  } finally {
    await Promise.all(holders.map(close));
  }
});

test('pickPort falls back to an OS-assigned port when the whole span is busy', async () => {
  const base = await pickPort(0);
  const holders = [];
  try {
    for (let p = base; p <= base + 2; p++) {
      try { holders.push(await occupy(p)); } catch { /* ignore */ }
    }
    // span 2 means base..base+2 — all held, so this falls through to bind(0).
    const chosen = await pickPort(base, 2);
    assert.ok(Number.isInteger(chosen) && chosen > 0);
    assert.ok(!holders.some((h) => h.address().port === chosen));
  } finally {
    await Promise.all(holders.map(close));
  }
});

test('pickPort never throws on nonsense input — a launcher must still start', async () => {
  for (const bad of ['nope', null, undefined, -1, 70000, NaN]) {
    const chosen = await pickPort(bad);
    assert.ok(Number.isInteger(chosen) && chosen > 0, `pickPort(${String(bad)}) -> ${chosen}`);
  }
});

// ───────────────────────── publish / withdraw ─────────────────────────

test('publish writes the entry shape the plan documents', async () => {
  const ledger = tempLedger();
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', { comfyui: 8188 }, ledger);
  const data = JSON.parse(fs.readFileSync(ledger, 'utf8'));
  assert.deepEqual(Object.keys(data), ['buzzedit']);
  const e = data.buzzedit;
  assert.equal(e.port, 8099);
  assert.equal(e.pid, process.pid);
  assert.equal(e.health, 'http://127.0.0.1:8099/api/health');
  assert.deepEqual(e.extra, { comfyui: 8188 });
  assert.ok(!Number.isNaN(Date.parse(e.started_at)), `started_at parses: ${e.started_at}`);
});

test('publish merges rather than replaces, so one app never loses another', async () => {
  const ledger = tempLedger();
  await publish('dexter', 8098, 'http://127.0.0.1:8098/health', null, ledger);
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', null, ledger);
  await publish('buzzcode', 8089, 'http://127.0.0.1:8089/health', null, ledger);
  assert.deepEqual(Object.keys(entries(ledger)).sort(), ['buzzcode', 'buzzedit', 'dexter']);
  assert.equal(entry('dexter', ledger).port, 8098);
});

test('a re-publish replaces only its own entry', async () => {
  const ledger = tempLedger();
  await publish('dexter', 8098, 'http://127.0.0.1:8098/health', null, ledger);
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', null, ledger);
  await publish('buzzedit', 8100, 'http://127.0.0.1:8100/api/health', null, ledger);
  assert.equal(entry('buzzedit', ledger).port, 8100);
  assert.equal(entry('dexter', ledger).port, 8098);
});

test('withdraw removes only its own entry, and tolerates a missing one', async () => {
  const ledger = tempLedger();
  await publish('dexter', 8098, 'http://127.0.0.1:8098/health', null, ledger);
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', null, ledger);
  await withdraw('buzzedit', ledger);
  await withdraw('buzzedit', ledger);           // idempotent
  await withdraw('never-published', ledger);
  assert.deepEqual(Object.keys(entries(ledger)), ['dexter']);
  assert.equal(entry('buzzedit', ledger), null);
});

test('a corrupt ledger is rebuilt, not fatal', async () => {
  const ledger = tempLedger();
  fs.mkdirSync(path.dirname(ledger), { recursive: true });
  fs.writeFileSync(ledger, '{ this is not json');
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', null, ledger);
  assert.equal(entry('buzzedit', ledger).port, 8099);
});

test('a stale lock is broken after five seconds, and released afterwards', async () => {
  const ledger = tempLedger();
  fs.mkdirSync(path.dirname(ledger), { recursive: true });
  const lock = `${ledger}.lock`;
  fs.writeFileSync(lock, '999999');
  const old = (Date.now() - 60_000) / 1000;
  fs.utimesSync(lock, old, old);
  await publish('buzzedit', 8099, 'http://127.0.0.1:8099/api/health', null, ledger);
  assert.equal(entry('buzzedit', ledger).port, 8099);
  assert.equal(fs.existsSync(lock), false);
});

test('concurrent publishes all survive', async () => {
  const ledger = tempLedger();
  const apps = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  await Promise.all(apps.map((a, i) =>
    publish(a, 9000 + i, `http://127.0.0.1:${9000 + i}/health`, null, ledger)));
  assert.deepEqual(Object.keys(entries(ledger)).sort(), apps);
});

test('entries and entry on a ledger that does not exist', () => {
  const ledger = path.join(os.tmpdir(), 'buzzcaf-nope', 'ports.json');
  assert.deepEqual(entries(ledger), {});
  assert.equal(entry('buzzedit', ledger), null);
});

// ─────────────────────────────── baseOf() ───────────────────────────────

test('baseOf strips the path and rewrites localhost to 127.0.0.1', () => {
  assert.equal(baseOf('http://127.0.0.1:8098/health'), 'http://127.0.0.1:8098');
  assert.equal(baseOf('http://localhost:8099/api/health'), 'http://127.0.0.1:8099');
  assert.equal(baseOf('http://127.0.0.1:8099/'), 'http://127.0.0.1:8099');
});

// ───────────────────────────── identifies() ─────────────────────────────

test('identifies accepts only the right app', async () => {
  const server = await healthServer({ app: 'buzzedit', status: 'ok' });
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    assert.equal(await identifies(base, '/health', 'buzzedit'), true);
    assert.equal(await identifies(base, '/health', 'buzzcaf'), false, 'a foreign app is not a find');
  } finally {
    await close(server);
  }
});

test('identifies rejects a non-JSON body, a 500, and a dead port', async () => {
  const plain = await healthServer('hello, i am not json');
  const broken = await healthServer({ app: 'buzzedit' }, { status: 500 });
  const dead = await pickPort(0);
  try {
    assert.equal(await identifies(`http://127.0.0.1:${plain.address().port}`, '/health', 'buzzedit'), false);
    assert.equal(await identifies(`http://127.0.0.1:${broken.address().port}`, '/health', 'buzzedit'), false);
    assert.equal(await identifies(`http://127.0.0.1:${dead}`, '/health', 'buzzedit'), false);
  } finally {
    await close(plain);
    await close(broken);
  }
});

test('identifies gives up on a listener that never answers', async () => {
  const silent = await occupy(await pickPort(0));
  try {
    const t0 = Date.now();
    assert.equal(await identifies(`http://127.0.0.1:${silent.address().port}`, '/health', 'buzzedit'), false);
    assert.ok(Date.now() - t0 < 5000, 'the health timeout is honoured');
  } finally {
    await close(silent);
  }
});

// ────────────────────────────── discover() ──────────────────────────────

test('discover prefers an explicit env URL', async () => {
  forget();
  const url = await discover('buzzedit', 8099, '/api/health', 2, 'http://localhost:9999/', 0, tempLedger());
  assert.equal(url, 'http://127.0.0.1:9999', 'trailing slash trimmed, localhost rewritten');
});

test('discover finds the app through the ledger', async () => {
  forget();
  const ledger = tempLedger();
  const server = await healthServer({ app: 'buzzedit', status: 'ok' });
  const port = server.address().port;
  try {
    await publish('buzzedit', port, `http://127.0.0.1:${port}/health`, null, ledger);
    // span 0 and a preferred that cannot match: only the ledger can answer.
    assert.equal(await discover('buzzedit', 1, '/health', 0, null, 0, ledger),
      `http://127.0.0.1:${port}`);
  } finally {
    await close(server);
  }
});

test('discover ignores a ledger entry whose process is gone', async () => {
  forget();
  const ledger = tempLedger();
  fs.mkdirSync(path.dirname(ledger), { recursive: true });
  fs.writeFileSync(ledger, JSON.stringify({
    buzzedit: {
      port: 8099, pid: 999999, started_at: new Date().toISOString(),
      health: 'http://127.0.0.1:8099/api/health', extra: {},
    },
  }));
  assert.equal(await discover('buzzedit', 0, '/api/health', 0, null, 0, ledger), null);
});

test('discover ignores a ledger entry that answers as a different app', async () => {
  forget();
  const ledger = tempLedger();
  const impostor = await healthServer({ app: 'caliberai', status: 'ok' });
  const port = impostor.address().port;
  try {
    // A live pid (ours) but the wrong identity: rule 4 says keep looking.
    await publish('buzzedit', port, `http://127.0.0.1:${port}/health`, null, ledger);
    assert.equal(await discover('buzzedit', 1, '/health', 0, null, 0, ledger), null);
  } finally {
    await close(impostor);
  }
});

test('discover falls back to a scan with the identity check', async () => {
  forget();
  const ledger = tempLedger();
  const impostor = await healthServer({ app: 'somebody-else' });
  const server = await healthServer({ app: 'buzzedit', status: 'ok' });
  const port = server.address().port;
  try {
    // Nothing published: the scan walks the span, skips whoever answers wrong,
    // and stops at the one that names itself.
    const found = await discover('buzzedit', Math.max(1, port - 5), '/health', 10, null, 0, ledger);
    assert.equal(found, `http://127.0.0.1:${port}`);
    assert.notEqual(found, `http://127.0.0.1:${impostor.address().port}`);
  } finally {
    await close(server);
    await close(impostor);
  }
});

test('discover answers null when nothing is there', async () => {
  forget();
  const base = await pickPort(0);
  assert.equal(await discover('buzzedit', base, '/health', 2, null, 0, tempLedger()), null);
});

test('discover caches within the ttl and forgets on demand', async () => {
  forget();
  const ledger = tempLedger();
  const server = await healthServer({ app: 'buzzedit', status: 'ok' });
  const port = server.address().port;
  const first = await discover('buzzedit', port, '/health', 0, null, 30, ledger);
  assert.equal(first, `http://127.0.0.1:${port}`);
  await close(server);
  // The server is gone but the cache is not yet cold.
  assert.equal(await discover('buzzedit', port, '/health', 0, null, 30, ledger), first);
  forget('buzzedit');
  assert.equal(await discover('buzzedit', port, '/health', 0, null, 0, ledger), null);
});

test('publish clears a stale cached answer', async () => {
  forget();
  const ledger = tempLedger();
  const dead = await pickPort(0);
  assert.equal(await discover('buzzedit', dead, '/health', 0, null, 30, ledger), null);
  const server = await healthServer({ app: 'buzzedit', status: 'ok' });
  const port = server.address().port;
  try {
    await publish('buzzedit', port, `http://127.0.0.1:${port}/health`, null, ledger);
    assert.equal(await discover('buzzedit', 1, '/health', 0, null, 0, ledger),
      `http://127.0.0.1:${port}`, 'the cached "offline" was dropped by publish');
  } finally {
    await close(server);
  }
});
