// Buzzcaf shared port helper — the Node implementation.
//
// GUARDIAN_PLAN.md section 11 is the contract: no port is hardcoded, a busy
// port is stepped over (never evicted), the chosen port is published to a
// shared ledger once the server answers its own health check, and every other
// app finds it from there.
//
// This file is the **canonical** Node copy. It lives at
// `BuzzEdit/scripts/buzzcaf-ports.mjs` and is copied byte-identical into every
// other Node/Tauri app (BuzzViolin today). Do not fork it: fix it here and
// re-copy. It depends on nothing outside `node:` builtins on purpose — a
// launcher has to work before `npm install` has ever run.
//
// It is the same helper as `dexter/backend/buzzcaf_ports.py`, function for
// function, with the same ledger, the same lock, the same discovery order:
//
//   ledgerPath()                                    -> string
//   pickPort(preferred, span, host)                 -> Promise<number>
//   publish(app, port, health, extra, ledger)       -> Promise<void>
//   withdraw(app, ledger)                           -> Promise<void>
//   entries(ledger) / entry(app, ledger)            -> the raw ledger
//   discover(app, preferred, healthPath, span, …)   -> Promise<string|null>

import net from 'node:net';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';

export const SPAN = 20;
export const DEFAULT_HOST = '127.0.0.1';

const LOCK_STALE_MS = 5000;
const LOCK_RETRY_MS = 20;
const LOCK_TIMEOUT_MS = 5000;
const HEALTH_TIMEOUT_MS = 1500;
const CONNECT_TIMEOUT_MS = 250;
const DEFAULT_TTL = 3;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Where the shared ledger lives. `BUZZCAF_PORTS_FILE` overrides it. */
export function ledgerPath() {
  const override = (process.env.BUZZCAF_PORTS_FILE || '').trim();
  if (override) return path.resolve(override);
  // %LOCALAPPDATA% on Windows; a POSIX-shaped fallback so tests and any future
  // non-Windows box still have somewhere sane to write.
  const base = process.env.LOCALAPPDATA
    || (process.platform === 'darwin'
      ? path.join(os.homedir(), 'Library', 'Application Support')
      : path.join(os.homedir(), '.local', 'share'));
  return path.join(base, 'Buzzcaf', 'ports.json');
}

function resolveLedger(ledger) {
  return ledger ? path.resolve(String(ledger)) : ledgerPath();
}

// ──────────────────────────── picking a port ────────────────────────────

/**
 * Whether `port` can actually be bound right now.
 *
 * A real bind, not a connect probe: a probe answers "free" for a port held by
 * a socket in TIME_WAIT or bound to another interface, and then the child
 * process fails with EADDRINUSE anyway. `exclusive` keeps Node from quietly
 * sharing the port with a cluster peer.
 */
function bindable(port, host = DEFAULT_HOST) {
  return new Promise((resolve) => {
    const server = net.createServer();
    let settled = false;
    const done = (ok) => {
      if (settled) return;
      settled = true;
      try { server.close(() => resolve(ok)); } catch { resolve(ok); }
    };
    server.once('error', () => done(false));
    server.once('listening', () => done(true));
    try {
      server.listen({ port, host, exclusive: true });
    } catch {
      done(false);
    }
  });
}

/** Bind port 0 and report what the OS handed out. */
function ephemeralPort(host = DEFAULT_HOST) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen({ port: 0, host, exclusive: true }, () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

/**
 * The preferred port if it is free, else the next free one, else port 0.
 *
 * Never evicts. Whatever holds the preferred port keeps it; we move.
 *
 * There is an unavoidable gap between this bind and the server's — the socket
 * has to be closed before uvicorn (or Node, or Rust) can take it. That is the
 * price of picking on behalf of another process, and the plan accepts it,
 * because the alternative — killing whoever holds the port — is exactly what
 * this section exists to abolish.
 */
export async function pickPort(preferred, span = SPAN, host = DEFAULT_HOST) {
  let start = Number.parseInt(preferred, 10);
  if (!Number.isInteger(start)) start = 0;
  let width = Number.parseInt(span, 10);
  if (!Number.isInteger(width) || width < 0) width = SPAN;

  if (start > 0 && start <= 65535) {
    for (let port = start; port <= Math.min(start + width, 65535); port++) {
      if (await bindable(port, host)) return port;
    }
  }
  // Everything in the window is taken: let the OS name one. An odd port beats
  // a startup that gives up, and `publish()` tells everyone where it went.
  return ephemeralPort(host);
}

// ──────────────────────────────── the lock ────────────────────────────────

/**
 * Take `ports.json.lock`, run `fn`, release it.
 *
 * Created with `wx` (fails if it exists), retried, and broken when it is older
 * than five seconds — a crashed writer must not wedge every other app on the
 * machine. If the lock cannot be taken at all we proceed anyway: losing one
 * entry to a race is a smaller failure than a launcher that refuses to start.
 */
async function withLock(ledger, fn) {
  const lock = `${ledger}.lock`;
  fs.mkdirSync(path.dirname(ledger), { recursive: true });
  const deadline = Date.now() + LOCK_TIMEOUT_MS;
  let held = false;
  while (Date.now() < deadline) {
    try {
      fs.writeFileSync(lock, String(process.pid), { flag: 'wx' });
      held = true;
      break;
    } catch (err) {
      if (err && err.code !== 'EEXIST') break;
      try {
        if (Date.now() - fs.statSync(lock).mtimeMs > LOCK_STALE_MS) {
          fs.rmSync(lock, { force: true });
        }
      } catch { /* it vanished under us; loop and retry */ }
      await sleep(LOCK_RETRY_MS);
    }
  }
  try {
    return await fn();
  } finally {
    if (held) { try { fs.rmSync(lock, { force: true }); } catch { /* ignore */ } }
  }
}

function readLedger(file) {
  try {
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    return (data && typeof data === 'object' && !Array.isArray(data)) ? data : {};
  } catch {
    // Missing, empty, or corrupt. A corrupt ledger is rebuilt rather than
    // fatal: every entry in it is re-published by its owner on next launch.
    return {};
  }
}

/** temp + rename, so a torn write can never be observed by a reader. */
function writeLedger(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
  fs.renameSync(tmp, file);
}

// ───────────────────────────── publishing ─────────────────────────────

/**
 * Record where this app landed. Call it once the health route answers.
 *
 * Publishing before the server is up is what turns a ledger into a liar: the
 * next app to look sees an entry, trusts the pid, and waits on a socket that
 * will never open.
 */
export async function publish(app, port, health, extra = null, ledger = null) {
  const file = resolveLedger(ledger);
  const record = {
    port: Number.parseInt(port, 10),
    pid: process.pid,
    started_at: new Date().toISOString().replace(/\.\d{3}Z$/, '+00:00'),
    health: String(health),
    extra: { ...(extra || {}) },
  };
  await withLock(file, () => {
    const data = readLedger(file);
    data[String(app)] = record;
    writeLedger(file, data);
  });
  forget(String(app));
}

/** Remove this app's entry, and only this app's entry. */
export async function withdraw(app, ledger = null) {
  const file = resolveLedger(ledger);
  const key = String(app);
  await withLock(file, () => {
    const data = readLedger(file);
    if (key in data) {
      delete data[key];
      writeLedger(file, data);
    }
  });
  forget(key);
}

/** The whole ledger, as read. No liveness filtering — that is `discover`. */
export function entries(ledger = null) {
  return readLedger(resolveLedger(ledger));
}

/**
 * One app's entry, or null.
 *
 * For the two cases a base URL cannot express: a sub-service recorded under
 * `extra`, and a third-party server whose health route will never say `app`.
 */
export function entry(app, ledger = null) {
  const found = entries(ledger)[String(app)];
  return (found && typeof found === 'object') ? { ...found } : null;
}

// ───────────────────────────── discovery ─────────────────────────────

/**
 * Is this pid a live process?
 *
 * `process.kill(pid, 0)` is safe here even on Windows: libuv special-cases
 * signal 0 into an existence check (OpenProcess + GetExitCodeProcess) and
 * never calls TerminateProcess. The Python helper has to go the long way round
 * because CPython maps every non-zero signal onto TerminateProcess.
 */
function pidAlive(pid) {
  const n = Number.parseInt(pid, 10);
  if (!Number.isInteger(n) || n <= 0) return false;
  try {
    process.kill(n, 0);
    return true;
  } catch (err) {
    // EPERM means it exists and belongs to someone else.
    return !!err && err.code === 'EPERM';
  }
}

/** 'http://127.0.0.1:8098/health' -> 'http://127.0.0.1:8098'. */
export function baseOf(url) {
  const raw = String(url);
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    return raw.replace(/\/+$/, '');
  }
  // localhost resolves to ::1 first on Windows and costs ~2 s per dead call.
  const host = parsed.hostname.toLowerCase() === 'localhost' ? DEFAULT_HOST : parsed.hostname;
  return `${parsed.protocol}//${host}${parsed.port ? `:${parsed.port}` : ''}`;
}

function healthUrl(base, healthPath) {
  return `${String(base).replace(/\/+$/, '')}/${String(healthPath).replace(/^\/+/, '')}`;
}

/** A cheap TCP connect, so a scan of 21 dead ports costs milliseconds. */
function listening(host, port) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const done = (ok) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(CONNECT_TIMEOUT_MS);
    socket.once('connect', () => done(true));
    socket.once('timeout', () => done(false));
    socket.once('error', () => done(false));
    try { socket.connect(port, host); } catch { done(false); }
  });
}

/** The health body as an object, `{}` for a 200 that is not JSON, null for no answer. */
function probe(url, timeoutMs = HEALTH_TIMEOUT_MS) {
  return new Promise((resolve) => {
    let req;
    const fail = () => { try { req?.destroy(); } catch { /* ignore */ } resolve(null); };
    try {
      req = http.get(url, {
        timeout: timeoutMs,
        headers: { 'User-Agent': 'buzzcaf-ports/1' },
      }, (res) => {
        if (res.statusCode !== 200) { res.resume(); return fail(); }
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          body += chunk;
          if (body.length > 65536) { res.destroy(); }
        });
        res.on('end', () => {
          try {
            const parsed = JSON.parse(body);
            resolve(parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {});
          } catch {
            resolve({});
          }
        });
        res.on('error', fail);
      });
    } catch {
      return resolve(null);
    }
    req.on('timeout', fail);
    req.on('error', fail);
  });
}

/**
 * True only when the listener at `base` says it is `app`.
 *
 * This is rule 4, and it is the whole reason discovery is trustworthy: a scan
 * that skips it is what once mistook CaliberAI for the Studio.
 */
export async function identifies(base, healthPath, app) {
  const body = await probe(healthUrl(base, healthPath));
  return !!body && body.app === app;
}

const cache = new Map(); // app -> { expires, url }

function cached(app) {
  const found = cache.get(app);
  if (!found) return { hit: false, url: null };
  if (Date.now() >= found.expires) return { hit: false, url: null };
  return { hit: true, url: found.url };
}

function remember(app, url, ttl) {
  if (ttl > 0) cache.set(app, { expires: Date.now() + ttl * 1000, url });
  return url;
}

/** Drop a cached answer (all of them when called with no app). */
export function forget(app = null) {
  if (app === null) cache.clear();
  else cache.delete(String(app));
}

/**
 * Where `app` is listening, as a base URL — or null when it is not.
 *
 * Order (rule 5): an explicit env URL → the ledger entry whose pid is alive
 * *and* whose health names the app → a scan of `preferred … preferred+span`
 * with the same identity check → null. The result, offline included, is cached
 * for `ttl` seconds.
 */
export async function discover(
  app,
  preferred,
  healthPath = '/health',
  span = SPAN,
  envUrl = null,
  ttl = DEFAULT_TTL,
  ledger = null,
  host = DEFAULT_HOST,
) {
  const key = String(app);
  const { hit, url } = cached(key);
  if (hit) return url;
  const found = await discoverNow(key, preferred, healthPath, span, envUrl, ledger, host);
  return remember(key, found, ttl);
}

async function discoverNow(app, preferred, healthPath, span, envUrl, ledger, host) {
  // 1. The owner said where it is. `<APP>_URL` is the convention when the
  //    caller passes nothing. Not identity-checked: an explicit override wins.
  const explicit = envUrl || process.env[`${app.toUpperCase()}_URL`];
  if (explicit && String(explicit).trim()) return baseOf(String(explicit).trim());

  // 2. The ledger: an entry is only believed when its process is alive and the
  //    listener still names itself.
  const found = entry(app, ledger);
  if (found && Number.isInteger(Number(found.port)) && pidAlive(found.pid)) {
    const base = found.health ? baseOf(found.health) : `http://${host}:${found.port}`;
    if (await identifies(base, healthPath, app)) return base;
  }

  // 3. The scan, for an app that is running but never published — started by
  //    hand, or from a build that predates this helper.
  const start = Number.parseInt(preferred, 10);
  let width = Number.parseInt(span, 10);
  if (!Number.isInteger(width) || width < 0) width = SPAN;
  if (Number.isInteger(start) && start > 0) {
    for (let port = start; port <= Math.min(start + width, 65535); port++) {
      if (!(await listening(host, port))) continue;
      const base = `http://${host}:${port}`;
      if (await identifies(base, healthPath, app)) return base;
    }
  }

  return null;
}

export default {
  SPAN, DEFAULT_HOST, ledgerPath, pickPort, publish, withdraw,
  entries, entry, discover, identifies, baseOf, forget,
};
