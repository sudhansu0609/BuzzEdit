// Deciding where BuzzEdit's two servers go.
//
// Split out of main.cjs so it can be exercised without launching Electron (and
// therefore without launching ComfyUI, which owns the GPU). Nothing here
// imports `electron`; see scripts/port-plan.test.mjs.
//
// GUARDIAN_PLAN.md section 11. The preferred numbers are read on every call, so
// a test can move them with the environment the same way the owner can.

const path = require('path');
const { pathToFileURL } = require('url');

const PORT_SPAN = 20;
// 127.0.0.1, never "localhost": the backend binds IPv4 only, while on Windows
// "localhost" resolves to ::1 first, so every health poll would be refused
// before it reached the server.
const HOST = '127.0.0.1';

/** The port BuzzEdit's backend would *like*. A wish, never a fact. */
function preferredBackendPort() {
  return Number.parseInt(process.env.BUZZEDIT_PORT || '', 10) || 8099;
}

/** The port ComfyUI would like. */
function preferredComfyPort() {
  return Number.parseInt(process.env.COMFYUI_PORT || '', 10) || 8188;
}

/** The port the Vite dev server would like. */
function preferredVitePort() {
  return Number.parseInt(process.env.BUZZEDIT_VITE_PORT || '', 10) || 5173;
}

// The shared port helper is an ES module (one canonical copy per language, see
// scripts/buzzcaf-ports.mjs) and this file is CommonJS, so it comes in through
// a dynamic import — supported in the Electron main process since 28.
let portsModule = null;
async function ports() {
  if (!portsModule) {
    const file = path.join(__dirname, '..', 'scripts', 'buzzcaf-ports.mjs');
    portsModule = await import(pathToFileURL(file).href);
  }
  return portsModule;
}

/** A plain GET that resolves to the parsed JSON body, or null. */
function getJson(url, timeoutMs = 1500) {
  const http = require('http');
  return new Promise((resolve) => {
    let req;
    const fail = () => { try { req?.destroy(); } catch { /* ignore */ } resolve(null); };
    try {
      req = http.get(url, { timeout: timeoutMs }, (res) => {
        if (res.statusCode !== 200) { res.resume(); return fail(); }
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (c) => { body += c; });
        res.on('end', () => {
          try { resolve(JSON.parse(body)); } catch { resolve(null); }
        });
        res.on('error', fail);
      });
    } catch { return resolve(null); }
    req.on('timeout', fail);
    req.on('error', fail);
  });
}

/**
 * Where ComfyUI is, or where we will put it.
 *
 * ComfyUI is a third-party server (rule 8): discovered from env and by a scan,
 * never assumed. Its `/system_stats` route will never carry an `app` field, so
 * "does this look like ComfyUI" is the closest thing to an identity check
 * available — but it is still a check, and a stranger on the preferred port
 * does not pass it.
 *
 * Returns `{ port, adopt }`; `adopt` means something already serves there and
 * we must not spawn a second one.
 */
async function resolveComfy(probe = getJson) {
  const { pickPort } = await ports();
  const preferred = preferredComfyPort();

  // An explicit COMFYUI_URL is the owner speaking; take it as given.
  const explicit = (process.env.COMFYUI_URL || '').trim();
  if (explicit) {
    let port = preferred;
    try { port = Number.parseInt(new URL(explicit).port, 10) || preferred; } catch { /* keep */ }
    return { port, adopt: true, reason: 'COMFYUI_URL' };
  }

  for (let port = preferred; port <= preferred + PORT_SPAN; port++) {
    const stats = await probe(`http://${HOST}:${port}/system_stats`, 800);
    if (stats && stats.system) return { port, adopt: true, reason: 'scan' };
  }
  return { port: await pickPort(preferred, PORT_SPAN, HOST), adopt: false, reason: 'picked' };
}

/**
 * Where the backend is, or where we will put it.
 *
 * Rule 5 in full: an explicit env URL, then the ledger, then an identity scan.
 * When none of those finds a live BuzzEdit backend we pick a free port for one.
 */
async function resolveBackend() {
  const { discover, pickPort } = await ports();
  const preferred = preferredBackendPort();
  const found = await discover(
    'buzzedit', preferred, '/api/health', PORT_SPAN,
    process.env.BUZZEDIT_URL || null, 0,
  );
  if (found) {
    const port = Number.parseInt(new URL(found).port, 10) || preferred;
    return { port, adopt: true, url: found };
  }
  const port = await pickPort(preferred, PORT_SPAN, HOST);
  return { port, adopt: false, url: `http://${HOST}:${port}` };
}

module.exports = {
  HOST,
  PORT_SPAN,
  ports,
  getJson,
  preferredBackendPort,
  preferredComfyPort,
  preferredVitePort,
  resolveBackend,
  resolveComfy,
};
