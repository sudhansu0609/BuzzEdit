// node --test electron/comfy-crash.test.cjs
//
// Unit tests for the ComfyUI exit classifier, isolated from Electron and from
// ever spawning a real process.

const test = require('node:test');
const assert = require('node:assert/strict');

const { looksLikeBrokenEnv, diedAtStartup } = require('./comfy-crash.cjs');

test('a normal startup — an optional backend missing — is not a crash', () => {
  const line = 'Found comfy_kitchen backend triton: {\'available\': False, '
    + "'unavailable_reason': \"ImportError: No module named 'triton'\"}";
  assert.equal(looksLikeBrokenEnv(line), false);
});

test('a custom node skipping its own optional dependency is not a crash', () => {
  const line = 'Cannot import C:\\Users\\singh\\Documents\\ComfyUI\\custom_nodes\\'
    + "SomeNode module for custom nodes: No module named 'loguru'";
  assert.equal(looksLikeBrokenEnv(line), false);
});

test('an (IMPORT FAILED) custom node banner is not a crash', () => {
  const line = '(IMPORT FAILED): C:\\Users\\singh\\Documents\\ComfyUI\\custom_nodes\\SomeNode';
  assert.equal(looksLikeBrokenEnv(line), false);
});

test('a full traceback from a custom node is not a crash, even on its exception line', () => {
  const text = [
    'Traceback (most recent call last):',
    '  File "C:\\Users\\singh\\Documents\\ComfyUI\\custom_nodes\\SomeNode\\__init__.py", line 3, in <module>',
    '    import loguru',
    "ModuleNotFoundError: No module named 'loguru'",
  ].join('\n');
  assert.equal(looksLikeBrokenEnv(text), false);
});

test('a mix of startup noise and nothing else is not a crash', () => {
  const text = [
    'Found comfy_kitchen backend triton: {\'unavailable_reason\': "ImportError: No module named \'triton\'"}',
    'Cannot import C:\\...\\custom_nodes\\X module for custom nodes: No module named \'loguru\'',
    '(IMPORT FAILED): ...custom_nodes\\X',
  ].join('\n');
  assert.equal(looksLikeBrokenEnv(text), false);
});

test('a real broken core environment is a crash', () => {
  assert.equal(looksLikeBrokenEnv("ModuleNotFoundError: No module named 'torch'"), true);
});

test('a real broken core environment stays a crash next to harmless startup noise', () => {
  const text = [
    'Found comfy_kitchen backend triton: {\'unavailable_reason\': "ImportError: No module named \'triton\'"}',
    'Traceback (most recent call last):',
    '  File "C:\\Users\\singh\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\main.py", line 10, in <module>',
    '    import torch',
    "ModuleNotFoundError: No module named 'torch'",
  ].join('\n');
  assert.equal(looksLikeBrokenEnv(text), true);
});

test('diedAtStartup is true within the threshold and false after it', () => {
  const spawnedAt = 1_000_000;
  assert.equal(diedAtStartup(spawnedAt, spawnedAt + 1000), true);
  assert.equal(diedAtStartup(spawnedAt, spawnedAt + 179_999), true);
  assert.equal(diedAtStartup(spawnedAt, spawnedAt + 180_001), false);
  // No recorded spawn time: never withhold the check over it.
  assert.equal(diedAtStartup(0, spawnedAt + 999_999), true);
});
