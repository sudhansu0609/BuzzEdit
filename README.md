# BuzzEdit - AI-Powered Video Editor

## Install

Run **`installer\BuzzEdit Setup.exe`**. It installs BuzzEdit in place (in this folder) and:

1. builds the environment (`installer\setup_env.ps1`): the backend `.venv` (Python 3.11, CUDA 12.8
   PyTorch, `backend/requirements.txt`), `node_modules` and the UI bundle, the narration engine
   `engines/tts/.venv` (Chatterbox Multilingual, Kokoro, faster-whisper on torch 2.7.1+cu128) and the
   voice conversion engine `engines/seedvc` (`scripts/setup_seedvc.ps1`);
2. makes the Desktop and Start Menu (Buzzcaf Media) shortcuts to `START_APP.vbs`;
3. registers BuzzEdit in Settings > Apps and writes `installer/installed.json`, which keeps
   SmartCleaner from offering its environments for cleanup.

Needs Python 3.11 (`py -3.11`), Node.js LTS, Git and ffmpeg on PATH. Re-run Setup to repair an
install: each step skips what is already there. BuzzEdit also repairs itself: when `START_APP`
finds an environment missing (the backend or an engine venv, or Electron), it runs Setup in its
own window first and then starts, and the Studio's Health page does the same before it launches
BuzzEdit or the nightly run. `installer\Uninstall BuzzEdit.exe` removes the
shortcuts and the Apps entry and keeps the folder. Without the wizard:

```powershell
powershell -ExecutionPolicy Bypass -File installer\setup_env.ps1
```

Models download on first use (Whisper, Chatterbox, seed-vc). ComfyUI Desktop, its models and
custom nodes are described in [`workflows/README.md`](workflows/README.md).

## Project Structure
- `backend/` - FastAPI backend for video processing pipeline
- `src/` - React + Electron frontend
- `electron/` - Electron main process
- `workflows/` - every ComfyUI graph BuzzEdit runs ([README](workflows/README.md))
- `installer/` - the setup wizard and `setup_env.ps1`
- `data/` - Project data, job queue, state

## Development
```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Start development
npm run dev
```
