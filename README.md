# BuzzEdit - AI-Powered Video Editor

## Project Structure
- `backend/` - FastAPI backend for video processing pipeline
- `src/` - React + Electron frontend
- `electron/` - Electron main process
- `workflows/` - ComfyUI workflow templates
- `data/` - Project data, job queue, state

## Setup
```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Start development
npm run dev
```
