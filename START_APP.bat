@echo off
title BuzzEdit Desktop App
color 0A
cls

:: ComfyUI Desktop install: source tree, user data (models/input/output) and its
:: own venv. The old portable build was cu118 while everything else here is cu12,
:: and mixing the two is what produced "the procedure entry point could not be
:: located in the dynamic link library".
set COMFY_CODE=C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI
set COMFY_DATA=C:\Users\singh\Documents\ComfyUI
set COMFY_PY=C:\Users\singh\Documents\ComfyUI\.venv\Scripts\python.exe
set WORKSPACE_DIR=%~dp0

:: 1. Always rebuild the React bundle so UI changes are never masked by a stale dist.
echo Building BuzzEdit desktop bundle...
call npm run build

:: 2. Start ComfyUI (Desktop install) in background
echo Starting AI Engine (ComfyUI GPU)...
:: ComfyUI logs emoji, and on the console's cp1252 codepage the logging call
:: itself throws UnicodeEncodeError and takes the process down. The child
:: inherits this setting.
set PYTHONIOENCODING=utf-8
start "ComfyUI AI Engine" /min /d "%COMFY_CODE%" "%COMFY_PY%" main.py --base-directory "%COMFY_DATA%"

:: 3. Start Electron Desktop Application
echo Launching BuzzEdit Desktop App...
cd /d %WORKSPACE_DIR%
set NODE_ENV=production
npx electron .
