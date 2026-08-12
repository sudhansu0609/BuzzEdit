@echo off
title Buzzcaf Editor Desktop App
color 0A
cls

set COMFY_DIR=B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable
set WORKSPACE_DIR=%~dp0

:: 1. Ensure production React dist bundle is built
if not exist "%WORKSPACE_DIR%dist\index.html" (
    echo Building Buzzcaf Editor desktop bundle...
    call npm run build
)

:: 2. Start Portable ComfyUI in background
echo Starting AI Engine (ComfyUI GPU)...
start "ComfyUI AI Engine" /min cmd /c "cd /d %COMFY_DIR% && .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build"

:: 3. Start Electron Desktop Application
echo Launching Buzzcaf Editor Desktop App...
cd /d %WORKSPACE_DIR%
set NODE_ENV=production
npx electron .
