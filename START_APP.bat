@echo off
:: BuzzEdit launcher shim.
::
:: A .bat is a console program, so this window exists for as long as the script
:: does -- which is why launching the app used to open a cmd window alongside the
:: UI. It now hands off to START_APP.vbs (a windowless host) and exits
:: immediately, so the console is gone within a moment of the double-click.
::
:: For a visible console -- watching the build, reading Electron's stdout --
:: run:  START_APP.bat --console

if /I "%~1"=="--console" goto :console

start "" wscript.exe "%~dp0START_APP.vbs"
exit /b 0

:console
title BuzzEdit Desktop App
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_APP.ps1"
