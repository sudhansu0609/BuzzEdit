@echo off
rem Try the background-music generator on its own (no BuzzEdit app, no video).
rem   MUSIC_TEST.bat                       -> lists the styles
rem   MUSIC_TEST.bat horror_dark           -> one 30 s track
rem   MUSIC_TEST.bat sad_violin --seconds 60 --variations 3 --tags "sarangi, rain"
cd /d "%~dp0backend"
"%~dp0.venv\Scripts\python.exe" tools\music_test.py %*
if "%~1"=="" pause
