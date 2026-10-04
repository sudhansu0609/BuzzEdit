@echo off
rem Generation Benchmark: current vs upgraded B-roll clip and still workflows.
rem Close Buzzcaf Studio first. Plan: docs\GENERATION_UPGRADE_PLAN.md
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\tools\run_generation_benchmark.ps1"
pause
