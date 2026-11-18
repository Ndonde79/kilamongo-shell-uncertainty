@echo off
rem Kilamongo grade-tonnage, dataset v2: SGS + OK in the four shells, zone-oriented variogram, reporting envelope.
cd /d "%~dp0"
title Kilamongo v2 simulation - do not close
set "PY=python"
"%PY%" -X faulthandler -u gt_sim_zone2.py 50 --blocks --envelope --vario=vario_model_zone.json --tag=_v2 > gt_console_v2.txt 2>&1
echo exit code %ERRORLEVEL% >> gt_console_v2.txt
type gt_console_v2.txt
echo Finished.
pause
