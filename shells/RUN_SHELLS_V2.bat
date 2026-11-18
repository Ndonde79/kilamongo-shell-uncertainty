@echo off
rem Kilamongo v2 dataset (exploration + GC export): rebuild the three DeepRBF 0.3 g/t shells with the same settings as before.
cd /d "%~dp0"
title Kilamongo v2 shells - do not close
set "PY=python"
"%PY%" -X faulthandler -u -m deeprbf.workflow.shell kilamongo_indicator.json > shells_console.txt 2>&1
echo indicator exit %ERRORLEVEL% >> shells_console.txt
"%PY%" -X faulthandler -u -m deeprbf.workflow.shell kilamongo_signed_distance.json >> shells_console.txt 2>&1
echo signed_distance exit %ERRORLEVEL% >> shells_console.txt
"%PY%" -X faulthandler -u -m deeprbf.workflow.shell kilamongo_contacts.json >> shells_console.txt 2>&1
echo contacts exit %ERRORLEVEL% >> shells_console.txt
type shells_console.txt
echo Finished.
pause
