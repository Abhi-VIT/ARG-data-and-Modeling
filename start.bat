@echo off
setlocal
title ARG Data Studio
cd /d "%~dp0"
if errorlevel 1 goto failed

echo.
echo Starting ARG Data Studio...
echo Keep this window open while using the app.
echo.

if exist "venv\Scripts\python.exe" goto launch
echo Creating the project virtual environment...
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 -m venv venv
if errorlevel 1 goto failed
goto launch

:use_python
where python >nul 2>nul
if errorlevel 1 goto missing_python
python -m venv venv
if errorlevel 1 goto failed

:launch
"venv\Scripts\python.exe" "scripts\start_local.py" %*
if errorlevel 1 goto failed
exit /b 0

:missing_python
echo Python was not found. Install Python 3.12.10, then open start.bat again.
goto failed

:failed
echo.
echo ARG Data Studio could not start. See the message above for details.
echo Your existing data has not been deleted.
pause
exit /b 1
