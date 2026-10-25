@echo off
rem ============================================================
rem  DeepFaceLab 4.1 - launch the GUI
rem  Double-click this file. No console window stays open.
rem ============================================================
setlocal
cd /d "%~dp0"

rem Electron must not run in plain-node mode
set "ELECTRON_RUN_AS_NODE="

set "ELECTRON=desktop\node_modules\electron\dist\electron.exe"
set "PY=%CD%\_python\Scripts\pythonw.exe"

if not exist "%ELECTRON%" goto no_electron
if not exist "%PY%" goto no_python

start "" /D "%CD%\desktop" "%ELECTRON%" .
exit /b 0

:no_electron
echo [ERROR] Electron is not installed.
echo.
echo   Missing: %ELECTRON%
echo   Fix: open a terminal in the desktop folder and run
echo.
echo       npm install
echo.
pause
exit /b 1

:no_python
echo [ERROR] Portable Python runtime not found.
echo.
echo   Missing: %PY%
echo   The _python folder must sit next to this script.
echo.
pause
exit /b 1
