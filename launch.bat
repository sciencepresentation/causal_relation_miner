@echo off
setlocal
REM Causal Relationship Extractor - Batch Launcher (auto-setup, robust)
echo ================================================
echo   Causal Relationship Extractor
echo   Preparing environment and starting app...
echo ================================================
echo.

REM Change to the folder of this script
cd /d "%~dp0"

set "VENV_DIR=%~dp0myenv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PY%" goto :HAVE_VENV

echo [INFO] Creating Python virtual environment at "%VENV_DIR%"...
where py >nul 2>nul
if %ERRORLEVEL%==0 (
	py -3 -m venv "%VENV_DIR%"
) else (
	python -m venv "%VENV_DIR%"
)

:HAVE_VENV
if exist "%VENV_PY%" goto :VENV_OK
echo [ERROR] Could not find or create venv Python: %VENV_PY%
echo         Please install Python 3.9+ and try again.
pause
exit /b 1

:VENV_OK
echo [INFO] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip

if not exist requirements.txt goto :SKIP_REQ
echo [INFO] Installing required packages - first run may take a few minutes
"%VENV_PY%" -m pip install -r requirements.txt

:SKIP_REQ
echo [INFO] Starting Streamlit app...
"%VENV_PY%" -m streamlit run app.py

echo.
pause
endlocal
