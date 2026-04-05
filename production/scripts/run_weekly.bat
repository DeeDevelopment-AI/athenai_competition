@echo off
REM Weekly Pipeline Runner
REM ======================
REM Schedule this with Windows Task Scheduler to run every Saturday
REM
REM Usage:
REM   run_weekly.bat                          # Use Sharadar API (default)
REM   run_weekly.bat --csv weekly_quotes.csv  # Use CSV file instead
REM   run_weekly.bat --force-retrain          # Force model retraining

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0
set PRODUCTION_DIR=%SCRIPT_DIR%..
set PROJECT_ROOT=%PRODUCTION_DIR%\..

REM Activate virtual environment (adjust path if needed)
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
) else if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
)

REM Default to Sharadar if no arguments
if "%~1"=="" (
    echo Running weekly pipeline with Sharadar API...
    python "%SCRIPT_DIR%weekly_pipeline.py" --sharadar
) else (
    echo Running weekly pipeline with arguments: %*
    python "%SCRIPT_DIR%weekly_pipeline.py" %*
)

REM Check exit code
if %errorlevel% neq 0 (
    echo Pipeline failed with error code %errorlevel%
    exit /b %errorlevel%
)

echo Pipeline completed successfully!
