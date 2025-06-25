@echo off
REM ============================================================================
REM setup.bat — Windows one-shot environment bootstrapper
REM ----------------------------------------------------------------------------
REM 1) Creates a venv (if it doesn't exist)
REM 2) Activates it
REM 3) Pip-installs requirements.txt
REM ============================================================================
setlocal enabledelayedexpansion
set "VENV_DIR=venv"

REM ---------------------------------------------------------------------------
REM Locate a working Python interpreter
REM ---------------------------------------------------------------------------
set "PY=py -3"
%PY% --version >NUL 2>&1 || (
    set "PY=python"
    %PY% --version >NUL 2>&1 || (
        echo [ERROR] No Python 3 interpreter found. Install Python 3 and try again.
        exit /b 1
    )
)

REM ---------------------------------------------------------------------------
REM 1) Create venv if missing
REM ---------------------------------------------------------------------------
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Virtual environment already exists: %VENV_DIR%
) else (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

REM ---------------------------------------------------------------------------
REM 2) Activate venv
REM ---------------------------------------------------------------------------
echo [INFO] Activating virtual environment ...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 3) Install requirements
REM ---------------------------------------------------------------------------
if not exist requirements.txt (
    echo [WARN] requirements.txt not found — skipping package install.
) else (
    echo [INFO] Upgrading pip ...
    python -m pip install --upgrade pip
    echo [INFO] Installing dependencies from requirements.txt ...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        exit /b 1
    )
)

echo.
echo [SUCCESS] Environment is ready. Remember: this CMD session is now inside the venv.
echo          To deactivate later, type:  deactivate
exit /b 0
