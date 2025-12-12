@echo off
echo ============================================================
echo DNLP Project - Environment Setup (setup.bat)
echo ============================================================

REM ------------------------------------------------------------
REM Check if Python exists and version >= 3.9
REM ------------------------------------------------------------
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10.x from https://www.python.org/downloads/
    pause
    exit /b
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Detected Python %PYVER%

for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

IF %MAJOR% LSS 3 (
    echo ERROR: Python 3.9 or higher is required.
    pause
    exit /b
)

IF %MAJOR%==3 IF %MINOR% LSS 9 (
    echo ERROR: Python 3.9 or higher is required.
    pause
    exit /b
)

REM ------------------------------------------------------------
REM Check if .venv already exists
REM ------------------------------------------------------------
IF EXIST .venv (
    echo WARNING: .venv already exists.
    echo This setup should only be used for the FIRST installation.
    echo If the environment is already set up, do NOT rerun setup.bat.
    pause
    exit /b
)

REM ------------------------------------------------------------
REM Create virtual environment
REM ------------------------------------------------------------
echo Creating virtual environment (.venv)...
python -m venv .venv

IF NOT EXIST .venv (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b
)

REM ------------------------------------------------------------
REM Upgrade pip and core tools
REM ------------------------------------------------------------
echo Upgrading pip, setuptools, wheel...
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel

REM ------------------------------------------------------------
REM Install project requirements
REM ------------------------------------------------------------
echo Installing requirements...
.\.venv\Scripts\pip install -r requirements.txt

REM ------------------------------------------------------------
REM Optional GPU support for PyTorch
REM ------------------------------------------------------------
echo Would you like to install PyTorch with GPU support? (y/n)
set /p gpuchoice="> "

IF /I "%gpuchoice%"=="y" (
    echo Installing PyTorch CUDA version...
    .\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) ELSE (
    echo Skipping GPU installation.
)

REM ------------------------------------------------------------
REM Sanity checks
REM ------------------------------------------------------------
echo Testing PyTorch installation...
.\.venv\Scripts\python -c "import torch; print('Torch OK:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo Testing Chronos-2 import...
.\.venv\Scripts\python -c "from chronos import Chronos2Pipeline; print('Chronos-2 imported successfully')"

REM ------------------------------------------------------------
REM Done
REM ------------------------------------------------------------
echo ============================================================
echo Setup complete!
echo
echo To run the project:
echo     .venv\Scripts\python main.py
echo
echo To run tests:
echo     .venv\Scripts\pytest -q
echo ============================================================

exit /b
