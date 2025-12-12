@echo off
echo ============================================================
echo DNLP Project - Environment Setup (setup.bat)
echo ============================================================

REM Check if Python exists
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10.x from https://www.python.org/downloads/
    pause
    exit /b
)

REM Check if .venv already exists
IF EXIST .venv (
    echo WARNING: .venv already exists.
    echo This setup should only be used for the FIRST installation.
    echo Use "make fullrun" instead of reinstalling.
    pause
    exit /b
)

echo Creating virtual environment (.venv)...
python -m venv .venv

IF NOT EXIST .venv (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b
)

echo Upgrading pip...
.\.venv\Scripts\pip install --upgrade pip setuptools wheel

echo Installing requirements...
.\.venv\Scripts\pip install -r requirements.txt

echo Would you like to install PyTorch with GPU support? (y/n)
set /p gpuchoice="> "

IF /I "%gpuchoice%"=="y" (
    echo Installing PyTorch CUDA version...
    .\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) ELSE (
    echo Skipping GPU installation.
)

echo Testing PyTorch installation...
.\.venv\Scripts\python -c "import torch; print('Torch OK:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo Testing Chronos-2 import...
.\.venv\Scripts\python -c "from chronos import Chronos2Pipeline; print('Chronos-2 imported successfully')"

echo ============================================================
echo Setup complete!
echo To run the project:
echo     .venv\Scripts\python main.py
echo To run tests:
echo     .venv\Scripts\pytest -q
echo ============================================================

exit /b
