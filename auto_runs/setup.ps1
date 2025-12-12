<# 
===============================================================================
DNLP Project - Environment Setup Script (Windows PowerShell)

IMPORTANT:
Use this script ONLY for the FIRST installation of the project.
If the .venv folder already exists, do NOT run this again.

Instead use:
    make fullrun
    make run
    make check
    make robustness

Run this script with:
    powershell -ExecutionPolicy Bypass -File setup.ps1

===============================================================================
#>

Write-Host "=== DNLP Project Environment Setup ==="

# Check if environment already exists
if (Test-Path ".venv") {
    Write-Host "WARNING: .venv already exists. Setup aborted."
    Write-Host "Use 'make fullrun' instead of reinstalling."
    exit
}

# Create virtual environment
Write-Host "Creating virtual environment .venv..."
python -m venv .venv

if (!(Test-Path ".venv")) {
    Write-Host "ERROR: Failed to create virtual environment."
    exit
}

# Upgrade pip
Write-Host "Upgrading pip..."
.\.venv\Scripts\pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing requirements..."
.\.venv\Scripts\pip install -r requirements.txt

# Ask for GPU install
$useGPU = Read-Host "Install PyTorch with GPU support? (y/n)"

if ($useGPU -eq "y") {
    Write-Host "Installing PyTorch CUDA version..."
    .\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "Skipping GPU installation."
}

# Test PyTorch installation
Write-Host "Testing PyTorch..."
.\.venv\Scripts\python -c "import torch; print('Torch OK:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Test Chronos installation
Write-Host "Testing Chronos-2..."
.\.venv\Scripts\python -c "from chronos import Chronos2Pipeline; print('Chronos-2 imported successfully')"

Write-Host "=== Setup complete ==="
Write-Host "You can now run:"
Write-Host "  .venv\Scripts\python main.py"
Write-Host "Or run tests:"
Write-Host "  .venv\Scripts\pytest -q"
