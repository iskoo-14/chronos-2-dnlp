# =========================================================
# DNLP Project Makefile
# =========================================================

PYTHON=python
VENV=.venv
REQ=requirements.txt


# ---------------------------------------------------------
# Create virtual environment and install dependencies (CPU)
# ---------------------------------------------------------
env:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip setuptools wheel || $(VENV)/bin/pip install --upgrade pip setuptools wheel
	$(VENV)/Scripts/pip install -r $(REQ) || $(VENV)/bin/pip install -r $(REQ)
	@echo "Virtual environment created successfully."


# ---------------------------------------------------------
# Install PyTorch with GPU support (CUDA 12.1 default)
# ---------------------------------------------------------
gpu:
	@echo "Installing PyTorch with CUDA support..."
	$(VENV)/Scripts/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || $(VENV)/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@echo "PyTorch GPU installation completed."


# ---------------------------------------------------------
# Full initialization: env + envtest + robustness + run
# ---------------------------------------------------------
init:
	make env
	make envtest
	make test
	make run
	@echo "Project initialized successfully."


# ---------------------------------------------------------
# Quick environment checker
# ---------------------------------------------------------
check:
	$(VENV)/Scripts/pytest tests/test_environment.py -q || $(VENV)/bin/pytest tests/test_environment.py -q


# ---------------------------------------------------------
# Full run with environment check (for existing environments)
# ---------------------------------------------------------
fullrun:
	make check
	make run
	make robustness
	@echo "Full run completed successfully."


# ---------------------------------------------------------
# Run the full forecasting pipeline
# ---------------------------------------------------------
run:
	$(VENV)/Scripts/python main.py || $(VENV)/bin/python main.py


# ---------------------------------------------------------
# Run all pytest tests (environment + robustness)
# ---------------------------------------------------------
test:
	$(VENV)/Scripts/pytest -q || $(VENV)/bin/pytest -q


# ---------------------------------------------------------
# Run environment tests only
# ---------------------------------------------------------
envtest:
	$(VENV)/Scripts/pytest tests/test_environment.py -q || $(VENV)/bin/pytest tests/test_environment.py -q


# ---------------------------------------------------------
# Run robustness tests only
# ---------------------------------------------------------
robustness:
	$(VENV)/Scripts/pytest tests/test_robustness.py -q || $(VENV)/bin/pytest tests/test_robustness.py -q


# ---------------------------------------------------------
# Remove the virtual environment
# ---------------------------------------------------------
clean:
	rmdir /S /Q $(VENV) 2>nul || rm -rf $(VENV)
	@echo "Environment removed."
