import os
import sys
import pytest


def test_python_version():
    """Python must be version 3.10 or higher."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert major >= 3 and minor >= 10, \
        f"Python 3.10 or higher required. Found {major}.{minor}"


def test_torch_installed():
    """Check that PyTorch is installed."""
    try:
        import torch
    except ImportError:
        pytest.fail("PyTorch not installed. Run: pip install torch")


def test_torch_gpu_available():
    """Check if GPU is available. This is not mandatory, but provides a warning."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU not detected. The project can still run on CPU.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        assert isinstance(gpu_name, str)


def test_chronos_installed():
    """Check that Chronos-2 can be imported."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        pytest.fail("Chronos-2 not installed. Run: pip install chronos-forecasting")


def test_dataset_files_exist():
    """Ensure train.csv and store.csv exist inside src/data."""
    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "../src/data/train.csv")
    store_path = os.path.join(base, "../src/data/store.csv")

    assert os.path.exists(train_path), f"Missing file: {train_path}"
    assert os.path.exists(store_path), f"Missing file: {store_path}"


def test_project_structure():
    """Check that the main project directories exist."""
    base = os.path.dirname(os.path.abspath(__file__))
    required_dirs = [
        "../src/data",
        "../src/features",
        "../src/models"
    ]

    for directory in required_dirs:
        path = os.path.join(base, directory)
        assert os.path.isdir(path), f"Missing directory: {path}"
