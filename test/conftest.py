import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_model_config  # noqa: E402


@pytest.fixture(scope="session")
def model_config():
    return load_model_config("config/model.yaml")


@pytest.fixture
def tone_16k():
    """One second of a 220 Hz tone at 16 kHz — a clean, voiced test signal."""

    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)

    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def tone_32k():
    t = np.linspace(0, 1.0, 32000, endpoint=False, dtype=np.float32)

    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
