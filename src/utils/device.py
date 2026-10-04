import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config string into a concrete torch device."""

    if spec != "auto":
        return torch.device(spec)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)

    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1024**3

    return f"{device} ({name}, {total:.1f} GB)"
