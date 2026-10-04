from pathlib import Path

import torch

from src.utils.exceptions import CheckpointError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    learning_rate: float,
    epoch: int,
    step: int,
    path: str | Path,
    config: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    torch.save(
        {
            "model": state,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        path,
    )

    logger.info(f"Saved checkpoint -> {path} (epoch {epoch}, step {step})")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = False,
) -> tuple[int, int, float]:
    """Restore weights in place; returns (epoch, step, learning_rate)."""

    path = Path(path)

    if not path.exists():
        raise CheckpointError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("model", payload)

    target = model.module if hasattr(model, "module") else model
    current = target.state_dict()

    merged = {}
    skipped = []

    for key, value in current.items():
        incoming = saved.get(key)

        if incoming is not None and incoming.shape == value.shape:
            merged[key] = incoming
        else:
            skipped.append(key)
            merged[key] = value

    target.load_state_dict(merged, strict=strict)

    if skipped:
        logger.warning(
            f"{len(skipped)} tensors kept at init (shape mismatch or absent), "
            f"e.g. {skipped[:3]}"
        )

    if optimizer is not None and payload.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError as exc:
            logger.warning(f"Optimizer state not restored: {exc}")

    epoch = int(payload.get("epoch", 0))
    step = int(payload.get("step", 0))
    lr = float(payload.get("learning_rate", 0.0))

    logger.info(f"Loaded checkpoint {path} (epoch {epoch}, step {step})")

    return epoch, step, lr


def latest_checkpoint(directory: str | Path, prefix: str) -> Path | None:
    directory = Path(directory)

    if not directory.exists():
        return None

    candidates = sorted(
        (p for p in directory.glob(f"{prefix}_*.pth") if p.stem.split("_")[-1].isdigit()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    return candidates[-1] if candidates else None


def prune_checkpoints(directory: str | Path, prefix: str, keep: int) -> None:
    directory = Path(directory)

    candidates = sorted(
        (p for p in directory.glob(f"{prefix}_*.pth") if p.stem.split("_")[-1].isdigit()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    for stale in candidates[:-keep] if keep > 0 else []:
        stale.unlink(missing_ok=True)
        logger.info(f"Pruned old checkpoint {stale.name}")
