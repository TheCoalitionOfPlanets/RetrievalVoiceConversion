"""Discovers speakers from a dataset layout.

Two layouts are accepted:

    data/raw/<speaker>/*.wav   -> multi-speaker, folder name is the speaker
    data/raw/*.wav             -> single speaker, named after the root folder
"""

from pathlib import Path

from src.embeddings.speaker_embedding import SpeakerMap
from src.utils.audio import AUDIO_SUFFIXES
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def discover_speakers(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)

    if not root.exists():
        raise DatasetError(f"Dataset root not found: {root}")

    def audio_in(directory: Path) -> list[Path]:
        return sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )

    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir() and audio_in(d)]

    if subdirs:
        speakers = {d.name: audio_in(d) for d in subdirs}
    else:
        files = audio_in(root)

        if not files:
            raise DatasetError(f"No audio files found under {root}")

        speakers = {root.name: files}

    for name, files in speakers.items():
        logger.info(f"Speaker '{name}': {len(files)} files")

    return speakers


def build_speaker_map(root: str | Path) -> tuple[SpeakerMap, dict[str, list[Path]]]:
    speakers = discover_speakers(root)

    mapping = SpeakerMap()

    for name in sorted(speakers):
        mapping.add(name)

    return mapping, speakers
