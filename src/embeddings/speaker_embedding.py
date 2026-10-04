"""Speaker id <-> name mapping shared by preprocessing and inference."""

import json
from pathlib import Path

from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeakerMap:
    """A stable, persisted name -> integer id assignment."""

    def __init__(self, mapping: dict[str, int] | None = None):
        self._mapping: dict[str, int] = dict(mapping or {})

    def add(self, name: str) -> int:
        if name not in self._mapping:
            self._mapping[name] = len(self._mapping)

        return self._mapping[name]

    def get(self, name: str) -> int:
        if name not in self._mapping:
            raise DatasetError(
                f"Unknown speaker '{name}'; known: {sorted(self._mapping)}"
            )

        return self._mapping[name]

    def name_of(self, speaker_id: int) -> str:
        for name, value in self._mapping.items():
            if value == speaker_id:
                return name

        raise DatasetError(f"No speaker with id {speaker_id}")

    def __len__(self) -> int:
        return len(self._mapping)

    def __contains__(self, name: str) -> bool:
        return name in self._mapping

    def as_dict(self) -> dict[str, int]:
        return dict(self._mapping)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._mapping, f, indent=2, sort_keys=True)

        logger.info(f"Saved speaker map ({len(self)} speakers) -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerMap":
        path = Path(path)

        if not path.exists():
            raise DatasetError(f"Speaker map not found: {path}")

        with open(path, "r") as f:
            return cls(json.load(f))
