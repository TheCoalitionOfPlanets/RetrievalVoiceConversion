from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingSample:
    """One preprocessed utterance as stored on disk by the preprocess stage."""

    wav_path: str
    feature_path: str
    f0_path: str
    f0_coarse_path: str
    speaker_id: int
    num_frames: int

    @classmethod
    def from_line(cls, line: str) -> "TrainingSample":
        parts = line.strip().split("|")

        if len(parts) != 6:
            raise ValueError(f"Malformed filelist entry: {line!r}")

        return cls(
            wav_path=parts[0],
            feature_path=parts[1],
            f0_path=parts[2],
            f0_coarse_path=parts[3],
            speaker_id=int(parts[4]),
            num_frames=int(parts[5]),
        )

    def to_line(self) -> str:
        return "|".join(
            [
                self.wav_path,
                self.feature_path,
                self.f0_path,
                self.f0_coarse_path,
                str(self.speaker_id),
                str(self.num_frames),
            ]
        )


@dataclass
class FeatureBundle:
    """Everything the extractor produces for a single utterance."""

    waveform: np.ndarray
    sample_rate: int
    content: np.ndarray
    f0: np.ndarray
    f0_coarse: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(min(self.content.shape[0], self.f0.shape[0]))
