from pathlib import Path

import librosa

from src.models.audio import AudioData
from src.utils.exceptions import AudioLoadError


class AudioLoader:

    @staticmethod
    def load(
        path: str,
        sample_rate: int = 16000
    ) -> AudioData:

        if not Path(path).exists():
            raise AudioLoadError(
                f"File not found: {path}"
            )

        waveform, sr = librosa.load(
            path,
            sr=sample_rate,
            mono=True
        )

        return AudioData(
            waveform=waveform,
            sample_rate=sr
        )
