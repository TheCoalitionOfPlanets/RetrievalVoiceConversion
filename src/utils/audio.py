from pathlib import Path

import librosa

from src.models.audio import AudioData


class AudioLoader:

    @staticmethod
    def load(
        path: str,
        sample_rate: int = 16000
    ) -> AudioData:

        waveform, sr = librosa.load(
            path,
            sr=sample_rate,
            mono=True
        )

        return AudioData(
            waveform=waveform,
            sample_rate=sr
        )