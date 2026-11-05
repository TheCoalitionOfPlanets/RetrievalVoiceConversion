"""Orchestrates content + pitch extraction for a single utterance."""

import numpy as np

from src.feature_extraction.hubert import HubertExtractor
from src.feature_extraction.rmvpe import PitchExtractor
from src.models.audio import AudioData
from src.models.dataset import FeatureBundle
from src.models.features import HubertFeatures
from src.utils.audio import resample
from src.utils.logger import get_logger
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class FeatureExtractor:
    """Turns raw audio into the (content, f0, f0_coarse) triple the model wants.

    HuBERT emits 50 frames/sec at 16 kHz. The decoder consumes
    sample_rate / hop_length frames/sec (100 at 32 kHz / hop 320), so content
    frames are repeat-interleaved to match before anything is aligned.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str = "cpu",
        fp16: bool = False,
    ):
        self.config = config or ModelConfig()

        self.hubert = HubertExtractor(
            self.config.hubert, device=device, fp16=fp16
        )
        self.pitch = PitchExtractor(self.config.pitch, device=device, fp16=fp16)

        audio = self.config.audio
        frames_per_second = audio.sample_rate / audio.hop_length
        hubert_fps = audio.hubert_sample_rate / 320.0

        self.upsample_factor = max(1, int(round(frames_per_second / hubert_fps)))

    def load(self) -> "FeatureExtractor":
        self.hubert.load()
        self.pitch.load()

        return self

    def extract_content(self, audio_16k: AudioData) -> HubertFeatures:
        features = self.hubert.extract(audio_16k)

        if self.upsample_factor > 1:
            features = HubertFeatures(
                embeddings=np.repeat(
                    features.embeddings, self.upsample_factor, axis=0
                )
            )

        return features

    def extract(
        self,
        audio: AudioData,
        pitch_shift: int = 0,
        f0_method: str | None = None,
    ) -> FeatureBundle:
        """`audio` must already be at the model's native sample rate."""

        cfg = self.config.audio

        wav_16k = (
            audio.waveform
            if audio.sample_rate == cfg.hubert_sample_rate
            else resample(audio.waveform, audio.sample_rate, cfg.hubert_sample_rate)
        )
        audio_16k = AudioData(waveform=wav_16k, sample_rate=cfg.hubert_sample_rate)

        content = self.extract_content(audio_16k).embeddings

        # The decoder needs exactly one f0 value per content frame.
        pitch = self.pitch.extract(
            audio_16k,
            num_frames=content.shape[0],
            pitch_shift=pitch_shift,
            method=f0_method,
        )

        frames = min(content.shape[0], pitch.num_frames)

        return FeatureBundle(
            waveform=audio.waveform,
            sample_rate=audio.sample_rate,
            content=content[:frames],
            f0=pitch.f0[:frames],
            f0_coarse=pitch.f0_coarse[:frames],
        )
