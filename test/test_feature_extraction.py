"""Pitch, spectrogram and alignment behaviour (no network downloads)."""

import numpy as np
import pytest
import torch

from src.feature_extraction.rmvpe import PitchExtractor
from src.feature_extraction.spectrogram import mel_spectrogram_torch
from src.feature_extraction.spectrogram import spectrogram_torch
from src.models.audio import AudioData
from src.utils.exceptions import FeatureExtractionError
from src.utils.settings import PitchConfig


def test_pitch_tracks_a_known_tone(tone_16k):
    extractor = PitchExtractor(PitchConfig())

    features = extractor.extract(AudioData(tone_16k, 16000))

    voiced = features.f0[features.f0 > 0]

    assert voiced.size > 0, "a pure tone must register as voiced"
    assert 200.0 < float(np.median(voiced)) < 240.0


def test_pitch_shift_is_a_semitone_ratio(tone_16k):
    extractor = PitchExtractor(PitchConfig())
    audio = AudioData(tone_16k, 16000)

    base = extractor.extract(audio).f0
    up = extractor.extract(audio, pitch_shift=12).f0

    mask = base > 0
    ratio = up[mask] / base[mask]

    assert np.allclose(ratio, 2.0, atol=1e-4)


def test_coarse_pitch_stays_in_range(tone_16k):
    extractor = PitchExtractor(PitchConfig())

    features = extractor.extract(AudioData(tone_16k, 16000))
    coarse = features.f0_coarse

    assert coarse.dtype == np.int64
    assert coarse.min() >= 0
    assert coarse.max() <= 255
    # Unvoiced frames must map to the reserved zero bin.
    assert np.all(coarse[features.f0 < 1e-3] == 0)


def test_pitch_resamples_to_requested_frames(tone_16k):
    extractor = PitchExtractor(PitchConfig())

    features = extractor.extract(AudioData(tone_16k, 16000), num_frames=137)

    assert features.f0.shape == (137,)
    assert features.f0_coarse.shape == (137,)


def test_pitch_rejects_wrong_sample_rate(tone_32k):
    extractor = PitchExtractor(PitchConfig())

    with pytest.raises(FeatureExtractionError, match="16000"):
        extractor.extract(AudioData(tone_32k, 32000))


def test_unknown_f0_method_is_rejected():
    with pytest.raises(FeatureExtractionError, match="Unknown f0 method"):
        PitchExtractor(PitchConfig(method="nonexistent"))


def test_spectrogram_frame_count_matches_hop(model_config, tone_32k):
    audio = model_config.audio

    wav = torch.from_numpy(tone_32k).unsqueeze(0)
    spec = spectrogram_torch(
        wav, audio.filter_length, audio.hop_length, audio.win_length
    )

    assert spec.shape[1] == audio.filter_length // 2 + 1
    assert spec.shape[2] == tone_32k.size // audio.hop_length


def test_mel_spectrogram_shape(model_config, tone_32k):
    audio = model_config.audio

    mel = mel_spectrogram_torch(
        torch.from_numpy(tone_32k).unsqueeze(0),
        audio.filter_length,
        audio.n_mel_channels,
        audio.sample_rate,
        audio.hop_length,
        audio.win_length,
        audio.mel_fmin,
        audio.mel_fmax,
    )

    assert mel.shape[1] == audio.n_mel_channels
    assert torch.isfinite(mel).all()


def test_content_upsampling_factor_matches_rates(model_config):
    """HuBERT runs at 50 fps; a 32k/hop-320 decoder needs 100 fps."""

    from src.feature_extraction.extractor import FeatureExtractor

    extractor = FeatureExtractor(model_config)

    expected = (model_config.audio.sample_rate / model_config.audio.hop_length) / (
        model_config.audio.hubert_sample_rate / 320.0
    )

    assert extractor.upsample_factor == round(expected)
