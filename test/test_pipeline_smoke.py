"""End-to-end smoke test over synthetic audio.

Covers preprocess -> dataset -> index without touching the network: the HuBERT
model is replaced by a deterministic stub, so only our own wiring is exercised.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from src.models.features import HubertFeatures
from src.training.dataset import VoiceConversionCollate
from src.training.dataset import VoiceConversionDataset
from src.utils.settings import ModelConfig


class StubHubert:
    """Emits 50 fps features like the real encoder, without the weights."""

    def __init__(self, dim=768):
        self.dim = dim

    def load(self):
        return self

    def extract(self, audio):
        frames = max(1, audio.waveform.size // 320)
        rng = np.random.default_rng(frames)

        return HubertFeatures(
            embeddings=rng.standard_normal((frames, self.dim)).astype(np.float32)
        )


@pytest.fixture
def raw_dataset(tmp_path):
    """Two speakers, two files each, of voiced synthetic speech."""

    root = tmp_path / "raw"
    rng = np.random.default_rng(0)

    for speaker, f0 in (("alice", 180.0), ("bob", 110.0)):
        folder = root / speaker
        folder.mkdir(parents=True)

        for index in range(2):
            t = np.linspace(0, 3.0, 32000 * 3, endpoint=False)

            # A buzzy harmonic stack reads as voiced to the pitch tracker.
            signal = sum(
                np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)
            )
            signal = (0.3 * signal + 0.01 * rng.standard_normal(t.size)).astype(
                np.float32
            )

            sf.write(folder / f"take{index}.wav", signal, 32000)

    return root


def test_preprocess_then_train_batch(tmp_path, raw_dataset, monkeypatch):
    from src.services.feature_service import FeatureService

    config = ModelConfig()
    service = FeatureService(config)

    monkeypatch.setattr(service.extractor, "hubert", StubHubert())

    processed = tmp_path / "processed"
    count = service.preprocess(str(raw_dataset), str(processed))

    assert count > 0, "preprocessing must emit at least one segment"
    assert (processed / "filelist.txt").exists()
    assert (processed / "speakers.json").exists()

    speakers = __import__("json").loads((processed / "speakers.json").read_text())
    assert speakers == {"alice": 0, "bob": 1}

    dataset = VoiceConversionDataset(
        processed / "filelist.txt", config.audio, min_frames=1, max_frames=10_000
    )

    assert len(dataset) == count

    content, f0_coarse, f0, spec, wav, sid = dataset[0]

    frames = content.size(0)

    # The whole architecture depends on these four staying in lockstep.
    assert f0.size(0) == frames
    assert f0_coarse.size(0) == frames
    assert spec.size(1) == frames
    assert wav.size(1) == frames * config.audio.hop_length
    assert spec.size(0) == config.audio.filter_length // 2 + 1
    assert content.size(1) == config.hubert.embedding_dim

    batch = VoiceConversionCollate()([dataset[i] for i in range(min(2, len(dataset)))])

    assert batch[0].dim() == 3
    assert torch.isfinite(batch[0]).all()


def test_index_builds_from_preprocessed_features(tmp_path, raw_dataset, monkeypatch):
    from src.retrieval.faiss_index import FaissIndex
    from src.services.feature_service import FeatureService

    config = ModelConfig()
    service = FeatureService(config)
    monkeypatch.setattr(service.extractor, "hubert", StubHubert())

    processed = tmp_path / "processed"
    service.preprocess(str(raw_dataset), str(processed))

    stacked = np.concatenate(
        [np.load(f) for f in sorted((processed / "features").glob("*.npy"))], axis=0
    )

    index = FaissIndex().build(stacked)

    assert index.ntotal == stacked.shape[0]
    assert index.dim == 768
