"""Retrieval, config plumbing and audio helpers."""

import numpy as np
import pytest

from src.embeddings.speaker_embedding import SpeakerMap
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.retrieval_service import RetrievalService
from src.utils.config import load_inference_config
from src.utils.config import load_model_config
from src.utils.config import load_train_config
from src.utils.exceptions import DatasetError
from src.utils.exceptions import RetrievalError
from src.utils.settings import IndexConfig


@pytest.fixture
def feature_bank():
    rng = np.random.default_rng(0)

    return rng.standard_normal((2000, 64)).astype(np.float32)


def test_index_roundtrips_through_disk(tmp_path, feature_bank):
    index = FaissIndex().build(feature_bank)

    index_path = tmp_path / "s.index"
    features_path = tmp_path / "s.npy"
    index.save(index_path, features_path)

    reloaded = FaissIndex.load(index_path, features_path)

    assert reloaded.ntotal == feature_bank.shape[0]
    assert reloaded.dim == 64


def test_search_returns_vectors_near_the_query(feature_bank):
    index = FaissIndex().build(feature_bank)

    queries = feature_bank[:16]
    retrieved = index.search(queries, k=4)

    assert retrieved.shape == queries.shape
    assert np.isfinite(retrieved).all()

    # Retrieving a stored vector should land closer to it than a random vector.
    random_vec = np.random.default_rng(1).standard_normal(64).astype(np.float32)

    own = np.linalg.norm(retrieved[0] - queries[0])
    other = np.linalg.norm(random_vec - queries[0])

    assert own < other


def test_index_rejects_wrong_dimensionality():
    with pytest.raises(RetrievalError, match=r"\(N, D\)"):
        FaissIndex().build(np.zeros((10, 4, 4), dtype=np.float32))


def test_index_rejects_a_single_vector():
    with pytest.raises(RetrievalError, match="at least 2"):
        FaissIndex().build(np.zeros((1, 8), dtype=np.float32))


def test_rate_zero_is_a_passthrough(tmp_path, feature_bank):
    index = FaissIndex().build(feature_bank)
    index.save(tmp_path / "s.index", tmp_path / "s.npy")

    service = RetrievalService(
        IndexConfig(path=str(tmp_path / "s.index"), features=str(tmp_path / "s.npy"))
    ).load()

    query = feature_bank[:8]
    out = service.retrieve(query, rate=0.0)

    assert np.array_equal(out, query)


def test_rate_one_fully_replaces_the_query(tmp_path, feature_bank):
    index = FaissIndex().build(feature_bank)
    index.save(tmp_path / "s.index", tmp_path / "s.npy")

    service = RetrievalService(
        IndexConfig(path=str(tmp_path / "s.index"), features=str(tmp_path / "s.npy"))
    ).load()

    query = np.random.default_rng(2).standard_normal((8, 64)).astype(np.float32)

    blended = service.retrieve(query, rate=1.0)
    pure = service.index.search(query, k=service.config.k)

    assert np.allclose(blended, pure, atol=1e-5)


def test_missing_index_degrades_instead_of_crashing(tmp_path):
    service = RetrievalService(
        IndexConfig(path=str(tmp_path / "absent.index"))
    ).load()

    assert not service.is_available

    query = np.zeros((4, 64), dtype=np.float32)

    assert np.array_equal(service.retrieve(query), query)


def test_speaker_map_assigns_stable_ids(tmp_path):
    mapping = SpeakerMap()

    assert mapping.add("alice") == 0
    assert mapping.add("bob") == 1
    assert mapping.add("alice") == 0

    path = tmp_path / "speakers.json"
    mapping.save(path)

    assert SpeakerMap.load(path).as_dict() == {"alice": 0, "bob": 1}


def test_unknown_speaker_raises():
    with pytest.raises(DatasetError, match="Unknown speaker"):
        SpeakerMap({"alice": 0}).get("carol")


def test_shipped_configs_parse():
    model = load_model_config("config/model.yaml")
    train = load_train_config("config/train.yaml")
    inference = load_inference_config("config/inference.yaml")

    assert model.audio.sample_rate == 32000
    assert train.training.batch_size > 0
    assert 0.0 <= inference.index.rate <= 1.0


def test_upsample_rates_multiply_to_the_hop_length():
    """If these disagree the decoder emits the wrong number of samples."""

    model = load_model_config("config/model.yaml")

    assert int(np.prod(model.generator.upsample_rates)) == model.audio.hop_length


def test_segment_size_divides_into_whole_frames():
    model = load_model_config("config/model.yaml")
    train = load_train_config("config/train.yaml")

    assert train.training.segment_size % model.audio.hop_length == 0


def test_model_config_survives_a_dict_roundtrip():
    from src.utils.settings import ModelConfig

    original = load_model_config("config/model.yaml")
    restored = ModelConfig.from_dict(original.to_dict())

    assert restored.audio == original.audio
    assert restored.generator.upsample_rates == original.generator.upsample_rates


def test_peak_normalisation_hits_the_target():
    from src.utils.audio import normalize_peak

    signal = np.array([0.1, -0.2, 0.05], dtype=np.float32)

    assert np.abs(normalize_peak(signal, 0.95)).max() == pytest.approx(0.95)


def test_peak_normalisation_leaves_silence_alone():
    from src.utils.audio import normalize_peak

    silence = np.zeros(100, dtype=np.float32)

    assert np.array_equal(normalize_peak(silence), silence)


def test_audio_slicing_produces_bounded_chunks():
    from src.utils.audio import slice_audio

    waveform = np.random.default_rng(3).standard_normal(16000 * 10).astype(np.float32)

    chunks = slice_audio(waveform, 16000, chunk_seconds=3.0, overlap_seconds=0.3)

    assert len(chunks) > 1
    assert all(c.size <= 3 * 16000 for c in chunks)
    assert all(c.size >= 0.5 * 16000 for c in chunks)


def test_short_voiced_runs_are_merged_not_discarded():
    """Syllable-length runs must survive; dropping them empties a corpus."""

    from src.utils.audio import split_on_silence

    sr = 16000
    rng = np.random.default_rng(4)

    # 0.3 s of tone / 0.1 s of silence, repeated — every run is under min_seconds.
    tone = np.sin(
        2 * np.pi * 200 * np.linspace(0, 0.3, int(0.3 * sr), endpoint=False)
    ).astype(np.float32)
    gap = (1e-5 * rng.standard_normal(int(0.1 * sr))).astype(np.float32)

    waveform = np.concatenate([np.concatenate([tone, gap]) for _ in range(20)])

    pieces = split_on_silence(waveform, sr, min_seconds=0.5, max_seconds=4.0)

    assert pieces, "short runs must be merged into usable chunks"
    assert all(p.size >= 0.5 * sr for p in pieces)
    assert all(p.size <= 4.0 * sr for p in pieces)

    kept = sum(p.size for p in pieces) / waveform.size
    assert kept > 0.5, f"only {kept:.0%} of the audio survived slicing"


def test_silence_only_input_yields_nothing():
    from src.utils.audio import split_on_silence

    assert split_on_silence(np.zeros(16000, dtype=np.float32), 16000) == []
