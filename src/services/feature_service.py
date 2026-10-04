"""Preprocessing service: raw audio -> sliced wavs + features + filelist."""

from pathlib import Path

import numpy as np

from src.embeddings.speaker_encoder import build_speaker_map
from src.feature_extraction.extractor import FeatureExtractor
from src.models.audio import AudioData
from src.models.dataset import TrainingSample
from src.utils.audio import AudioLoader
from src.utils.audio import normalize_peak
from src.utils.audio import split_on_silence
from src.utils.logger import get_logger
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class FeatureService:
    """Walks the raw dataset once and writes everything training needs."""

    def __init__(
        self,
        model_config: ModelConfig,
        device: str = "cpu",
        fp16: bool = False,
    ):
        self.config = model_config
        self.extractor = FeatureExtractor(model_config, device=device, fp16=fp16)

    def preprocess(
        self,
        raw_dir: str,
        processed_dir: str,
        max_seconds: float = 4.0,
        min_seconds: float = 0.5,
    ) -> int:
        """Returns the number of training segments written."""

        self.extractor.load()

        audio_cfg = self.config.audio
        out_root = Path(processed_dir)

        for sub in ("wavs", "features", "f0", "f0_coarse"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

        speaker_map, speakers = build_speaker_map(raw_dir)
        speaker_map.save(out_root / "speakers.json")

        entries: list[TrainingSample] = []

        for name in sorted(speakers):
            speaker_id = speaker_map.get(name)

            for path in speakers[name]:
                entries.extend(
                    self._process_file(
                        path,
                        name,
                        speaker_id,
                        out_root,
                        max_seconds,
                        min_seconds,
                    )
                )

        filelist = out_root / "filelist.txt"
        filelist.write_text("\n".join(e.to_line() for e in entries) + "\n")

        total_frames = sum(e.num_frames for e in entries)
        hours = total_frames * audio_cfg.hop_length / audio_cfg.sample_rate / 3600

        logger.info(
            f"Wrote {len(entries)} segments ({hours:.2f} h usable audio) "
            f"-> {filelist}"
        )

        if len(speaker_map) > self.config.generator.spk_embed_dim:
            logger.warning(
                f"{len(speaker_map)} speakers exceeds spk_embed_dim="
                f"{self.config.generator.spk_embed_dim}; raise it in model.yaml"
            )

        return len(entries)

    def _process_file(
        self,
        path: Path,
        speaker_name: str,
        speaker_id: int,
        out_root: Path,
        max_seconds: float,
        min_seconds: float,
    ) -> list[TrainingSample]:
        audio_cfg = self.config.audio

        try:
            source = AudioLoader.load(str(path), sample_rate=audio_cfg.sample_rate)
        except Exception as exc:
            logger.warning(f"Skipping {path.name}: {exc}")
            return []

        segments = split_on_silence(
            source.waveform,
            audio_cfg.sample_rate,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )

        entries = []

        for index, segment in enumerate(segments):
            stem = f"{speaker_name}_{path.stem}_{index:04d}"

            try:
                entry = self._write_segment(
                    segment, stem, speaker_id, out_root
                )
            except Exception as exc:
                logger.warning(f"Skipping segment {stem}: {exc}")
                continue

            if entry is not None:
                entries.append(entry)

        if entries:
            logger.info(f"{path.name}: {len(entries)} segments")

        return entries

    def _write_segment(
        self,
        segment: np.ndarray,
        stem: str,
        speaker_id: int,
        out_root: Path,
    ) -> TrainingSample | None:
        audio_cfg = self.config.audio

        waveform = normalize_peak(segment).astype(np.float32)

        # Trim to a whole number of hop frames so spec/content line up exactly.
        frames = waveform.size // audio_cfg.hop_length

        if frames < 1:
            return None

        waveform = waveform[: frames * audio_cfg.hop_length]

        bundle = self.extractor.extract(
            AudioData(waveform=waveform, sample_rate=audio_cfg.sample_rate)
        )

        usable = min(frames, bundle.num_frames)

        if usable < 1:
            return None

        paths = {
            "wav": out_root / "wavs" / f"{stem}.npy",
            "feat": out_root / "features" / f"{stem}.npy",
            "f0": out_root / "f0" / f"{stem}.npy",
            "f0c": out_root / "f0_coarse" / f"{stem}.npy",
        }

        np.save(paths["wav"], waveform[: usable * audio_cfg.hop_length])
        np.save(paths["feat"], bundle.content[:usable].astype(np.float32))
        np.save(paths["f0"], bundle.f0[:usable].astype(np.float32))
        np.save(paths["f0c"], bundle.f0_coarse[:usable].astype(np.int64))

        return TrainingSample(
            wav_path=str(paths["wav"]),
            feature_path=str(paths["feat"]),
            f0_path=str(paths["f0"]),
            f0_coarse_path=str(paths["f0c"]),
            speaker_id=speaker_id,
            num_frames=usable,
        )
