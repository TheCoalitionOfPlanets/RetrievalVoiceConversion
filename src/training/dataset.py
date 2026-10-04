"""Dataset, collate and length-bucketing for RVC training."""

from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from src.feature_extraction.spectrogram import spectrogram_torch
from src.models.dataset import TrainingSample
from src.utils.exceptions import DatasetError
from src.utils.logger import get_logger
from src.utils.settings import AudioConfig

logger = get_logger(__name__)


class VoiceConversionDataset(torch.utils.data.Dataset):
    """Reads the preprocessed filelist and serves aligned training tensors.

    Each item is (content, f0_coarse, f0, spec, wav, speaker_id), with every
    frame-rate tensor trimmed to a common length so the decoder's slicing stays
    exact.
    """

    def __init__(
        self,
        filelist: str | Path,
        audio_config: AudioConfig,
        min_frames: int = 60,
        max_frames: int = 900,
    ):
        self.audio = audio_config
        self.min_frames = min_frames
        self.max_frames = max_frames

        self.samples = self._load_filelist(filelist)
        self.lengths = [s.num_frames for s in self.samples]

        if not self.samples:
            raise DatasetError(
                f"No usable samples in {filelist} after filtering to "
                f"[{min_frames}, {max_frames}] frames"
            )

        logger.info(f"Dataset: {len(self.samples)} utterances from {filelist}")

    def _load_filelist(self, filelist: str | Path) -> list[TrainingSample]:
        path = Path(filelist)

        if not path.exists():
            raise DatasetError(
                f"Filelist not found: {path}. Run scripts/preprocess.py first."
            )

        samples, dropped = [], 0

        for line in path.read_text().splitlines():
            if not line.strip():
                continue

            sample = TrainingSample.from_line(line)

            if self.min_frames <= sample.num_frames <= self.max_frames:
                samples.append(sample)
            else:
                dropped += 1

        if dropped:
            logger.info(f"Dropped {dropped} utterances outside the frame range")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        content = torch.from_numpy(np.load(sample.feature_path)).float()
        f0 = torch.from_numpy(np.load(sample.f0_path)).float()
        f0_coarse = torch.from_numpy(np.load(sample.f0_coarse_path)).long()

        wav = torch.from_numpy(
            np.load(sample.wav_path).astype(np.float32)
        ).unsqueeze(0)

        spec = self._spectrogram(wav)

        # Align everything to the shortest of the three frame-rate tensors.
        frames = min(content.size(0), f0.size(0), spec.size(-1))

        content = content[:frames]
        f0 = f0[:frames]
        f0_coarse = f0_coarse[:frames]
        spec = spec[:, :frames]
        wav = wav[:, : frames * self.audio.hop_length]

        return (
            content,
            f0_coarse,
            f0,
            spec,
            wav,
            torch.LongTensor([sample.speaker_id]),
        )

    def _spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        spec = spectrogram_torch(
            wav,
            self.audio.filter_length,
            self.audio.hop_length,
            self.audio.win_length,
            center=False,
        )

        return torch.squeeze(spec, 0)


class VoiceConversionCollate:
    """Right-pads a batch and reports true lengths, sorted longest-first."""

    def __call__(self, batch):
        order = torch.argsort(
            torch.LongTensor([x[3].size(1) for x in batch]), descending=True
        )

        max_frames = max(x[3].size(1) for x in batch)
        max_wav = max(x[4].size(1) for x in batch)

        content_dim = batch[0][0].size(1)
        spec_dim = batch[0][3].size(0)
        size = len(batch)

        content = torch.zeros(size, max_frames, content_dim, dtype=torch.float)
        pitch = torch.zeros(size, max_frames, dtype=torch.long)
        pitchf = torch.zeros(size, max_frames, dtype=torch.float)
        spec = torch.zeros(size, spec_dim, max_frames, dtype=torch.float)
        wav = torch.zeros(size, 1, max_wav, dtype=torch.float)

        content_lengths = torch.LongTensor(size)
        spec_lengths = torch.LongTensor(size)
        wav_lengths = torch.LongTensor(size)
        sid = torch.LongTensor(size)

        for i, index in enumerate(order):
            row = batch[index]

            frames = row[0].size(0)
            content[i, :frames] = row[0]
            pitch[i, :frames] = row[1]
            pitchf[i, :frames] = row[2]
            content_lengths[i] = frames

            spec[i, :, : row[3].size(1)] = row[3]
            spec_lengths[i] = row[3].size(1)

            wav[i, :, : row[4].size(1)] = row[4]
            wav_lengths[i] = row[4].size(1)

            sid[i] = row[5][0]

        return (
            content,
            content_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wav,
            wav_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.Sampler):
    """Groups similar-length utterances so padding stays cheap.

    Single-process by design: at 4 GB there is one GPU, so num_replicas is 1.
    """

    def __init__(
        self,
        dataset: VoiceConversionDataset,
        batch_size: int,
        boundaries: list[int] | None = None,
        shuffle: bool = True,
    ):
        super().__init__()

        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries or [60, 120, 200, 300, 450, 600, 900]
        self.shuffle = shuffle
        self.epoch = 0

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_batches = self.total_size // self.batch_size

    def _bisect(self, length: int) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] < length <= self.boundaries[i + 1]:
                return i

        return -1

    def _create_buckets(self):
        buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]

        for i, length in enumerate(self.lengths):
            index = self._bisect(length)

            if index != -1:
                buckets[index].append(i)

        # Drop empty buckets, keeping boundaries consistent.
        for i in range(len(buckets) - 1, -1, -1):
            if not buckets[i]:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        counts = []

        for bucket in buckets:
            remainder = (self.batch_size - (len(bucket) % self.batch_size)) % (
                self.batch_size
            )
            counts.append(len(bucket) + remainder)

        return buckets, counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)

        indices = []

        for bucket in self.buckets:
            if self.shuffle:
                indices.append(
                    torch.randperm(len(bucket), generator=generator).tolist()
                )
            else:
                indices.append(list(range(len(bucket))))

        batches = []

        for i, bucket in enumerate(self.buckets):
            order = indices[i]
            padding = self.num_samples_per_bucket[i] - len(bucket)

            if padding and order:
                order = order + (order * (padding // len(order) + 1))[:padding]

            ids = [bucket[idx] for idx in order]

            for start in range(0, len(ids) - self.batch_size + 1, self.batch_size):
                batches.append(ids[start : start + self.batch_size])

        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]

        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches
