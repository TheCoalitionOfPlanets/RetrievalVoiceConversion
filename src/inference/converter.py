"""Runs a trained synthesizer over arbitrary input audio."""

import time
from pathlib import Path

import numpy as np
import torch

from src.feature_extraction.extractor import FeatureExtractor
from src.models.audio import AudioData
from src.models.conversion import ConversionRequest
from src.models.conversion import ConversionResult
from src.models.vits.model import SynthesizerTrnMs768NSFsid
from src.retrieval.retrieval_service import RetrievalService
from src.utils.audio import AudioLoader
from src.utils.audio import rms_match
from src.utils.device import describe_device
from src.utils.device import resolve_device
from src.utils.exceptions import ConversionError
from src.utils.logger import get_logger
from src.utils.settings import ConversionConfig
from src.utils.settings import InferenceConfig
from src.utils.settings import ModelConfig

logger = get_logger(__name__)


class VoiceConverter:

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ):
        self.model_config = model_config
        self.config = inference_config
        self.device = resolve_device(inference_config.device)

        self.fp16 = self.device.type == "cuda"

        self.feature_extractor = FeatureExtractor(
            model_config, device=str(self.device), fp16=self.fp16
        )
        self.retriever = RetrievalService(inference_config.index)

        self.net_g: SynthesizerTrnMs768NSFsid | None = None

    # ------------------------------------------------------------- loading

    def load(self) -> "VoiceConverter":
        if self.net_g is not None:
            return self

        logger.info(f"Loading converter on {describe_device(self.device)}")

        checkpoint = Path(self.config.checkpoint)

        if not checkpoint.exists():
            raise ConversionError(
                f"Checkpoint not found: {checkpoint}. Train a model first."
            )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # A checkpoint carries the config it was trained with; trust it over YAML.
        if payload.get("config"):
            self.model_config = ModelConfig.from_dict(payload["config"])
            self.feature_extractor = FeatureExtractor(
                self.model_config, device=str(self.device), fp16=self.fp16
            )

        net_g = SynthesizerTrnMs768NSFsid.from_config(
            self.model_config,
            self.model_config.audio.hop_length,  # segment size is unused at infer
        )
        net_g.load_state_dict(payload.get("model", payload), strict=False)
        net_g.remove_weight_norm()
        net_g = net_g.to(self.device).eval()

        if self.fp16:
            net_g = net_g.half()

        self.net_g = net_g

        self.feature_extractor.load()
        self.retriever.load()

        logger.info("Converter ready")

        return self

    # ---------------------------------------------------------- conversion

    def convert(self, request: ConversionRequest) -> ConversionResult:
        if self.net_g is None:
            self.load()

        settings = self._resolve(request)
        started = time.time()

        source = AudioLoader.load(
            request.input_audio_path, sample_rate=self.model_config.audio.sample_rate
        )

        chunks = self._chunk(source.waveform, settings)
        pieces = [self._convert_chunk(chunk, request, settings) for chunk in chunks]

        output = self._stitch(pieces, settings)

        if settings.rms_mix_rate < 1.0:
            output = rms_match(source.waveform, output, settings.rms_mix_rate)

        output = np.clip(output, -1.0, 1.0).astype(np.float32)

        result_audio = AudioData(
            waveform=output, sample_rate=self.model_config.audio.sample_rate
        )
        AudioLoader.save(result_audio, request.output_audio_path)

        elapsed = time.time() - started

        logger.info(
            f"Converted {Path(request.input_audio_path).name} "
            f"({source.duration:.1f}s) in {elapsed:.1f}s "
            f"-> {request.output_audio_path}"
        )

        return ConversionResult(
            waveform=output,
            sample_rate=self.model_config.audio.sample_rate,
            output_path=request.output_audio_path,
            duration=source.duration,
            elapsed_seconds=elapsed,
        )

    def _resolve(self, request: ConversionRequest) -> ConversionConfig:
        """Per-request overrides win over the config file."""

        base = self.config.conversion

        return ConversionConfig(
            speaker_id=request.speaker_id,
            pitch_shift=request.pitch_shift,
            f0_method=request.f0_method or base.f0_method,
            protect=base.protect if request.protect is None else request.protect,
            rms_mix_rate=base.rms_mix_rate,
            output_sample_rate=base.output_sample_rate,
            chunk_seconds=base.chunk_seconds,
            crossfade_seconds=base.crossfade_seconds,
        )

    def _chunk(
        self, waveform: np.ndarray, settings: ConversionConfig
    ) -> list[np.ndarray]:
        sample_rate = self.model_config.audio.sample_rate

        size = int(settings.chunk_seconds * sample_rate)
        overlap = int(settings.crossfade_seconds * sample_rate)

        if waveform.size <= size:
            return [waveform]

        hop = size - overlap
        chunks = []

        for start in range(0, waveform.size, hop):
            piece = waveform[start : start + size]

            if piece.size < overlap:
                break

            chunks.append(piece)

            if start + size >= waveform.size:
                break

        return chunks

    def _stitch(
        self, pieces: list[np.ndarray], settings: ConversionConfig
    ) -> np.ndarray:
        if len(pieces) == 1:
            return pieces[0]

        overlap = int(
            settings.crossfade_seconds * self.model_config.audio.sample_rate
        )
        output = pieces[0]

        for piece in pieces[1:]:
            n = min(overlap, output.size, piece.size)

            if n <= 0:
                output = np.concatenate([output, piece])
                continue

            fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
            blended = output[-n:] * (1.0 - fade) + piece[:n] * fade

            output = np.concatenate([output[:-n], blended, piece[n:]])

        return output

    @torch.no_grad()
    def _convert_chunk(
        self,
        waveform: np.ndarray,
        request: ConversionRequest,
        settings: ConversionConfig,
    ) -> np.ndarray:
        audio = AudioData(
            waveform=waveform, sample_rate=self.model_config.audio.sample_rate
        )

        bundle = self.feature_extractor.extract(
            audio,
            pitch_shift=settings.pitch_shift,
            f0_method=settings.f0_method,
        )

        content = bundle.content

        # The retrieval step: pull each frame toward the target's feature bank.
        index_rate = (
            self.config.index.rate
            if request.index_rate is None
            else request.index_rate
        )

        if self.retriever.is_available and index_rate > 0:
            retrieved = self.retriever.retrieve(content, rate=index_rate)

            # `protect` keeps unvoiced frames closer to the original features,
            # which preserves consonants that retrieval tends to smear.
            if settings.protect < 0.5:
                voiced = (bundle.f0 > 0).astype(np.float32)[:, None]
                keep = settings.protect * (1.0 - voiced)
                content = retrieved * (1.0 - keep) + content * keep
            else:
                content = retrieved

        dtype = torch.float16 if self.fp16 else torch.float32

        phone = torch.from_numpy(content).unsqueeze(0).to(self.device, dtype)
        phone_lengths = torch.LongTensor([phone.size(1)]).to(self.device)
        pitch = torch.from_numpy(bundle.f0_coarse).unsqueeze(0).to(self.device)
        nsff0 = torch.from_numpy(bundle.f0).unsqueeze(0).to(self.device, dtype)
        sid = torch.LongTensor([settings.speaker_id]).to(self.device)

        output, _, _ = self.net_g.infer(phone, phone_lengths, pitch, nsff0, sid)

        return output.float().squeeze().cpu().numpy()
