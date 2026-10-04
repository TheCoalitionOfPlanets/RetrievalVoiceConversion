"""HuBERT / ContentVec content-feature extraction."""

import numpy as np
import torch

from src.feature_extraction.base import BaseFeatureExtractor
from src.models.audio import AudioData
from src.models.features import HubertFeatures
from src.utils.exceptions import FeatureExtractionError
from src.utils.settings import HubertConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

HUBERT_SAMPLE_RATE = 16000


class HubertExtractor(BaseFeatureExtractor):
    """Wraps a HF HuBERT checkpoint and returns frame-level content vectors.

    The model runs at 16 kHz and emits one frame per 320 input samples (50 fps).
    """

    def __init__(
        self,
        config: HubertConfig | None = None,
        device: torch.device | str = "cpu",
        fp16: bool = False,
    ):
        self.config = config or HubertConfig()
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model = None

    def load(self) -> "HubertExtractor":
        if self.model is not None:
            return self

        from transformers import HubertModel

        logger.info(f"Loading HuBERT: {self.config.model_name}")

        try:
            model = HubertModel.from_pretrained(self.config.model_name)
        except Exception as exc:
            raise FeatureExtractionError(
                f"Could not load HuBERT '{self.config.model_name}': {exc}"
            ) from exc

        model = model.to(self.device).eval()

        if self.fp16:
            model = model.half()

        self.model = model

        return self

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def extract(self, audio: AudioData) -> HubertFeatures:
        if self.model is None:
            self.load()

        if audio.sample_rate != HUBERT_SAMPLE_RATE:
            raise FeatureExtractionError(
                f"HuBERT expects {HUBERT_SAMPLE_RATE} Hz, got {audio.sample_rate}"
            )

        waveform = torch.from_numpy(audio.waveform).float()
        waveform = waveform.unsqueeze(0).to(self.device)

        if self.fp16:
            waveform = waveform.half()

        outputs = self.model(waveform, output_hidden_states=True)

        layer = self.config.output_layer
        hidden_states = outputs.hidden_states

        if layer >= len(hidden_states):
            raise FeatureExtractionError(
                f"Requested layer {layer} but model has {len(hidden_states) - 1}"
            )

        features = hidden_states[layer].squeeze(0).float().cpu().numpy()

        return HubertFeatures(embeddings=features.astype(np.float32))

    @torch.no_grad()
    def extract_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        """Batched variant for the inference path; returns (B, T, D) on device."""

        if self.model is None:
            self.load()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        if self.fp16:
            waveform = waveform.half()

        outputs = self.model(waveform, output_hidden_states=True)

        return outputs.hidden_states[self.config.output_layer].float()
