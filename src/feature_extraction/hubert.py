import numpy as np

from src.feature_extraction.base import (
    BaseFeatureExtractor,
)
from src.models.audio import AudioData
from src.models.features import HubertFeatures


class HubertExtractor(
    BaseFeatureExtractor
):

    def load(self):

        print("Loading HuBERT")

    def extract(
        self,
        audio: AudioData
    ) -> HubertFeatures:

        embeddings = np.random.rand(
            100,
            768
        )

        return HubertFeatures(
            embeddings=embeddings
        )