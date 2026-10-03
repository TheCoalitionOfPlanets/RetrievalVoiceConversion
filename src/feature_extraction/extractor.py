from src.models.audio import AudioData
from src.models.features import HubertFeatures

from src.feature_extraction.hubert import (
    HubertExtractor
)


class FeatureExtractor:

    def __init__(self):

        self.hubert = HubertExtractor()

    def extract(
        self,
        audio: AudioData
    ) -> HubertFeatures:

        return self.hubert.extract(audio)
