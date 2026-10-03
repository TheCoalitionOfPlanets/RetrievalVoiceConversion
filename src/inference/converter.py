from src.utils.audio import AudioLoader

from src.feature_extraction.extractor import (
    FeatureExtractor
)

from src.retrieval.retrieval_service import (
    RetrievalService
)

from src.models.vits.model import (
    VITSModel
)


class VoiceConverter:

    def __init__(self):

        self.feature_extractor = (
            FeatureExtractor()
        )

        self.retriever = (
            RetrievalService()
        )

        self.vits = (
            VITSModel()
        )

    def convert(
        self,
        audio_path: str
    ):

        audio = AudioLoader.load(
            audio_path
        )

        features = (
            self.feature_extractor.extract(
                audio
            )
        )

        retrieved = (
            self.retriever.retrieve(
                features
            )
        )

        output = self.vits.synthesize(
            retrieved,
            None
        )

        return output
