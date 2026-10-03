from src.feature_extraction.hubert import (
    HubertExtractor,
)
from src.utils.audio import AudioLoader

audio = AudioLoader.load(
    "sample.wav"
)

extractor = HubertExtractor()

extractor.load()

features = extractor.extract(
    audio
)

print(
    features.embeddings.shape
)
