from src.feature_extraction.hubert import HubertExtractor


class ModelRegistry:

    def __init__(self):
        self._feature_extractor = None

    def get_feature_extractor(self):

        if self._feature_extractor is None:
            self._feature_extractor = HubertExtractor()

        return self._feature_extractor
