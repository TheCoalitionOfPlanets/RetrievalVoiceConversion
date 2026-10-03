class FeatureService:

    def __init__(self, extractor):
        self.extractor = extractor

    def extract(self, audio):
        return self.extractor.extract(audio)