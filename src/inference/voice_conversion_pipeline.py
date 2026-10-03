from src.inference.pipeline import BasePipeline


class VoiceConversionPipeline(BasePipeline):

    def run(self):

        print("Extract HuBERT Features")

        print("Extract Pitch")

        print("Retrieve Features")

        print("Generate Audio")