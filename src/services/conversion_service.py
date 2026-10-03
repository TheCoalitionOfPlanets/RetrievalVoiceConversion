from src.models.conversion import (
    ConversionRequest
)


class ConversionService:

    def convert(
        self,
        request: ConversionRequest
    ):

        print(
            f"Converting {request.input_audio_path}"
        )
