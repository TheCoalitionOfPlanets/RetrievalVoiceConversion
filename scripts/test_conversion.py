from src.models.conversion import (
    ConversionRequest
)

from src.services.conversion_service import (
    ConversionService
)

request = ConversionRequest(
    input_audio_path="input.wav",
    output_audio_path="output.wav",
    speaker_id="prieyan"
)

service = ConversionService()

service.convert(request)
