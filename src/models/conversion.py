from dataclasses import dataclass


@dataclass
class ConversionRequest:
    input_audio_path: str
    speaker_id: str


@dataclass
class ConversionResponse:
    output_audio_path: str