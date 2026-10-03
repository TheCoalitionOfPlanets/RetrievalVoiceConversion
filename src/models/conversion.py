from dataclasses import dataclass


@dataclass
class ConversionRequest:

    input_audio_path: str

    output_audio_path: str

    speaker_id: str
