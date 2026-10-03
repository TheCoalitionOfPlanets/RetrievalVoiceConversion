class VoiceMogException(Exception):
    pass


class AudioLoadError(VoiceMogException):
    pass


class FeatureExtractionError(VoiceMogException):
    pass


class ConversionError(VoiceMogException):
    pass