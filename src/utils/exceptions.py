class VoiceMogException(Exception):
    pass


class AudioLoadError(VoiceMogException):
    pass


class FeatureExtractionError(VoiceMogException):
    pass


class ConversionError(VoiceMogException):
    pass


class ConfigError(VoiceMogException):
    pass


class DatasetError(VoiceMogException):
    pass


class CheckpointError(VoiceMogException):
    pass


class RetrievalError(VoiceMogException):
    pass
