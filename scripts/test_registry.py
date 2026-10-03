from src.services.model_registry import ModelRegistry

registry = ModelRegistry()

extractor = registry.get_feature_extractor()

print(type(extractor).__name__)
