from .dku_config import DkuConfig


class ExtractConfig(DkuConfig):
    def __init__(self):
        self.name = 'extract'
        self.output_role = 'feature_dataset'

    def _load_recipe_param(self):
        super()._load_recipe_param()
        self.extract_layer_index = int(self.recipe_config['extract_layer_index'])
