from .dku_config import DkuConfig


class ExtractConfig(DkuConfig):
    def __init__(self):
        self.name = 'extract'
        self.output_role = 'feature_dataset'
        super(ExtractConfig, self).__init__()

    def _load_recipe_param(self):
        super(ExtractConfig, self)._load_recipe_param()
        self.extract_layer_index = int(self.recipe_config['extract_layer_index'])
