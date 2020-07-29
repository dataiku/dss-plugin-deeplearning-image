from .dku_config import DkuConfig
from dataiku.customrecipe import get_recipe_config


class ExtractConfig(DkuConfig):
    def __init__(self):
        self.name = 'extract'
        self.output_role = 'feature_dataset'
        config = get_recipe_config()
        super(ExtractConfig, self).__init__(config)

    def _load_recipe_param(self, config):
        super(ExtractConfig, self)._load_recipe_param(config)
        self.add_param(name='extract_layer_index', value=int(self.config.get('extract_layer_index', -2)))
