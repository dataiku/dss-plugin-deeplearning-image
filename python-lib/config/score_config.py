from .dku_config import DkuConfig
from dataiku.customrecipe import get_recipe_config

class ScoreConfig(DkuConfig):
    def __init__(self):
        self.name = 'score'
        self.output_role = 'scored_dataset'
        config = get_recipe_config()
        super(ScoreConfig, self).__init__(config)

    def _load_recipe_param(self, config):
        super(ScoreConfig, self)._load_recipe_param(config)
        self.add_param(name='extract_layer_index', value=int(self.config['max_nb_labels']))
        self.add_param(name='extract_layer_index', value=float(self.config['min_threshold']))
