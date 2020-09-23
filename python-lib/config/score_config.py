from .dku_config import DkuConfig


class ScoreConfig(DkuConfig):
    def __init__(self, config):
        self.name = 'score'
        self.output_role = 'scored_dataset'
        super(ScoreConfig, self).__init__(config)

    def _load_recipe_param(self, config):
        super(ScoreConfig, self)._load_recipe_param(config)
        self.add_param(name='max_nb_labels', value=int(self.config['max_nb_labels']))
        self.add_param(name='min_threshold', value=float(self.config['min_threshold']))
