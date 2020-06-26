from .dku_config import DkuConfig


class ScoreConfig(DkuConfig):
    def __init__(self):
        self.name = 'score'
        self.output_role = 'scored_dataset'

    def _load_recipe_param(self):
        super(ScoreConfig, self)._load_recipe_param()
        self.max_nb_labels = int(self.recipe_config['max_nb_labels'])
        self.min_threshold = float(self.recipe_config['min_threshold'])
        print("####################################################")
        print("#################### {} ########################".format(self.max_nb_labels))
        print("#################### {} ########################".format(self.min_threshold))
        print("####################################################")
