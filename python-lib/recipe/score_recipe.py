from .dku_recipe import DkuRecipe
import dku_deeplearning_image.constants as constants
from utils_objects import DkuModel


class ScoreRecipe(DkuRecipe):
    def __init__(self, config):
        super(ScoreRecipe, self).__init__(config)

    def load_dku_model(self, model_folder):
        self.dku_model = DkuModel(model_folder)
        self.dku_model.load_model(self.config, constants.SCORING)

    def compute(self, images_folder, model_folder):
        self.load_dku_model(model_folder)
        return self.dku_model.score_image_folder(
            images_folder=images_folder,
            min_threshold=self.config.min_threshold,
            limit=self.config.max_nb_labels,
            classify=True
        )