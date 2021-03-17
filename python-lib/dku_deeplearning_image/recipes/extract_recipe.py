from .dku_recipe import DkuRecipe
import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.misc_objects import DkuModel


class ExtractRecipe(DkuRecipe):
    def __init__(self, config):
        super(ExtractRecipe, self).__init__(config)

    def load_dku_model(self, model_folder):
        self.dku_model = DkuModel(model_folder)
        self.dku_model.load_model(self.config, constants.SCORE)
        self.dku_model.truncate_output(self.config.extract_layer_index)

    def compute(self, images_folder, model_folder):
        self.load_dku_model(model_folder)
        return self.dku_model.score_image_folder(
            images_folder=images_folder,
            classify=False
        )
