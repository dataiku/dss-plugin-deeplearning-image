from .dku_model import DkuModel
import dku_deeplearning_image.utils as utils
from keras.models import Model
import dku_deeplearning_image.constants as constants

class ExtractModel(DkuModel):
    def __init__(self, input_model_folder, config):
        super().__init__(config)
        self.input_model_folder = input_model_folder
        self.load()

    def load(self):
        super().load(
            mf_path=self.input_model_folder,
            goal=constants.SCORING
        )
        self.model = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.layers[self.config.extract_layer_index].output
        )

    def extract_features(self, images_folder):
        images_paths = images_folder.list_paths_in_partition()
        return utils.score(
            dku_model=self,
            images_folder=images_folder,
            images_paths=images_paths
        )