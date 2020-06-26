from .dku_model import DkuModel
import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.constants as constants


class ScoreModel(DkuModel):
    def __init__(self, input_model_folder, config):
        super(ScoreModel, self).__init__(input_model_folder, config)
        self.load()

    def load(self):
        super(ScoreModel, self).load(
            mf_path=self.input_model_folder,
            goal=constants.SCORING
        )
        self.model = self.base_model

    def classify(self, images_folder, label_df):
        images_paths = images_folder.list_paths_in_partition()
        return utils.score(
            dku_model=self,
            images_folder=images_folder,
            images_paths=images_paths,
            min_threshold=self.config.min_threshold,
            limit=self.config.max_nb_labels,
            labels_df=label_df
        )