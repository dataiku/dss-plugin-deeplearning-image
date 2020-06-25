from .dku_model import DkuModel
import dl_image_toolbox_utils as utils
import constants

class ScoreModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = None
        self.model = None

    def load(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.SCORING
        )
        self.model = self.base_model

    def classify(self, images_folder, label_df):
        images_paths = images_folder.list_paths_in_partition()
        return utils.score(
            dku_model=self,
            images_folder=images_folder,
            images_paths=images_paths,
            limit=self.config.limit,
            min_threshold=self.config.min_threshold,
            labels_df=label_df
        )