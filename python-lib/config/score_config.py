from .dku_config import DkuConfig
from dataiku.customrecipe import get_recipe_config
import dku_deeplearning_image.constants as constants
import dku_deeplearning_image.dl_image_toolbox_utils as utils
import pandas as pd


class ScoreConfig(DkuConfig):
    def __init__(self):
        self.name = 'score'
        self.output_role = 'scored_dataset'

    def _load_recipe_param(self):
        super()._load_recipe_param()

        self.max_nb_labels = int(self.recipe_config['max_nb_labels'])
        self.min_threshold = float(self.recipe_config['min_threshold'])

    def _load_label_df(self):
        details_model_label = self.model_folder.get_path_details(constants.MODEL_LABELS_FILE)
        if details_model_label['exists'] and not details_model_label["directory"]:
            labels_path = self.model_folder.get_download_stream(constants.MODEL_LABELS_FILE)
            self.labels_df = pd.read_csv(labels_path, sep=",").set_index('id')
        else:
            print("------ \n Info: No csv file in the model folder, will not use class names. \n ------")
            self.labels_df = None

    def _load_input(self):
        super()._load_input()
        self._load_label_df()
