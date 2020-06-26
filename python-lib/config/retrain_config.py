from .dku_config import DkuConfig
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
import dataiku
import dku_deeplearning_image.constants as constants
import numpy as np
import dku_deeplearning_image.utils as utils


class RetrainConfig(DkuConfig):
    def __init__(self):
        self.name = 'retrain'
        self.output_role = 'model_output'

    def _load_recipe_param(self):
        super(RetrainConfig, self)._load_recipe_param()

        self.list_gpu = self.recipe_config["list_gpu"]
        self.gpu_allocation = self.recipe_config["gpu_allocation"]
        self.train_ratio = float(self.recipe_config["train_ratio"])
        self.input_shape = (int(self.recipe_config["image_width"]), int(self.recipe_config["image_height"]), 3)
        self.batch_size = int(self.recipe_config["batch_size"])
        self.model_pooling = self.recipe_config["model_pooling"]
        self.model_reg = self.recipe_config["model_reg"]
        self.model_dropout = float(self.recipe_config["model_dropout"])
        self.layer_to_retrain = self.recipe_config["layer_to_retrain"]
        self.layer_to_retrain_n = int(self.recipe_config.get('layer_to_retrain_n', 0))
        self.optimizer = self.recipe_config["model_optimizer"]
        self.learning_rate = self.recipe_config["model_learning_rate"]
        self.custom_params_opti = self.recipe_config.get("model_custom_params_opti", [])
        self.nb_epochs = int(self.recipe_config["nb_epochs"])
        self.nb_steps_per_epoch = int(self.recipe_config["nb_steps_per_epoch"])
        self.nb_validation_steps = int(self.recipe_config["nb_validation_steps"])
        self.data_augmentation = self.recipe_config["data_augmentation"]
        self.n_augmentation = int(self.recipe_config["n_augmentation"])
        self.custom_params_data_augment = self.recipe_config.get("model_custom_params_data_augmentation", [])
        self.use_tensorboard = self.recipe_config["tensorboard"]
        self.random_seed = int(self.recipe_config["random_seed"])

    def _load_label_df(self):
        label_dataset_input_name = get_input_names_for_role('label_dataset')[0]
        self.label_dataset = dataiku.Dataset(label_dataset_input_name)
        renaming_mapping = {
            self.recipe_config["col_filename"]: constants.FILENAME,
            self.recipe_config["col_label"]: constants.LABEL
        }
        self.label_df = self.label_dataset.get_dataframe().rename(columns=renaming_mapping)[renaming_mapping.values()]
        self.labels = list(np.unique(self.label_df[constants.LABEL]))
        self.n_classes = len(self.labels)

    def _load_input(self):
        super(RetrainConfig, self)._load_input()
        self._load_label_df()
        utils.save_model_info(self.model_folder)
