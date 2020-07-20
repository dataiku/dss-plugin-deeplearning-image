from .dku_config import DkuConfig


class RetrainConfig(DkuConfig):
    def __init__(self):
        self.name = 'retrain'
        self.output_role = 'model_output'
        super(RetrainConfig, self).__init__()

    def _load_recipe_param(self):
        super(RetrainConfig, self)._load_recipe_param()

        self.col_filename = self.recipe_config["col_filename"]
        self.col_label = self.recipe_config["col_label"]
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
        if self.n_augmentation and self.n_augmentation > self.batch_size:
            raise ValueError("The number of augmentations must be lower than the batch size. Aborting.")
        self.custom_params_data_augment = self.recipe_config.get("model_custom_params_data_augmentation", [])
        self.use_tensorboard = self.recipe_config["tensorboard"]
        self.random_seed = int(self.recipe_config["random_seed"])
