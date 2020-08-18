from .dku_config import DkuConfig


class RetrainConfig(DkuConfig):
    def __init__(self):
        self.name = 'retrain'
        self.output_role = 'model_output'
        super(RetrainConfig, self).__init__()

    def _load_recipe_param(self):
        super(RetrainConfig, self)._load_recipe_param()
        self.col_filename = self.recipe_config.get("col_filename")
        self.col_label = self.recipe_config.get("col_label")
        self.train_ratio = float(self.recipe_config.get("train_ratio"))
        self.input_shape = (int(self.recipe_config.get("image_width")), int(self.recipe_config.get("image_height")), 3)
        self.batch_size = int(self.recipe_config.get("batch_size"))
        self.model_pooling = self.recipe_config.get("model_pooling")
        self.model_reg = self.recipe_config.get("model_reg")
        self.model_dropout = float(self.recipe_config.get("model_dropout"))
        self.layer_to_retrain = self.recipe_config.get("layer_to_retrain")
        self.layer_to_retrain_n = int(self.recipe_config.get('layer_to_retrain_n', 1))
        self.optimizer = self.recipe_config.get("model_optimizer")
        self.learning_rate = self.recipe_config.get("model_learning_rate")
        self.custom_params_opti = self.recipe_config.get("model_custom_params_opti", [])
        self.nb_epochs = int(self.recipe_config.get("nb_epochs"))
        self.nb_steps_per_epoch = int(self.recipe_config.get("nb_steps_per_epoch"))
        self.nb_validation_steps = int(self.recipe_config.get("nb_validation_steps"))
        self.data_augmentation = self.recipe_config.get("data_augmentation")
        self.n_augmentation = int(self.recipe_config.get("n_augmentation"))
        self.custom_params_data_augment = self.recipe_config.get("model_custom_params_data_augmentation", [])
        self.use_tensorboard = self.recipe_config.get("tensorboard")
        self.random_seed = int(self.recipe_config.get("random_seed"))

    def _check_params(self):
        super(RetrainConfig, self)._check_params()
        assert self.col_filename, "You must provide a column name for image filename."
        assert self.col_label, "You must provide a column name for image label."
        assert self.layer_to_retrain_n, "You must retrain at least one layer."
        if self.n_augmentation and self.n_augmentation > self.batch_size:
            raise ValueError("The number of augmentations must be lower than the batch size. Aborting.")