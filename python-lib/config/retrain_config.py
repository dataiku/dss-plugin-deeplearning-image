from .dku_config import DkuConfig


class RetrainConfig(DkuConfig):
    def __init__(self, config):
        self.name = 'retrain'
        self.output_role = 'model_output'
        super(RetrainConfig, self).__init__(config)

    def _load_recipe_param(self, config):
        super(RetrainConfig, self)._load_recipe_param(config)
        self.add_param(name='col_filename', value=self.config.get("col_filename"), required=True)
        self.add_param(name='col_label', value=self.config.get("col_label"), required=True)
        self.add_param(name='train_ratio', value=float(self.config.get("train_ratio")))
        self.add_param(
            name='input_shape',
            value=(int(self.config.get("image_width")), int(self.config.get("image_height")), 3))
        self.add_param(name='batch_size', value=int(self.config.get("batch_size")))
        self.add_param(name='model_pooling', value=self.config.get("model_pooling"))
        self.add_param(name='model_reg', value=self.config.get("model_reg"))
        self.add_param(name='model_dropout', value=float(self.config.get("model_dropout")))
        self.add_param(name='layer_to_retrain', value=self.config.get("layer_to_retrain"))
        self.add_param(name='layer_to_retrain_n', value=int(self.config.get("layer_to_retrain_n")), required=True)
        self.add_param(name='optimizer', value=self.config.get("model_optimizer"))
        self.add_param(name='learning_rate', value=self.config.get("model_learning_rate"))
        self.add_param(name='custom_params_opti', value=self.config.get("model_custom_params_opti"))
        self.add_param(name='nb_epochs', value=int(self.config.get("nb_epochs")))
        self.add_param(name='nb_steps_per_epoch', value=int(self.config.get("nb_steps_per_epoch")))
        self.add_param(name='nb_validation_steps', value=int(self.config.get("nb_validation_steps")))
        self.add_param(name='data_augmentation', value=self.config.get("data_augmentation"))
        n_augmentation = int(self.config.get("n_augmentation")) if self.data_augmentation else 0
        self.add_param(
            name='n_augmentation',
            value=n_augmentation,
            checks=[
                {
                    'type': 'custom',
                    'cond': not n_augmentation or (n_augmentation <= self.batch_size),
                    'err_msg': "The number of augmentations must be lower than the batch size. Aborting."
                }
            ]
        )
        self.add_param(name='custom_params_data_augment', value=self.get("model_custom_params_data_augmentation", []))
        self.add_param(name='use_tensorboard', value=self.config.get("tensorboard"))
        self.add_param(name='random_seed', value=int(self.config.get("random_seed")))