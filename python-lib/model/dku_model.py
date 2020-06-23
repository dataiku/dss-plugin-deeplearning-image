import dl_image_toolbox_utils as utils
from keras.models import Model
import tensorflow as tf
import constants
from sklearn.model_selection import train_test_split
from keras.utils.training_utils import multi_gpu_model
from keras import optimizers


class DkuModel:
    def __init__(self, config):
        self.config = config

    def load(self, **kwargs):
        with tf.device('/cpu:0'):
            model_and_pp = utils.load_instantiate_keras_model_preprocessing(**kwargs)
        self.base_model = model_and_pp["model"]
        self.preprocessing = model_and_pp["preprocessing"]
        self.model_params = model_and_pp["model_params"]
        self.model_input_shape = utils.get_model_input_shape(
            model=self.model_tf,
            mf_path=self.config.model_folder
        )

    def get_name(self):
        return self.config.name


class ExtractModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)
        self.features = None
        self.model = None

    def load(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.SCORING
        )
        self.model = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.layers[self.config.extract_layer_index].output
        )

    def extract_features(self, images_folder):
        images_paths = images_folder.list_paths_in_partition()
        self.features = utils.score(
            dku_model=self,
            images_folder=images_folder,
            images_paths=images_paths
        )


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

    def classify(self, images_folder):
        images_paths = images_folder.list_paths_in_partition()
        self.predictions = utils.score(
            dku_model=self,
            images_folder=images_folder,
            images_paths=images_paths,
            limit=self.config.limit,
            min_threshold=self.config.min_threshold,
            labels_df=self.config.labels_df
        )


class RetrainModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None

    def _load_model_and_pp(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.RETRAINING,
            input_shape=self.config.input_shape,
            pooling=self.config.model_pooling,
            reg=self.config.model_reg,
            dropout=self.config.model_dropout,
            n_classes=self.config.n_classes)

    def _set_trainable_layers(self):
        print("Will Retrain layer(s) with mode: {}".format(self.config.layer_to_retrain))
        layers = self.model.layers
        if self.config.layer_to_retrain == "all":
            n_last = len(layers)
        elif self.config.layer_to_retrain == "last":
            n_last = 1
        elif self.config.layer_to_retrain == "n_last":
            n_last = self.config.layer_to_retrain_n
        else:
            n_last = 0

        for i, lay in enumerate(layers):
            lay.trainable = i >= (len(layers) - n_last)

    def _build_train_test_sets(self):
        train_df, test_df = train_test_split(
            self.config.label_df,
            stratify=self.config.label_df[constants.LABEL],
            train_size=self.config.train_ratio,
            random_state=self.config.random_seed)
        return train_df, test_df

    def _get_optimizer_class(self, optimizer):
        if optimizer == "adam":
            model_opti_class = optimizers.Adam
        elif optimizer == "adagrad":
            model_opti_class = optimizers.Adagrad
        elif optimizer == "sgd":
            model_opti_class = optimizers.SGD
        else:
            print("Optimizer not supporter: {}. Applying adam.".format(optimizer))
            model_opti_class = optimizers.Adam
        return model_opti_class

    def _get_model_checkpoint(self, model_weights_path, model_config, model_and_pp, use_gpu):
        should_save_weights_only = utils.should_save_weights_only(model_config)

        if use_gpu:
            mcheck = utils.MultiGPUModelCheckpoint(
                filepath=model_weights_path,
                base_model=model_and_pp['base_model'],
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=should_save_weights_only
            )
        else:
            mcheck = ModelCheckpoint(
                filepath=model_weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=should_save_weights_only
            )
        return mcheck

    def load(self):
        self._load_model_and_pp()
        self.model = multi_gpu_model(self.base_model, self.config.n_gpu) if self.config.use_gpu else self.base_model
        self._set_trainable_layers()
        self.model.summary()

    def compile(self, optimizer, custom_params_opti, learning_rate):
        model_opti_class = self._get_optimizer_class(optimizer)

        # Cleaning custom parameters
        params_opti = utils.clean_custom_params(custom_params_opti)
        params_opti["lr"] = learning_rate

        model_opti = model_opti_class(**params_opti)
        self.model.compile(optimizer=model_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    def retrain(self, images_folder):
        pass
