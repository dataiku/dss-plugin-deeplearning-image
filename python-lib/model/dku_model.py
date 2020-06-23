import dl_image_toolbox_utils as utils
from keras.models import Model
import tensorflow as tf
import constants


class DkuModel:
    def __init__(self, config):
        self.config = config

    def load(self, **kwargs):
        model_and_pp = utils.load_instantiate_keras_model_preprocessing(**kwargs)
        self.base_model = model_and_pp["model"]
        self.preprocessing = model_and_pp["preprocessing"]
        self.model_params = model_and_pp["model_params"]
        self.model_input_shape = utils.get_model_input_shape(
            model=self.model_tf,
            mf_path=self.config.model_folder
        )

class ExtractModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)

    def load(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.SCORING
        )
        self.model = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.layers[self.config.extract_layer_index].output
        )

class ScoreModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)

    def load(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.SCORING
        )


class RetrainModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)

    def load(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.SCORING
        )

    def get_model_and_pp(self):
        model_and_pp = utils.load_instantiate_keras_model_preprocessing(
            config.model_folder,
            goal=constants.RETRAINING,
            input_shape=config.input_shape,
            pooling=config.model_pooling,
            reg=config.model_reg,
            dropout=config.model_dropout,
            n_classes=config.n_classes)
        return model_and_pp

    def set_trainable_layers(self, layer_to_retrain, layer_to_retrain_n=None):
        # CHOOSING LAYER TO RETRAIN
        print("Will Retrain layer(s) with mode: {}".format(layer_to_retrain))
        if layer_to_retrain == "all":
            for lay in model.layers:
                lay.trainable = True

        elif layer_to_retrain == "last":
            for lay in model.layers[:-1]:
                lay.trainable = False
            lay = model.layers[-1]
            lay.trainable = True

        elif layer_to_retrain == "n_last":
            n_last = layer_to_retrain_n
            for lay in model.layers[:-n_last]:
                lay.trainable = False
            for lay in model.layers[-n_last:]:
                lay.trainable = True

    def _load_model_with_gpu(self):
        with tf.device('/cpu:0'):
            config.model_and_pp = get_model_and_pp(config)
        config.model_and_pp['base_model'] = config.model_and_pp['model']
        config.model_and_pp['model'] = multi_gpu_model(config.model_and_pp['base_model'], config.n_gpu)

    def load_model_without_gpu(self):
        config.model_and_pp = get_model_and_pp(config)

    def load_model(self):
        load_model_with_gpu(config) if use_gpu else load_model_without_gpu(config)
        set_trainable_layers(config.model_and_pp['model'], config.layer_to_retrain, config.layer_to_retrain_n)
        config.model_and_pp['model'].summary()