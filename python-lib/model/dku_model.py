import dku_deeplearning_image.utils as utils
import tensorflow as tf


class DkuModel(object):
    def __init__(self, config):
        self.config = config

    def load(self, **kwargs):
        with tf.device('/cpu:0'):
            model_and_pp = utils.load_instantiate_keras_model_preprocessing(**kwargs)
        self.base_model = model_and_pp["model"]
        self.preprocessing = model_and_pp["preprocessing"]
        self.model_params = model_and_pp["model_params"]
        self.model_input_shape = utils.get_model_input_shape(
            model=self.base_model,
            mf_path=self.config.model_folder
        )

    def get_name(self):
        return self.config.name