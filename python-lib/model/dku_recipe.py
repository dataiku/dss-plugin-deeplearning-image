import dku_deeplearning_image.utils as utils
from utils.dku_model import DkuModel
import tensorflow as tf


class DkuRecipe(object):
    def __init__(self, config):
        self.config = config