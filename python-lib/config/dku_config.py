from dataiku.customrecipe import get_recipe_config
import dku_deeplearning_image.utils as utils
import time


class DkuConfig(object):
    def __init__(self):
        self.load()
        self.set_gpu_options()

    @utils.log_func(txt='config loading')
    def load(self):
        self._load_recipe_param()
        self._check_params()

    def _load_recipe_param(self):
        self.recipe_config = get_recipe_config()

    def set_gpu_options(self):
        self.should_use_gpu = self.recipe_config.get('should_use_gpu', False)
        gpu_usage = self.recipe_config.get('gpu_usage')
        self.gpu_list = self.recipe_config.get('gpu_list') if gpu_usage == 'custom' else []
        if gpu_usage == 'custom' and not self.gpu_list:
            raise ValueError('You have to select at least one GPU, or uncheck "Use GPU" checkbox.')
        gpu_memory = self.recipe_config.get('gpu_memory')
        self.gpu_memory_limit = self.recipe_config.get('memory_limit') if gpu_memory == 'custom' else 0
        self.use_gpu = self.should_use_gpu

    def get(self, key, default=None):
        return getattr(self, key, default)

    def _check_params(self):
        pass