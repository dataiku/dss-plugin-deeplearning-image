from dataiku.customrecipe import get_recipe_config
import dku_deeplearning_image.utils as utils
import time

class DkuConfig(object):
    def __init__(self):
        self.load()

    @utils.log_func(txt='config loading')
    def load(self):
        self._load_recipe_param()
        utils.display_gpu_device()

    def _load_recipe_param(self):
        self.recipe_config = get_recipe_config()
        should_use_gpu = self.recipe_config.get('should_use_gpu', False)
        gpu_usage = self.recipe_config.get('gpu_usage')
        gpu_list = self.recipe_config.get('gpu_list') if gpu_usage == 'custom' else []
        gpu_memory = self.recipe_config.get('gpu_memory')
        gpu_memory_limit = self.recipe_config.get('memory_limit') if gpu_memory == 'custom' else 0
        utils.set_gpu_options(
            should_use_gpu=should_use_gpu,
            gpu_list=gpu_list,
            memory_limit=gpu_memory_limit)
        self.use_gpu = should_use_gpu

    def get(self, key, default=None):
        return getattr(self, key, default)
