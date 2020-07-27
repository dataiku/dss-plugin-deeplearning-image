from dataiku.customrecipe import get_recipe_config
import dku_deeplearning_image.utils as utils


class DkuConfig(object):
    def __init__(self):
        self.load()

    @utils.log_func(txt='config loading')
    def load(self):
        self._load_recipe_param()
        self._check_params()
        utils.display_gpu_device()

    def _load_recipe_param(self):
        self.recipe_config = get_recipe_config()
        should_use_gpu = self.recipe_config.get('should_use_gpu', False)
        self.gpu_options = utils.load_gpu_options(
            should_use_gpu, self.recipe_config.get('list_gpu'), self.recipe_config.get('gpu_allocation'))
        self.n_gpu = self.gpu_options.get("n_gpu", 0)
        self.use_gpu = should_use_gpu and self.n_gpu > 1

    def _check_params(self):
        pass

    def get(self, key, default=None):
        return getattr(self, key, default)
