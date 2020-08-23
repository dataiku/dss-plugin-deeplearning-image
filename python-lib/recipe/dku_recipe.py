import dku_deeplearning_image.utils as utils


class DkuRecipe(object):
    def __init__(self, config):
        self.config = config
        self.set_gpu_options()

    def set_gpu_options(self):
        utils.set_gpu_options(
            should_use_gpu=self.config.should_use_gpu,
            gpu_list=self.config.gpu_list,
            memory_limit=self.config.gpu_memory_limit)