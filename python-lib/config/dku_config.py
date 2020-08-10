import dku_deeplearning_image.utils as utils
from .dss_parameter import DSSParameter


class DkuConfig(object):
    def __init__(self, config):
        self.load(config)

    @utils.log_func(txt='config loading')
    def load(self, config):
        self._load_recipe_param(config)

    def _load_recipe_param(self, config):
        self.config = config
        should_use_gpu = self.config.get('should_use_gpu', False)
        gpu_usage = self.config.get('gpu_usage')
        gpu_list = self.config.get('gpu_list') if gpu_usage == 'custom' else []
        gpu_memory = self.config.get('gpu_memory')
        gpu_memory_limit = self.config.get('memory_limit') if gpu_memory == 'custom' else 0
        utils.set_gpu_options(
            should_use_gpu=should_use_gpu,
            gpu_list=gpu_list,
            memory_limit=gpu_memory_limit)
        self.add_param(name='use_gpu', value=self.config.get('should_use_gpu', False))

    def add_param(self, name, **dss_param_kwargs):
        setattr(self, name, DSSParameter(name=name, **dss_param_kwargs))

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getattribute__(self, item):
        attr = object.__getattribute__(self, item)
        return attr.value if isinstance(attr, DSSParameter) else attr
