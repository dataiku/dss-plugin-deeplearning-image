import dku_deeplearning_image.utils as utils
from .dss_parameter import DSSParameter



class DkuConfig(object):
    def __init__(self, config):
        self.load(config)
        self.set_gpu_options()

    @utils.log_func(txt='config loading')
    def load(self, config):
        self._load_recipe_param(config)

    def _load_recipe_param(self, config):
        self.config = config

    def add_param(self, name, **dss_param_kwargs):
        setattr(self, name, DSSParameter(name=name, **dss_param_kwargs))

    def set_gpu_options(self):
        should_use_gpu = self.config.get('should_use_gpu', False)
        gpu_usage = self.config.get('gpu_usage')
        gpu_list = self.config.get('gpu_list') if gpu_usage == 'custom' else []
        if gpu_usage == 'custom' and not gpu_list:
            raise ValueError('You have to select at least one GPU, or uncheck "Use GPU" checkbox.')
        gpu_memory_allocation_mode = self.config.get('gpu_memory_allocation_mode')
        gpu_memory_limit = self.config.get('gpu_memory_limit')
        utils.set_gpu_options(
            should_use_gpu=should_use_gpu,
            gpu_list=gpu_list,
            gpu_memory_allocation_mode=gpu_memory_allocation_mode,
            memory_limit_ratio=gpu_memory_limit)
        self.add_param(name='should_use_gpu', value=should_use_gpu)
        self.add_param(name='gpu_list', value=gpu_list)
        self.add_param(name='gpu_usage', value=gpu_usage)
        self.add_param(name='gpu_memory_allocation_mode', value=gpu_memory_allocation_mode)
        self.add_param(name='gpu_memory_limit', value=gpu_memory_limit)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getattribute__(self, item):
        attr = object.__getattribute__(self, item)
        return attr.value if isinstance(attr, DSSParameter) else attr
