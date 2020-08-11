import dataiku
from dataiku.runnables import Runnable
import dku_deeplearning_image.constants as constants
import api_designer_utils.utils as utils

class MyRunnable(Runnable):
    """The base interface for a Python runnable"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        
    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of 
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return None

    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """
        client = dataiku.api_client()
        plugin = client.get_plugin(constants.PLUGIN_ID)
        python_interpreter = self.config.get('python_interpreter')
        custom_interpreter = self.config.get('custom_interpreter') if python_interpreter == 'CUSTOM' else ''
        utils.create_api_code_env(plugin, client, constants.ENV_NAME, python_interpreter, custom_interpreter)
        return utils.get_codeenv_output_msg(plugin, constants.ENV_NAME)
