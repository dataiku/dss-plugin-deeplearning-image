# This file is the actual code for the Python runnable create_api_dl_codeenv
from dataiku.runnables import Runnable
import constants
import dataiku 
from api_designer_utils import *

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
        use_gpu = self.config.get("use_gpu")
        code_env_name = constants.ENV_NAME_CPU
        if use_gpu : 
            code_env_name = constants.ENV_NAME_GPU
            
        client = dataiku.api_client()
        create_api_code_env(client, code_env_name, use_gpu)
        
        return "code env build is OK"
        