from dataiku.runnables import Runnable
import dataiku
import requests
import json
import pandas as pd
import dku_deeplearning_image.constants as constants
import dku_deeplearning_image.utils as utils
from utils_objects import DkuModel
from utils_objects import DkuFileManager
import time

# We deactivate GPU for this script, because all the methods only need to 
# fetch information about model and do not make computation

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
        self.client = dataiku.api_client()

        
    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of 
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return (100, 'NONE')

    def create_base_model(self):
        output_managed_id = self.config.get('output_managed_folder')
        output_new_folder_name = self.config.get('output_new_folder_name', '')
        model_choice = self.config.get('model_choice')

        # Creating new Managed Folder if needed
        project = self.client.get_project(self.project_key)

        if output_new_folder_name:
            output_folder_dss = project.create_managed_folder(output_new_folder_name)
        else:
            output_folder_dss = project.get_managed_folder(output_managed_id)

        output_folder = dataiku.Folder(output_folder_dss.get_definition()['name'], project_key=self.project_key)

        architecture, trained_on = model_choice.split('::')
        config = {
            "architecture": architecture,
            "trained_on": trained_on,
            "extract_layer_default_index": -2
        }

        new_model = DkuModel(output_folder, is_empty=True)
        new_model.set_config(config)

        return new_model

    def run(self, progress_callback):

        # Retrieving parameters
        new_model = self.create_base_model()
        new_model.download_from_web(cb=progress_callback)

        return "<span>DONE</span>"

