from dataiku.runnables import Runnable
import dataiku
import json
import os
import tempfile
import time
import requests
import pandas as pd
import dku_deeplearning_image.dku_constants as constants
import dku_deeplearning_image.utils as utils
from dku_deeplearning_image.misc_objects import DkuModel
from dku_deeplearning_image.misc_objects import DkuFileManager

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


    def run(self, progress_callback):

        # Retrieving parameters
        output_managed_id = self.config.get('output_managed_folder')
        output_new_folder_name = self.config.get('output_new_folder_name', '')
        model_choice = self.config.get('model_choice')
        
        # Creating new Managed Folder if needed
        project = self.client.get_project(self.project_key)

        if output_new_folder_name and output_managed_id == "create_new_folder":
            output_folder_dss = project.create_managed_folder(output_new_folder_name)
        else:
            output_folder_dss = project.get_managed_folder(output_managed_id)

        output_folder = dataiku.Folder(output_folder_dss.get_definition()['name'], project_key=self.project_key)
        new_model = DkuModel(output_folder, is_empty=True)

        architecture, trained_on = model_choice.split('::')
        config = {
            "architecture": architecture,
            "trained_on": trained_on,
            "extract_layer_default_index": -2
        }

        new_model.set_config(config)

        output_folder_dss.put_file(constants.CONFIG_FILE, json.dumps(config))

        # Keras 3: Load with weights='imagenet' and save_weights() to convert to Keras 3 format.
        # This handles the internal h5 structure differences between legacy Google files and Keras 3.
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_callback(10)

            model_top = new_model.application.model_func(weights='imagenet', include_top=True)
            weights_top_path = os.path.join(tmpdir, utils.get_weights_filename(with_top=True))
            model_top.save_weights(weights_top_path)

            progress_callback(40)

            model_notop = new_model.application.model_func(weights='imagenet', include_top=False)
            weights_notop_path = os.path.join(tmpdir, utils.get_weights_filename(with_top=False))
            model_notop.save_weights(weights_notop_path)

            progress_callback(70)

            with open(weights_top_path, 'rb') as f:
                output_folder.upload_stream(utils.get_weights_filename(with_top=True), f)
            with open(weights_notop_path, 'rb') as f:
                output_folder.upload_stream(utils.get_weights_filename(with_top=False), f)


        progress_callback(80)

        if trained_on == constants.IMAGENET:
            response = requests.get(constants.IMAGENET_URL)
            mapping_df = pd.read_json(response.text, orient="index")
            mapping_df = mapping_df.reset_index()
            mapping_df = mapping_df.rename(columns={"index": "id", 1: "className"})[["id", "className"]]
            DkuFileManager.write_to_folder(
                folder=output_folder,
                file_path=constants.MODEL_LABELS_FILE,
                content=mapping_df.to_csv(index=False, sep=","))

        new_model.load_model({}, constants.GOAL.SCORE)
        new_model.save_info(output_folder)
        return "<span>DONE</span>"

