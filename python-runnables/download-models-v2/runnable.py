from dataiku.runnables import Runnable
import dataiku
import requests
import json
import pandas as pd
import dku_deeplearning_image.constants as constants
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


    def run(self, progress_callback):

        # Retrieving parameters
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
        new_model = DkuModel(output_folder, is_empty=True)

        architecture, trained_on = model_choice.split('::')
        config = {
            "architecture": architecture,
            "trained_on": trained_on,
            "extract_layer_default_index": -2
        }

        new_model.set_config(config)

        # Downloading weights
        url_to_weights = new_model.get_weights_url()

        def update_percent(percent, last_update_time):
            new_time = time.time()
            if (new_time - last_update_time) > 3:
                progress_callback(percent)
                return new_time
            else:
                return last_update_time

        def download_files_to_managed_folder(output_f, files_info, chunk_size=8192):
            total_size = 0
            bytes_so_far = 0
            for file_info in files_info:
                response = requests.get(file_info["url"], stream=True)
                total_size += int(response.headers.get('content-length'))
                file_info["response"] = response
            update_time = time.time()
            for file_info in files_info:
                with output_f.get_writer(file_info["filename"]) as f:
                    for content in file_info["response"].iter_content(chunk_size=chunk_size):
                        bytes_so_far += len(content)
                        # Only scale to 80% because needs to compute model summary after download
                        percent = int(float(bytes_so_far) / total_size * 80)
                        update_time = update_percent(percent, update_time)
                        f.write(content)

        if trained_on == constants.IMAGENET:
            # Downloading mapping id <-> name for imagenet classes
            # File used by Keras in all its 'decode_predictions' methods
            # Found here : https://github.com/keras-team/keras/blob/2.1.1/keras/applications/imagenet_utils.py
            class_mapping_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        else:
            class_mapping_url = ''

        files_to_dl = [
            {"url": url_to_weights["top"], "filename": new_model.get_weights_path(with_top=True)},
            {"url": url_to_weights["no_top"], "filename": new_model.get_weights_path(with_top=False)}
        ]

        if class_mapping_url:
            files_to_dl.append({"url": class_mapping_url, "filename": constants.CLASSES_MAPPING_FILE})

        output_folder_dss.put_file(constants.CONFIG_FILE, json.dumps(config))
        download_files_to_managed_folder(output_folder, files_to_dl)

        if class_mapping_url:
            mapping_df = pd.read_json(output_folder.get_download_stream(constants.CLASSES_MAPPING_FILE), orient="index")
            mapping_df = mapping_df.reset_index()
            mapping_df = mapping_df.rename(columns={"index": "id", 1: "className"})[["id", "className"]]
            DkuFileManager.write_to_folder(
                folder=output_folder,
                file_path=constants.MODEL_LABELS_FILE,
                content=mapping_df.to_csv(index=False, sep=","))
            output_folder_dss.delete_file(constants.CLASSES_MAPPING_FILE)
        
        return "<span>DONE</span>"

