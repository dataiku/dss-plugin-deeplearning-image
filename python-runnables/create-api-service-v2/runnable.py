import dataiku
from dataiku.runnables import Runnable
from dku_deeplearning_image.config_handler import create_dku_config
import dku_deeplearning_image.api_designer.api_designer_utils as utils
import dku_deeplearning_image.dku_constants as constants


class MyRunnable(Runnable):
    def __init__(self, project_key, config, plugin_config):
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.plugin = self.client.get_plugin(constants.PLUGIN_ID)

    def get_progress_target(self):
        return None

    def run(self, progress_callback):
        config = create_dku_config(self.config, constants.API_DESIGNER, self.project)
        model_folder_id = config.get("model_folder_id")
        endpoint_id = config.get("endpoint_id")
        service_id = config.get("service_id")
        utils.copy_plugin_to_dss_folder(self.plugin, config.get("model_folder_id"), self.project_key)
        api_service = utils.create_or_get_api_service(
            project=self.project,
            create_new_service=config.get("create_new_service"),
            service_id=service_id
        )
        code_env = utils.create_or_get_code_env(
            project_key=self.project_key,
            client=self.client,
            create_new_code_env=config.get("create_new_code_env"),
            env_name=config.get("code_env_name"),
            python_interpreter=config.get("python_interpreter"),
            custom_interpreter=config.get("custom_interpreter"))
        endpoint_settings = utils.build_model_endpoint_settings(
            plugin=self.plugin,
            project_key=self.project_key,
            endpoint_id=endpoint_id,
            code_env_name=code_env.env_name,
            model_folder_id=model_folder_id,
            max_nb_labels=config.get('max_nb_labels'),
            min_threshold=config.get('min_threshold'))
        utils.create_python_endpoint(api_service, endpoint_settings)
        return utils.get_html_result(self.project_key, model_folder_id, service_id, endpoint_id)
