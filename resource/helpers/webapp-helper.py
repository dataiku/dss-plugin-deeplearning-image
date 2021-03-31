import dataiku
from dku_deeplearning_image.dku_constants import TENSORBOARD_LOGS
import dku_deeplearning_image.utils as utils
import logging

logger = logging.getLogger(__name__)
api_client = dataiku.api_client()


def has_path(folder, path):
    try:
        folder_paths = folder.list_paths_in_partition()
    except Exception:
        logger.warning(f"The folder {folder.get_name()} could not be read. Skipping it.")
        return False
    return len([p for p in folder_paths if utils.sanitize_path(p).startswith(path)]) > 0


def do(payload, config, plugin_config, inputs):
    if payload.get('parameterName') == 'retrained_model_folder':
        project_key = dataiku.default_project_key()
        project_managed_folders = api_client.get_project(project_key).list_managed_folders()

        choices = [{
            'label': '{} ({})'.format(mf['name'], mf['type']),
            'value': mf['id']
        } for mf in project_managed_folders if has_path(dataiku.Folder(mf['id']), TENSORBOARD_LOGS)]
        assert choices, "No folders have a model with Tensorboard logs."
    else:
        choices = []
    return {"choices": choices}
