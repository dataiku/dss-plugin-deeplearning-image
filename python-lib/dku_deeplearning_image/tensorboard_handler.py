import dataiku
from dataiku.customwebapp import get_webapp_config
import tensorboard.default as tb_default
import tensorboard as tb
from tensorboard.backend import application
import os
import logging
from argparse import ArgumentParser
from dataikuapi.utils import DataikuException

from dku_deeplearning_image.dku_constants import TENSORBOARD_LOGS


def load_logs_from_folder(folder_id):
    folder = dataiku.Folder(folder_id)
    try:
        _ = folder.get_info()
    except Exception:
        raise DataikuException(f'Folder with ID {folder_id} does not exist.')

    try:
        folder_path = folder.get_path()
    except Exception as err:
        if err.args[0].startswith('Folder is not on the local filesystem'):
            for path in folder.list_paths_in_partition():
                relative_path = '.' + path
                os.makedirs(os.path.dirname(relative_path), exist_ok=True)
                file = folder.get_download_stream(path)
                with open(relative_path, 'wb+') as nf:
                    nf.write(file.read())
            folder_path = '.'
        else:
            raise err
    return os.path.join(folder_path, TENSORBOARD_LOGS)


def __get_logs_path():
    return load_logs_from_folder(get_webapp_config().get('retrained_model_folder'))


def __get_custom_assets_zip_provider():
    path = os.path.join(os.path.dirname(tb.__file__), 'webfiles.zip')
    if not os.path.exists(path):
        logging.warning('webfiles.zip static assets not found: %s', path)
        return None
    return lambda: open(path, "rb")


def init_flags(loader_list):
    parser = ArgumentParser()
    for x in loader_list:
        x.define_flags(parser)
    flags = parser.parse_args([])
    return flags


def __get_tb_app(tensorboard_logs):
    plugins_or_loaders = tb_default.get_plugins()
    flags = init_flags(plugins_or_loaders)
    flags.purge_orphaned_data = True
    flags.reload_interval = 5.0
    flags.logdir = tensorboard_logs
    return application.standard_tensorboard_wsgi(
        plugin_loaders=plugins_or_loaders,
        assets_zip_provider=__get_custom_assets_zip_provider(),
        flags=flags
    )


# This is the hack found to serve the Tensorboard app in Flask. We will replace the Flask app
# by the WSGI app function beneath that are the same objects.
def tensorboard_wsgi_app(environ, start_response):
    if environ['PATH_INFO'] == '/__ping':
        # We need to define a new endpoint, used by DSS to check whether the backend has started successfully.
        status = '200 OK'
        headers = [('Content-type', 'text/plain; charset=utf-8')]
        start_response(status, headers)
        return [b"200"]
    else:
        return __get_tb_app(__get_logs_path())(environ, start_response)
