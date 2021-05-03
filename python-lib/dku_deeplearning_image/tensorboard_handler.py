import dataiku
from dataiku.customwebapp import get_webapp_config
import tensorboard.default as tb_default
import tensorboard as tb
from tensorboard.backend import application
import os
import logging
from argparse import ArgumentParser
from tensorboard.plugins import base_plugin
from dataikuapi.utils import DataikuException

from dku_deeplearning_image.dku_constants import TENSORBOARD_LOGS


def check_folder_exists(folder):
    try:
        _ = folder.get_info()
    except Exception:
        raise DataikuException(f'Folder with ID {folder} does not exist.')


def is_filesystem_folder(folder):
    try:
        _ = folder.get_path()
        return True
    except Exception:
        return False


def download_logs_to_local(folder):
    for path in folder.list_paths_in_partition():
        relative_path = '.' + path
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        file = folder.get_download_stream(path)
        with open(relative_path, 'wb+') as nf:
            nf.write(file.read())


def __get_logs_path():
    retrained_model_folder = dataiku.Folder(get_webapp_config().get('retrained_model_folder'))
    check_folder_exists(retrained_model_folder)

    if not is_filesystem_folder(retrained_model_folder):
        download_logs_to_local(retrained_model_folder)
        folder_path = '.'
    else:
        folder_path = retrained_model_folder.get_path()
    return os.path.join(folder_path, TENSORBOARD_LOGS)


def __get_custom_assets_zip_provider():
    path = os.path.join(os.path.dirname(tb.__file__), 'webfiles.zip')
    if not os.path.exists(path):
        logging.warning('webfiles.zip static assets not found: %s', path)
        return None
    return lambda: open(path, "rb")


def make_plugin_loader(plugin_spec):
    """Returns a plugin loader for the given plugin.
    Args:
      plugin_spec: A TBPlugin subclass, or a TBLoader instance or subclass.
    Returns:
      A TBLoader for the given plugin.
    """
    if issubclass(plugin_spec, base_plugin.TBLoader):
        return plugin_spec()
    if issubclass(plugin_spec, base_plugin.TBPlugin):
        return base_plugin.BasicLoader(plugin_spec)
    raise TypeError(f"Not a TBLoader or TBPlugin subclass: {plugin_spec}")


def init_flags(plugin_list):
    parser = ArgumentParser()
    for plugin in plugin_list:
        plugin.define_flags(parser=parser)
    flags = parser.parse_args([])
    return flags


def __get_tb_app(tensorboard_logs):
    plugins = [make_plugin_loader(plugin) for plugin in tb_default.get_plugins()]
    flags = init_flags(plugins)
    flags.purge_orphaned_data = True
    flags.reload_interval = 5.0
    flags.logdir = tensorboard_logs
    return application.standard_tensorboard_wsgi(
        plugin_loaders=plugins,
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
        logs_path = __get_logs_path()
        if logs_path:
            return __get_tb_app(logs_path)(environ, start_response)
        else:
            status = '400'
            headers = [('Content-type', 'text/html; charset=utf-8')]
            start_response(status, headers)
            return [get_no_folder_selected_error()]


def get_no_folder_selected_error():
    return b"""
        <div style="text-align: center; margin-top: 20px;">
            <span style="color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
            position: relative;
            padding: .75rem 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid transparent;
            border-radius: .25rem;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            ">
                You must select a folder in the <b>Settings</b> tab before starting the webapp.
                Select a folder containing the tensorboard logs and <b>restart the webapp</b>.
            <span>
        </div>
    """
