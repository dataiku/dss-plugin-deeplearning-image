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


def load_logs_from_folder(folder_id):
    folder = dataiku.Folder(folder_id)
    try:
        for path in folder.list_paths_in_partition():
            relative_path = '.' + path
            os.makedirs(os.path.dirname(relative_path), exist_ok=True)
            file = folder.get_download_stream(path)
            with open(relative_path, 'wb+') as nf:
                nf.write(file.read())
        return os.path.join('.', TENSORBOARD_LOGS)
    except Exception:
        raise DataikuException('Folder with ID %s does not exist.' % str(folder_id))


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
    map(lambda x: x.define_flags(parser), loader_list)
    flags = parser.parse_args([])
    return flags


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
    raise TypeError("Not a TBLoader or TBPlugin subclass: %r" % (plugin_spec,))


def __get_tb_app(tensorboard_logs):
    plugins_or_loaders = tb_default.get_plugins()
    loaders = [make_plugin_loader(plugin_spec) for plugin_spec in plugins_or_loaders]
    flags = init_flags(loaders)
    flags.purge_orphaned_data = True
    flags.reload_interval = 5.0
    flags.logdir = tensorboard_logs
    return application.standard_tensorboard_wsgi(
        plugin_loaders=loaders,
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
