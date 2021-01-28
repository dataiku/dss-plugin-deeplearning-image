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

from dku_deeplearning_image.constants import TENSORBOARD_LOGS


def get_logdir(folder_id):
    folder = dataiku.Folder(folder_id)
    try:
        folder_path = folder.get_path()
        return os.path.join(folder_path, TENSORBOARD_LOGS)
    except Exception as err:
        raise DataikuException('Folder with ID %s does not exist.' % str(folder_id))


def __get_logs_path():
    return get_logdir(get_webapp_config().get('retrained_model_folder'))


def __get_custom_assets_zip_provider():
    path = os.path.join(os.path.dirname(tb.__file__), 'webfiles.zip')
    if not os.path.exists(path):
        logging.warning('webfiles.zip static assets not found: %s', path)
        return None
    return lambda: open(path, "rb")


def init_flags(loader_list):
    parser = ArgumentParser()
    for loader in loader_list:
        loader.define_flags(parser)
    flags = parser.parse_args([])
    return flags


def make_plugin_loader(plugin_spec):
    """Returns a plugin loader for the given plugin.
    Args:
      plugin_spec: A TBPlugin subclass, or a TBLoader instance or subclass.
    Returns:
      A TBLoader for the given plugin.
    """
    if isinstance(plugin_spec, base_plugin.TBLoader):
        print("case1")
        return plugin_spec
    if isinstance(plugin_spec, type):
        if issubclass(plugin_spec, base_plugin.TBLoader):
            print("case2")
            return plugin_spec()
        if issubclass(plugin_spec, base_plugin.TBPlugin):
            print("case3")
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


def tensorboard_wsgi_app(environ, start_response):
    if environ['PATH_INFO'] == '/__ping':
        status = '200 OK'
        headers = [('Content-type', 'text/plain; charset=utf-8')]
        start_response(status, headers)
        return [b"200"]
    else:
        return __get_tb_app(__get_logs_path())(environ, start_response)

