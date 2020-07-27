import dataiku
import dku_deeplearning_image.constants as constants
from dataikuapi.utils import DataikuException
import os
from werkzeug.serving import make_server
import tensorboard as tb
import logging

logger = logging.getLogger(__name__)


class TensorboardCustomServer(tb.program.TensorBoardServer):
    """
        Start a Tensorboard in a different server. Hack inspired by this GitHub post :
        https://stackoverflow.com/questions/61942505/add-tensorboard-server-to-flask-endpoint
    """
    def __init__(self, tensorboard_app, flags):
        self.app = tensorboard_app
        self.host = flags.host
        self.srv = make_server(self.host, 0, self.app)

    def get_port(self):
        return self.srv.server_port

    def serve_forever(self):
        self.srv.serve_forever()

    def get_url(self):
        return "http://%s:%s" % (self.host, self.get_port())

    def print_serving_message(self):
        pass  # Werkzeug's `serving.run_simple` handles this


def get_logdir(folder_name):
    # Retrieve model managed-folder path
    client = dataiku.api_client()
    project_key = os.environ["DKU_CURRENT_PROJECT_KEY"]
    project = client.get_project(project_key)
    filtered_folders = list(filter(lambda x: x['name'] == folder_name, project.list_managed_folders()))
    if not filtered_folders:
        raise DataikuException(
            "The folder '{}' (in project '{}' cannot be found".format(folder_name, project_key))
    folder_path = dataiku.Folder(filtered_folders[0]['id'], project_key=project_key).get_path()
    log_path = os.path.join(folder_path, constants.TENSORBOARD_LOGS)
    return log_path


def start_server_and_return_url(folder_name, host="127.0.0.1"):
    program = tb.program.TensorBoard(server_class=TensorboardCustomServer)
    program.configure(logdir=get_logdir(folder_name), host=host)
    return program.launch()
