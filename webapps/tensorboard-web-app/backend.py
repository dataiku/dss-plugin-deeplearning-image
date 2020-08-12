import dataiku
from dataiku.customwebapp import get_webapp_config

dataiku.use_plugin_libs("deeplearning-image")
from dku_deeplearning_image.tensorboard_handle import start_server_and_return_url
from six.moves import urllib
import json
import os

###################################################################################################################
## VARIABLES THAT NEED TO BE SET
###################################################################################################################

# To work, your web-app requires to run on a code-env with the following libraries installed:
# tensorflow==2.2
# flask>=1.0,<1.1

model_folder_id = get_webapp_config().get('retrained_model_folder')

###################################################################################################################
## DEFINING AND LAUNCHING TENSORBOARD
###################################################################################################################

host = os.getenv('HOSTNAME')
server_url = start_server_and_return_url(model_folder_id, host)
server_url_parsed = urllib.parse.urlparse(server_url)
port = server_url_parsed.port


###################################################################################################################
## ROUTING
###################################################################################################################

@app.route('/tensorboard-endpoint')
def tensorboard_endpoint():
    url = "http://{}:{}/".format(host, port)
    return json.dumps({"tb_url": url})


@app.route('/data/<path:url>')
def proxy(url):
    redirect_url = "http://{}:{}/data/{}".format(host, port, url)
    return json.dumps({"tb_url": redirect_url})