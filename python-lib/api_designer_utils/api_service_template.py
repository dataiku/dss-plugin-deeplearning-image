from ast import literal_eval
import sys
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format="deeplearning-image-api-node \%(levelname)s - \%(message)s")

model_folder_path = folders[0]
# python_lib_path = os.path.join(model_folder_path, {python_lib_dir})
python_lib_path = os.path.join(model_folder_path, 'api_deployer/python_lib')

#Load plugin libs
sys.path.append(python_lib_path)

from config import ScoreConfig
from utils_objects import DkuModel
from utils_objects import VirtualManagedFolder
import dku_deeplearning_image.constants as constants

CONFIGS = {
    'max_nb_labels': 2,
    'min_threshold': 0.1
}

def get_model_folder(model_folder_path):
    return VirtualManagedFolder(model_folder_path)

def load_and_get_model(model_folder, config):
    dku_model = DkuModel(model_folder)
    dku_model.load_model(config, constants.SCORING)
    return dku_model

def score_image(model, config, img_b64):
    predictions = model.score_b64_image(img_b64, limit=config['max_nb_labels'], min_threshold=config['min_threshold'], classify=True)
    return predictions

config = ScoreConfig(CONFIGS)
model_folder = get_model_folder(model_folder_path)
model = load_and_get_model(model_folder, config)

def api_py_function(img_b64):
    prediction_batch = score_image(model, CONFIGS, img_b64)
    return literal_eval(prediction_batch[0])
