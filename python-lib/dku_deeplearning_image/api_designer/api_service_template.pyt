import sys
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format="deeplearning-image-api-node \%(levelname)s - \%(message)s")

model_folder_path = folders[0]
python_lib_path = os.path.join(model_folder_path, '{python_lib_dir}')
sys.path.append(python_lib_path)

from dku_deeplearning_image.misc_objects import DkuModel
from dku_deeplearning_image.misc_objects import VirtualManagedFolder
import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.config_handler as create_dku_config

config = {{
    'max_nb_labels': {max_nb_labels},
    'min_threshold': {min_threshold}
}}


def get_model_folder(mf_path):
    return VirtualManagedFolder(mf_path)


def load_and_get_model(mf, conf):
    dku_model = DkuModel(mf)
    dku_model.load_model(conf, constants.GOAL.SCORE)
    return dku_model


def score_image(md, conf, img_b64):
    predictions = md.score_b64_image(
        img_b64,
        limit=conf['max_nb_labels'],
        min_threshold=conf['min_threshold'],
        classify=True)
    return predictions


dku_config = create_dku_config(config, constants.GOAL.SCORE)
model_folder = get_model_folder(model_folder_path)
model = load_and_get_model(model_folder, dku_config)


def api_py_function(img_b64):
    prediction_batch = score_image(model, config, img_b64)
    return prediction_batch['prediction'][0]
