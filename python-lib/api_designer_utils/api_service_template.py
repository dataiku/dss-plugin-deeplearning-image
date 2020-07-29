import dataiku
import pandas as pd
from dataiku.customrecipe import *
import numpy as np
import json
import os
import glob
from io import BytesIO
from ast import literal_eval
import base64
import sys
import logging
from recipe import ScoreRecipe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format="deeplearning-image-api-node \%(levelname)s - \%(message)s")

model_folder_path = folders[0]

#Load plugin libs
sys.path.append(os.path.join(model_folder_path, "python-lib"))
import dl_tool_box_os as utils
import constants


max_nb_labels = {max_nb_labels}
min_threshold = {min_threshold}


# Model
model_and_pp = utils.load_instantiate_keras_model_preprocessing(model_folder_path, goal=constants.SCORING)
model = model_and_pp["model"]
preprocessing = model_and_pp["preprocessing"]
model_input_shape = utils.get_model_input_shape(model, model_folder_path)

# (classId -> Name) mapping
labels_df = None
labels_path = utils.get_file_path(model_folder_path, constants.MODEL_LABELS_FILE)
if os.path.isfile(labels_path):
    labels_df = pd.read_csv(labels_path, sep=",")
    labels_df = labels_df.set_index('id')
else:
    logger.info("No csv file in the model folder, will not use class names.")


def api_py_function(img_b64):
    #takes in input the image encoded as base64 base64.b64encode(open(img_path, "rb").read())
    #preprocess the image and score it

    logger.info("Start loading image")
    img_b64_decode = base64.b64decode(img_b64)
    img = BytesIO(img_b64_decode)
    logger.info("Finished loading image")

    logger.info("Start preprocessing image")
    preprocessed_img = utils.preprocess_img(img, model_input_shape, preprocessing)
    logger.info("Finished preprocessing image")
    batch_im = np.expand_dims(preprocessed_img, 0)

    logger.info("Start predicting")
    prediction_batch = utils.get_predictions(model, batch_im, max_nb_labels, min_threshold, labels_df)
    logger.info("Finished predicting")

    return literal_eval(prediction_batch[0])