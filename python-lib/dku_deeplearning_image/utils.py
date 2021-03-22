import os
import tensorflow as tf

from keras.preprocessing.image import img_to_array, load_img
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout
from keras import regularizers

import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.keras_applications import APPLICATIONS
import threading
import json
from collections import OrderedDict
import numpy as np
from datetime import datetime
import GPUtil
import pandas as pd
from PIL import UnidentifiedImageError, ImageFile
import logging
import warnings


logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


###################################################################################################################
# MODEL UTILS
###################################################################################################################

def add_pooling(model_output, pooling):
    if pooling == 'avg':
        return GlobalAveragePooling2D()(model_output)
    elif pooling == 'max':
        return GlobalMaxPooling2D()(model_output)
    else:
        return Flatten()(model_output)


def add_dropout(model_output, dropout):
    return Dropout(dropout)(model_output) if dropout else model_output


def get_regularizer(reg):
    if reg:
        reg_l1, reg_l2 = reg["l1"], reg["l2"]
        if reg_l1 and reg_l2:
            return regularizers.l1_l2(**reg)
        elif reg_l2:
            return regularizers.l2(reg["l2"])
        elif reg_l1:
            return regularizers.l1(reg["l1"])
    return None


###################################################################################################################
# MODELS LIST
###################################################################################################################


# INFO : when adding a new architecture, you must add a select-option in
# python-runnables/dl-toolbox-download-models/runnable.json with the label architecture_trainedon to make it available,
# along with new a constant in python-lib/constants.py


def is_keras_application(architecture):
    return architecture in [app.name for app in APPLICATIONS]


###############################################################
# GPU HANDLING
###############################################################

def deactivate_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def can_use_gpu():
    return len(tf.config.list_physical_devices('GPU')) > 0


def set_gpu_options(should_use_gpu, gpu_list, gpu_memory_allocation_mode, memory_limit_ratio=None):
    if should_use_gpu and can_use_gpu():
        logger.info("Loading GPU Options...")
        gpus = tf.config.list_physical_devices('GPU')
        for i, g in enumerate(gpus):
            g.id = i
        gpus_to_use = [gpus[int(i)] for i in gpu_list] or gpus
        logger.info(f"GPUs on the machine: {[g.id for g in GPUtil.getGPUs()]}")
        logger.info(f"Will use the following GPUs: {gpus_to_use}")
        if gpu_memory_allocation_mode == constants.GPU_MEMORY_LIMIT and memory_limit_ratio:
            for gpu in gpus_to_use:
                memory_limit = calculate_gpu_memory_allocation(memory_limit_ratio, gpu)
                logger.info(f"Restraining GPU {gpu} to {memory_limit} Mo ({memory_limit_ratio}%)")
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit))]
                )
        elif gpu_memory_allocation_mode == constants.GPU_MEMORY_GROWTH:
            map(lambda g: tf.config.experimental.set_memory_growth(g, True), gpus_to_use)
        tf.config.set_visible_devices(gpus_to_use, 'GPU')
    else:
        logger.info("Skipping GPU Options")
        deactivate_gpu()


def get_tf_strategy():
    return tf.distribute.MirroredStrategy()


def calculate_gpu_memory_allocation(memory_limit_ratio, gpu_to_use):
    gpu = [gpu for gpu in GPUtil.getGPUs() if gpu.id == gpu_to_use.id][0]
    return int((memory_limit_ratio / 100) * gpu.memoryTotal)

###################################################################################################################
# FILES LOGIC
###################################################################################################################


def get_weights_filename(with_top=False):
    return '{}{}.h5'.format(constants.WEIGHT_FILENAME, '' if with_top else constants.NOTOP_SUFFIX)


def get_file_path(folder_path, file_name):
    # Be careful to enforce that folder_path and file_name are actually strings
    return os.path.join(str(folder_path), str(file_name))


def get_cached_file_from_folder(folder, file_path, verbose=False):
    filename = file_path.replace('/', '_')
    if not (os.path.exists(filename)):
        with folder.get_download_stream(file_path) as stream:
            with open(filename, 'wb') as f:
                f.write(stream.read())
                if verbose:
                    logger.info(f"cached file {file_path}")
    else:
        if verbose:
            logger.info(f"read from cache {file_path}")
    return filename


def get_model_config_from_file(model_folder):
    return json.loads(model_folder.get_download_stream(constants.CONFIG_FILE).read())


def build_prediction_output_df(images_paths, predictions):
    output = pd.DataFrame()
    output["images"] = images_paths
    logger.info("------->" + str(output))
    output["prediction"] = predictions["prediction"]
    output["error"] = predictions["error"]
    return output


###################################################################################################################
# MISC.
###################################################################################################################
def log_func(txt):
    def inner(f):
        def wrapper(*args, **kwargs):
            logger.info(f"------ \n Info: Starting {txt} ({datetime.now().strftime('%H:%M:%S')}) \n ------")
            res = f(*args, **kwargs)
            logger.info(f"------ \n Info: Ending {txt} ({datetime.now().strftime('%H:%M:%S')}) \n ------")
            return res

        return wrapper

    return inner


def format_predictions_output(predictions, classify=False, labels_df=None, limit=None, min_threshold=None):
    if not classify:
        return predictions.tolist()
    formatted_predictions = []
    id_pred = lambda index: labels_df.loc[index][constants.LABEL] if labels_df is not None else str(index)
    for pred in predictions:
        formatted_pred = get_ordered_dict(
            {id_pred(i): float(pred[i]) for i in pred.argsort()[-limit:] if float(pred[i]) >= min_threshold})
        formatted_predictions.append(formatted_pred)
    return formatted_predictions


def get_predictions(model, batch, classify=False, limit=constants.DEFAULT_PRED_LIMIT, min_threshold=0, labels_df=None):
    predictions = model.predict(batch)
    return format_predictions_output(predictions, classify, labels_df, limit, min_threshold)


def get_ordered_dict(predictions):
    return json.dumps(OrderedDict(sorted(predictions.items(), key=(lambda x: -x[1]))))


def preprocess_img(img_path, img_shape, preprocessing):
    try:
        img = load_img(img_path, target_size=img_shape)
    except UnidentifiedImageError as err:
        logger.warning(f'The file {img_path} is not a valid image. skipping it. Error: {err}')
        return
    array = img_to_array(img)
    array = preprocessing(array)
    return array


def clean_custom_params(custom_params, params_type=""):
    def string_to_arg(string):
        if string.lower() == "true":
            res = True
        elif string.lower() == "false":
            res = False
        else:
            try:
                res = np.float(string)
            except ValueError:
                res = string
        return res

    cleaned_params = {}
    params_type = " '{}'".format(params_type) if params_type else ""
    for i, p in enumerate(custom_params):
        if not p.get("name", False):
            raise IOError(f"The {params_type} custom param #{i} must have a 'name'")
        if not p.get("value", False):
            raise IOError(f"The {params_type} custom param #{i} must have a 'value'")
        cleaned_params[p["name"]] = string_to_arg(p["value"])
    return cleaned_params


def sanitize_path(path):
    return path[1:] if path.startswith('/') else path


def is_path_in_folder(path, folder):
    return sanitize_path(path) in [sanitize_path(p) for p in folder.list_paths_in_partition()]


def dbg_msg(msg, title=''):
    logger.debug('DEBUG : {}'.format(title).center(100, '-'))
    logger.debug(msg)
    logger.debug(''.center(100, '-'))


###############################################################
# THREADSAFE GENERATOR / ITERATOR
# Inspired by :
#    https://github.com/fchollet/keras/issues/1638
#    http://anandology.com/blog/using-iterators-and-generators/
###############################################################


''' Make the generators threadsafe in case of multiple threads '''


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
