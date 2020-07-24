# -*- coding: utf-8 -*-

import os
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import tensorflow as tf
import dku_deeplearning_image.constants as constants
from dku_deeplearning_image.applications import APPLICATIONS
import threading
import json
from collections import OrderedDict
import numpy as np
from tensorflow.python.client import device_lib
from datetime import datetime

import sys
import dku_deeplearning_image.config_utils as config_utils
from utils_objects.dku_application import DkuApplication
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Support Truncated Images with PIL
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

###################################################################################################################
###################################################################################################################
## NEW FILE
###################################################################################################################
###################################################################################################################


###################################################################################################################
## MODEL UTILS
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

def get_application(architecture):
    dku_application_params = list(filter(lambda x: x['name'] == architecture, APPLICATIONS))
    if not dku_application_params:
        available_apps = [x['name'] for x in APPLICATIONS]
        raise IOError("The application you asked for is not available. Available are : {}.".format(available_apps))
    return DkuApplication(**dku_application_params[0])

###################################################################################################################
## MODELS LIST
###################################################################################################################


# INFO : when adding a new architecture, you must add a select-option in python-runnables/dl-toolbox-download-models/runnable.json
#        with the label architecture_trainedon to make it available, along with new a constant in python-lib/constants.py


def is_keras_application(architecture):
    return architecture in build_keras_application().keys()


###############################################################
## EXTRACT INFO FROM MODEL (SUMMARY AND LAYERS)
###############################################################



def save_model_info(mf_path, dku_model):
    model_info = {
        constants.SCORING: dku_model.get_info(),
        constants.BEFORE_TRAIN: dku_model.get_info(base=True)
    }

    with mf_path.get_writer(constants.MODEL_INFO_FILE) as w:
        w.write(json.dumps(model_info))

# TODO: Rename this function as it has a lot of border effects
def load_gpu_options(should_use_gpu, list_gpu_str, gpu_allocation):
    gpu_options = {}
    log_info("load_gpu_options")
    if should_use_gpu:
        log_info("should use GPU")
        list_gpu = list(map(int, list_gpu_str.replace(" ", "").split(",")))
        gpu_options["list_gpu"] = list_gpu
        gpu_options["n_gpu"] = len(list_gpu)

        config_tf = tf.ConfigProto()
        # os.environ['CUDA_VISIBLE_DEVICES'] = list_gpu_str
        # print  os.environ['CUDA_VISIBLE_DEVICES']
        config_tf.gpu_options.per_process_gpu_memory_fraction = gpu_allocation
        # config_tf.log_device_placement=True
        session = tf.Session(config=config_tf)
        from keras.backend.tensorflow_backend import set_session
        set_session(session)
    else:
        config_utils.deactivate_gpu()

    return gpu_options


###################################################################################################################
## FILES LOGIC
###################################################################################################################

def get_weights_filename(with_top=False):
    return '{}{}.h5'.format(constants.WEIGHT_FILENAME, '' if with_top else '_notop')


def write_config(mf_path, config):
    with mf_path.get_writer(constants.CONFIG_FILE) as w:
        w.write(json.dumps(config))
    # config_path = get_file_path(mf_path, constants.CONFIG_FILE)
    # with open(config_path, 'w') as f:
    #   json.dump(config, f)


def get_file_path(folder_path, file_name):
    # Be careful to enforce that folder_path and file_name are actually strings
    return os.path.join(safe_str(folder_path), safe_str(file_name))


def safe_str(val):
    if sys.version_info > (3, 0):
        return str(val)
    else:
        if isinstance(val, unicode):
            return val.encode("utf-8")
        else:
            return str(val)


def get_cached_file_from_folder(folder, file_path):
    filename = file_path.replace('/', '_')
    if not (os.path.exists(filename)):
        with folder.get_download_stream(file_path) as stream:
            with open(filename, 'wb') as f:
                f.write(stream.read())
                log_info("cached file %s" % file_path)
    else:
        log_info("read from cache %s" % file_path)
    return filename


def get_model_config_from_file(model_folder):
    return json.loads(model_folder.get_download_stream(constants.CONFIG_FILE).read())


def build_prediction_output_df(images_paths, predictions):
    output = pd.DataFrame()
    output["images"] = images_paths
    log_info("------->" + str(output))
    output["prediction"] = predictions["prediction"]
    output["error"] = predictions["error"]
    return output


###################################################################################################################
## MISC.
###################################################################################################################
def log_func(txt):
    def inner(f):
        def wrapper(*args, **kwargs):
            logger.info('------ \n Info: Starting {} ({}) \n ------'.format(txt, datetime.now().strftime('%H:%M:%S')))
            res = f(*args, **kwargs)
            logger.info('------ \n Info: Ending {} ({}) \n ------'.format(txt, datetime.now().strftime('%H:%M:%S')))
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
    img = load_img(img_path, target_size=img_shape)
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
            raise IOError("The{} custom param #{} must have a 'name'".format(params_type, i))
        if not p.get("value", False):
            raise IOError("The{} custom param #{} must have a 'value'".format(params_type, i))
        name = p["name"]
        value = string_to_arg(p["value"])
        cleaned_params[name] = value
    return cleaned_params


def dbg_msg(msg, title=''):
    logger.debug('DEBUG : {}'.format(title).center(100, '-'))
    logger.debug(msg)
    logger.debug(''.center(100, '-'))


def display_gpu_device():
    log_info(device_lib.list_local_devices())
    if tf.test.gpu_device_name():
        log_info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        log_info("Please install GPU version of TF")

def log_info(*args):
    logger.info(*args)

###############################################################
## THREADSAFE GENERATOR / ITERATOR
## Inspired by :
##    https://github.com/fchollet/keras/issues/1638
##    http://anandology.com/blog/using-iterators-and-generators/
###############################################################

class ThreadsafeIterator(object):
    """Takes an iterator/generator and makes it thread-safe
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadsafeIterator(f(*a, **kw))

    return g



###############################################################
## MODEL CHECKPOINT FOR MULTI GPU
## When using multiple GPUs, we need to save the base recipe,
## not the one defined by multi_gpu_model
## see example: https://keras.io/utils/#multi_gpu_model
## Therefore, to save the recipe after each epoch by leveraging
## ModelCheckpoint callback, we need to adapt it to save the
## base recipe. To do so, we pass the base recipe to the callback.
## Inspired by:
##   https://github.com/keras-team/keras/issues/8463#issuecomment-345914612
###############################################################

class MultiGPUModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__(filepath,
                                                      monitor=monitor,
                                                      verbose=verbose,
                                                      save_best_only=save_best_only,
                                                      save_weights_only=save_weights_only,
                                                      mode=mode,
                                                      period=period)
        self.base_model = base_model

    def on_epoch_end(self, epoch, logs=None):
        # Must behave like ModelCheckpoint on_epoch_end but save base_model instead

        # First retrieve recipe
        model = self.model

        # Then switching recipe to base recipe
        self.model = self.base_model

        # Calling super on_epoch_end
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # Resetting recipe afterwards
        self.model = model
