# -*- coding: utf-8 -*-

import os
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import tensorflow as tf
import dku_deeplearning_image.constants as constants
import threading
import json
from collections import OrderedDict
import StringIO
import numpy as np
from tensorflow.python.client import device_lib
from datetime import datetime

import sys
import tables #to get the h5 file stream from the folder API as a file to be read by the keras API
import dku_deeplearning_image.config_utils as config_utils
import pandas as pd

# Support Truncated Images with PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

###################################################################################################################
## MODELS LIST
###################################################################################################################

from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocessing
from keras.applications.xception import Xception, preprocess_input as xception_preprocessing
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocessing
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocessing


# INFO : when adding a new architecture, you must add a select-option in python-runnables/dl-toolbox-download-models/runnable.json
#        with the label architecture_trainedon to make it available, along with new a constant in python-lib/constants.py
def build_keras_application():
    from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocessing
    from keras.applications.xception import Xception, preprocess_input as xception_preprocessing
    from keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocessing
    from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocessing

    keras_applications = {
        constants.RESNET: {
            "model_func": ResNet50, 
            "preprocessing": resnet50_preprocessing,
            "input_shape": (224, 224),
            "weights": {
                constants.IMAGENET: {
                    "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                    "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
                }
            }
        },
        constants.XCEPTION: {
            "model_func": Xception,
            "preprocessing": xception_preprocessing,
            "input_shape": (299, 299),
            "weights": {
                constants.IMAGENET: {
                    "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5",
                    "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                }
            }
        },
        constants.INCEPTIONV3: {
            "model_func": InceptionV3,
            "preprocessing": inceptionv3_preprocessing,
            "input_shape": (299, 299),
            "weights": {
                constants.IMAGENET: {
                    "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                    "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
                }
            }
        },
        constants.VGG16: {
            "model_func": VGG16,
            "preprocessing": vgg16_preprocessing,
            "input_shape": (224, 224),
            "weights": {
                constants.IMAGENET: {
                    "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                    "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                }
            }
        }
    }
    return keras_applications

def is_keras_application(architecture):
    return architecture in build_keras_application().keys()

def get_extract_layer_index(architecture, trained_on):
    # May be more complicated as the list of models grows
    return -2


def get_weights_urls(architecture, trained_on):
    if is_keras_application(architecture):
        return build_keras_application()[architecture]["weights"][trained_on]
    else:
        return {}

def should_save_weights_only(config):
    return config["architecture"] in build_keras_application().keys()

###############################################################
## EXTRACT INFO FROM MODEL (SUMMARY AND LAYERS)
###############################################################

def get_model_input_shape(model, mf_path):

    input_shape = model.input_shape[1:3]

    # Check that model has an actual input shape
    if input_shape[0] == None or input_shape[1] == None:
        
        config = config_utils.get_config(mf_path)
        architecture = config["architecture"]

        if not is_keras_application(architecture):
            raise IOError("You must provide an input shape for your architecture '{}'".format(architecture))

        return build_keras_application()[architecture].get("input_shape", (224, 224))

    else:
        return input_shape


def get_layers_as_list(model):
    layers = model.layers
    return [layer.__class__.__name__ for layer in layers]

def get_model_summary(model):
    summary_io = StringIO.StringIO()
    model.summary(print_fn=lambda line: summary_io.write(line + "\n"))
    return summary_io.getvalue()

def save_model_info(mf_path):
    model_info = {}

    # For SCORING
    model_info[constants.SCORING] = compute_model_info(mf_path, constants.SCORING)

    # For BEFORE_TRAIN
    model_info[constants.BEFORE_TRAIN] = compute_model_info(mf_path, constants.BEFORE_TRAIN)
    
    with mf_path.get_writer(constants.MODEL_INFO_FILE) as w:
        w.write(json.dumps(model_info))


def compute_model_info(mf_path, goal):
    model_info = {}

    if goal == constants.SCORING:
        
        model_and_pp_scoring = load_instantiate_keras_model_preprocessing(mf_path, goal=constants.SCORING, verbose=False)
        layers_scoring = get_layers_as_list(model_and_pp_scoring["model"])
        summary_scoring = get_model_summary(model_and_pp_scoring["model"])

        model_info = {
            "layers": layers_scoring,
            "summary": summary_scoring
        }

    elif goal == constants.BEFORE_TRAIN:

        model_and_pp_bt = load_instantiate_keras_model_preprocessing(mf_path, goal=constants.BEFORE_TRAIN, verbose=False)
        summary_bt = get_model_summary(model_and_pp_bt["model"])

        model_info = {
            "summary": summary_bt
        }

    return model_info

###################################################################################################################
## LOAD MODEL
###################################################################################################################

def load_instantiate_keras_model_preprocessing(mf_path, goal, input_shape=None, pooling=None, 
                                                reg=None, dropout=None, n_classes=None, verbose=True):
    config = config_utils.get_config(mf_path)
    architecture = config["architecture"]
    trained_on = config["trained_on"]

    if is_keras_application(architecture):
        model_and_pp = load_keras_application(config, mf_path, goal, input_shape, pooling,reg, dropout, n_classes, verbose)

    # TODO : handle non keras application if such algorithms are available

    return model_and_pp

def load_keras_application(config, mf_path, goal, input_shape, pooling, reg, dropout, n_classes, verbose):
    architecture = config["architecture"]
    trained_on = config["trained_on"]
    retrained = config.get(constants.RETRAINED, False)
    top_params = config.get(constants.TOP_PARAMS, None)
    model_params = {}

    if trained_on != constants.IMAGENET:
        raise IOError("The architecture '{}', trained on '{}' cannot be found".format(architecture, trained_on))

    if retrained and top_params is None:
        raise IOError("Your config file is missing some parameters : '{}'".format(constants.TOP_PARAMS))

    if goal == constants.SCORING:

        if not retrained:
            
            if top_params is None:

                model = build_keras_application()[architecture]["model_func"](weights=None, include_top=True)
                model_weights_path = get_weights_path(mf_path, config)
                model.load_weights(model_weights_path)

            else:

                model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=top_params["input_shape"])
                model, model_params = enrich_model(model, pooling, dropout, reg, n_classes, top_params, verbose)
                model_weights_path = get_weights_path(mf_path, config, constants.CUSTOM_TOP_SUFFIX)
                model.load_weights(model_weights_path)

        else:

            model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=top_params["input_shape"])
            model, model_params = enrich_model(model, pooling, dropout, reg, n_classes, top_params, verbose)
            model_weights_path = get_weights_path(mf_path, config, constants.RETRAINED_SUFFIX)
            model.load_weights(model_weights_path)

    elif goal == constants.RETRAINING:

        if not retrained:

            model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=input_shape)
            model_weights_path = get_weights_path(mf_path, config, constants.NOTOP_SUFFIX)
            model.load_weights(model_weights_path)
            model, model_params = enrich_model(model, pooling, dropout, reg, n_classes, top_params, verbose)
            model_params["input_shape"] = input_shape

        else:

            model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=top_params["input_shape"])
            model, model_params = enrich_model(model, pooling, dropout, reg, n_classes, top_params, verbose)
            model_weights_path = get_weights_path(mf_path, config, constants.RETRAINED_SUFFIX)
            model.load_weights(model_weights_path)

    elif goal == constants.BEFORE_TRAIN:

        if not retrained:

            model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=input_shape)
            model_weights_path = get_weights_path(mf_path, config, constants.NOTOP_SUFFIX)
            model.load_weights(model_weights_path)

        else:

            model = build_keras_application()[architecture]["model_func"](weights=None, include_top=False, input_shape=top_params["input_shape"])
            model, model_params = enrich_model(model, pooling, dropout, reg, n_classes, top_params, verbose)
            model_weights_path = get_weights_path(mf_path, config, constants.RETRAINED_SUFFIX)
            model.load_weights(model_weights_path)

    return {"model": model, "preprocessing": build_keras_application()[architecture]["preprocessing"], "model_params": model_params}

def select_param(param_name, param_val, top_params):
    return param_val if param_val is not None else top_params[param_name]

def enrich_model(base_model, pooling, dropout, reg, n_classes, params, verbose):

    # Init params if not done before
    params = {} if params is None else params
    
    # Loading appropriate params
    params["pooling"] = select_param("pooling", pooling, params)
    params["n_classes"] = select_param("n_classes", n_classes, params)

    x = base_model.layers[-1].output
        
    if params["pooling"] == 'None' :
        x = Flatten()(x)
    elif params["pooling"] == 'avg' :
        x = GlobalAveragePooling2D()(x)
    elif params["pooling"] == 'max' :
        x = GlobalMaxPooling2D()(x)

    if dropout is not None and dropout != 0.0 :
        x = Dropout(dropout)(x)
        if verbose:
            print("Adding dropout to model with rate: {}".format(dropout))

    regularizer = None
    if reg is not None:
        reg_l2 = reg["l2"]
        reg_l1 = reg["l1"]
        if (reg_l1 != 0.0) and (reg_l2 != 0.0) :
            regularizer = regularizers.l1_l2(l1=reg_l1, l2=reg_l2)
        if (reg_l1 == 0.0) and (reg_l2 != 0.0) :
            regularizer = regularizers.l2(reg_l2)
        if (reg_l1 != 0.0) and (reg_l2 == 0.0) :
            regularizer = regularizers.l1(reg_l1)
        if verbose:
            print("Using regularizer for model: {}".format(reg))
    
    predictions = Dense(params["n_classes"], activation='softmax', name='predictions', kernel_regularizer=regularizer)(x)
    model = Model(input=base_model.input, output=predictions)

    return model, params

###################################################################################################################
## GPU
###################################################################################################################

# TODO: Rename this function as it has a lot of border effects
def load_gpu_options(should_use_gpu, list_gpu_str, gpu_allocation):
    gpu_options = {}
    print("load_gpu_options")
    if should_use_gpu:
        print("should use GPU")
        list_gpu = list(map(int, list_gpu_str.replace(" ", "").split(",")))
        gpu_options["list_gpu"] = list_gpu
        gpu_options["n_gpu"] = len(list_gpu)

        config_tf = tf.ConfigProto()
        #os.environ['CUDA_VISIBLE_DEVICES'] = list_gpu_str
        #print  os.environ['CUDA_VISIBLE_DEVICES']
        config_tf.gpu_options.per_process_gpu_memory_fraction = gpu_allocation
        #config_tf.log_device_placement=True
        session = tf.Session(config=config_tf)
        from keras.backend.tensorflow_backend import set_session
        set_session(session)
    else:
        config_utils.deactivate_gpu()

    return gpu_options



###################################################################################################################
## FILES LOGIC
###################################################################################################################

def get_weights_path(mf_path, config, suffix="", should_exist=True):
    weights_filename =  get_weights_filename(mf_path, config, suffix)
    if should_exist : 
        model_weights_path = mf_path.get_download_stream(weights_filename)
        #Hack to get the H5 stream coming from the folder API as a file 
        #Hack inspired from https://stackoverflow.com/questions/16654251/can-h5py-load-a-file-from-a-byte-array-in-memory 
        h5file = tables.open_file(weights_filename+"_temp", driver="H5FD_CORE",
                              driver_core_image=model_weights_path.read(),
                              driver_core_backing_store=0)
        h5file.copy_file(weights_filename, overwrite=True)
    print(weights_filename)
    return weights_filename

   # if not os.path.isfile(model_weights_path) and should_exist:
    #    raise IOError("No weigth file found")

    #return model_weights_path

def get_weights_filename(mf_path, config, suffix=""):
    return "{}_{}_weights{}.h5".format(config["architecture"], config["trained_on"], suffix)



def write_config(mf_path, config):
    
    with mf_path.get_writer(constants.CONFIG_FILE) as w:
        w.write(json.dumps(config))
    #config_path = get_file_path(mf_path, constants.CONFIG_FILE)
    #with open(config_path, 'w') as f:
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

def get_cached_file_from_folder(folder, file_path) :
    
    filename = file_path.replace('/','_')
    if not(os.path.exists(filename)) :
        with folder.get_download_stream(file_path) as stream:
             with open(filename, 'wb') as f : 
                f.write(stream.read())
                print("cached file %s" %file_path)
    else :
        print("read from cache %s" %file_path)
    return filename

def get_model_config_from_file(model_folder):
    return json.loads(model_folder.get_download_stream( constants.CONFIG_FILE).read())

def build_prediction_output_df(images_paths, predictions):
    output = pd.DataFrame()
    output["images"] = images_paths
    print("------->" + str(output))
    output["prediction"] = predictions["prediction"]
    output["error"] = predictions["error"]
    return output
###################################################################################################################
## MISC.
###################################################################################################################
def log_func(txt):
    def inner(f):
        def wrapper(*args, **kwargs):
            print('------ \n Info: Starting {} ({}) \n ------'.format(txt, datetime.now().strftime('%H:%M:%S')))
            res = f(*args, **kwargs)
            print('------ \n Info: Ending {} ({}) \n ------'.format(txt, datetime.now().strftime('%H:%M:%S')))
            return res
        return wrapper
    return inner

def format_predictions_output(predictions, classify=False, labels_df=None, limit=None, min_threshold=None):
    if not classify:
        return predictions.tolist()
    formatted_predictions = []
    id_pred = lambda index: labels_df.loc[index].className if labels_df else str(index)
    for pred in predictions:
        formatted_pred = get_ordered_dict(
            {id_pred(i): float(pred[i]) for i in pred.argsort()[-limit:] if float(pred[i]) >= min_threshold})
        formatted_predictions.append(formatted_pred)
    return formatted_predictions

def get_predictions(model, batch, classify=False, limit=constants.DEFAULT_PRED_LIMIT, min_threshold=0, labels_df=None):
    predictions = model.predict(batch)
    return format_predictions_output(predictions, labels_df, limit, min_threshold)

def score(dku_model, images_folder, images_paths, limit, min_threshold, labels_df=None):
    batch_size = constants.PREDICTION_BATCH_SIZE
    n = 0
    results = {"prediction": [], "error": []}
    num_images = len(images_paths)
    while True:
        if (n * batch_size) >= num_images: break
        next_batch_list, error_indices = [], []
        for index_in_batch, i in enumerate(range(n * batch_size, min((n + 1) * batch_size, num_images))):
            img_path = images_paths[i]
            try:
                preprocessed_img = preprocess_img(
                    img_path=images_folder.get_download_stream(img_path),
                    img_shape=dku_model.model_input_shape,
                    preprocessing=dku_model.preprocessing
                )
                next_batch_list.append(preprocessed_img)
            except IOError as e:
                print("Cannot read the image '{}', skipping it. Error: {}".format(img_path, e))
                error_indices.append(index_in_batch)
        next_batch = np.array(next_batch_list)

        prediction_batch = get_predictions(
            model=dku_model.model,
            batch=next_batch,
            classify=dku_model.get_name() == 'score',
            limit=limit,
            min_threshold=min_threshold,
            labels_df=labels_df
        )
        error_batch = [0] * len(prediction_batch)

        for err_index in error_indices:
            prediction_batch.insert(err_index, None)
            error_batch.insert(err_index, 1)

        results["prediction"].extend(prediction_batch)
        results["error"].extend(error_batch)
        n += 1
        print("{} images treated, out of {}".format(min(n * batch_size, num_images), num_images))
    return results

def get_ordered_dict(predictions):
    return json.dumps(OrderedDict(sorted(predictions.items(), key=(lambda x: -x[1]))))


def preprocess_img(img_path, img_shape, preprocessing):
    img = load_img(img_path,target_size=img_shape)
    array = img_to_array(img)
    array = preprocessing(array)
    return array

def clean_custom_params(custom_params, params_type=""):

    def string_to_arg(string):
        if string.lower() == "true":
            res = True
        elif string.lower() == "false":
            res = False
        else :
            try :
                res = np.float(string)
            except ValueError:
                res = string
        return res

    cleaned_params = {}
    params_type = " '{}'".format(params_type) if params_type else ""
    for i, p in enumerate(custom_params) :
        if not p.get("name", False):
            raise IOError("The{} custom param #{} must have a 'name'".format(params_type, i))
        if not p.get("value", False):
            raise IOError("The{} custom param #{} must have a 'value'".format(params_type, i))
        name = p["name"]
        value = string_to_arg(p["value"])
        cleaned_params[name] = value
    return cleaned_params

def display_gpu_device():
    print(device_lib.list_local_devices())
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

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
## Dictionary as class
###############################################################

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

###############################################################
## MODEL CHECKPOINT FOR MULTI GPU
## When using multiple GPUs, we need to save the base model,
## not the one defined by multi_gpu_model
## see example: https://keras.io/utils/#multi_gpu_model
## Therefore, to save the model after each epoch by leveraging
## ModelCheckpoint callback, we need to adapt it to save the
## base model. To do so, we pass the base model to the callback.
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

        # First retrieve model
        model = self.model

        # Then switching model to base model
        self.model = self.base_model

        # Calling super on_epoch_end
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # Resetting model afterwards
        self.model = model