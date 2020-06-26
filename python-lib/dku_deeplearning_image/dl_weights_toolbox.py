import os
import dku_deeplearning_image.constants as constants
import threading
import json
from collections import OrderedDict
import StringIO
import numpy as np
import sys
import tables #to get the h5 file stream from the folder API as a file to be read by the keras API
import dku_deeplearning_image.config_utils as config_utils
# Support Truncated Images with PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# INFO : when adding a new architecture, you must add a select-option in python-runnables/dl-toolbox-download-models/runnable.json
#        with the label architecture_trainedon to make it available, along with new a constant in python-lib/constants.py
keras_applications = {
    constants.RESNET: {
        #"model_func": ResNet50, 
        #"preprocessing": resnet50_preprocessing,
        "input_shape": (224, 224),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    constants.XCEPTION: {
        #"model_func": Xception,
        #"preprocessing": xception_preprocessing,
        "input_shape": (299, 299),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            }
        }
    },
    constants.INCEPTIONV3: {
        #"model_func": InceptionV3,
        #"preprocessing": inceptionv3_preprocessing,
        "input_shape": (299, 299),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    constants.VGG16: {
        #"model_func": VGG16,
        #"preprocessing": vgg16_preprocessing,
        "input_shape": (224, 224),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    }
}

def is_keras_application(architecture):
    return architecture in keras_applications.keys()

def get_extract_layer_index(architecture, trained_on):
    # May be more complicated as the list of models grows
    return -2


def get_weights_urls(architecture, trained_on):
    if is_keras_application(architecture):
        return keras_applications[architecture]["weights"][trained_on]
    else:
        return {}

def should_save_weights_only(config):
    if config["architecture"] in keras_applications.keys():
        return True
    return False
 
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

def get_extract_layer_index(architecture, trained_on):
    # May be more complicated as the list of models grows
    return -2


def write_config(mf_path, config):
    
    with mf_path.get_writer(constants.CONFIG_FILE) as w:
        w.write(json.dumps(config))
    #config_path = get_file_path(mf_path, constants.CONFIG_FILE)
    #with open(config_path, 'w') as f:
     #   json.dump(config, f)

def check_managed_folder_filesystem(managed_folder):
    managed_folder_info = managed_folder.get_info()
    managed_folder_name = managed_folder_info["name"]
    connection_type = managed_folder_info["type"]

    if connection_type != "Filesystem" :
        raise IOError("The managed folder '{}' has a '{}' connection. Only Filesystem based managed folders are supported.".format(managed_folder_name, connection_type))

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
    return filename
