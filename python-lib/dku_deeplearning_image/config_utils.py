#To move some utils function that does not needs Keras and so can be called from the backend without GPU installed and configured
import json
import os
import dku_deeplearning_image.constants as constants
import tensorflow as tf

def get_config(model_folder):
    return json.loads(model_folder.get_download_stream(constants.CONFIG_FILE).read())

def deactivate_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_tensorflow_version():
    import pkg_resources
    tf_lib = pkg_resources.working_set.by_key.get('tensorflow')
    tf_lib_gpu = pkg_resources.working_set.by_key.get('tensorflow-gpu')
    if tf_lib and tf_lib_gpu:
        raise IOError(
            'Both tensorflow and tensorflow-gpu are installed. You should use an isolated env to prevent conflicts.')
    elif tf_lib_gpu:
        print('WARNING: You are using an obsolete version of tensorlow for which cpu and gpu versions are separated.')
        return tf_lib_gpu.version
    return tf_lib.version

def can_use_gpu():
    real_n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    return True # real_n_gpus > 0
    
def get_model_info(model_folder, goal):
    
    #1st check that the files exist at the root of the model_folder
    if '/'+constants.MODEL_INFO_FILE  in model_folder.list_paths_in_partition() : 
        model_info = json.loads(model_folder.get_download_stream( constants.MODEL_INFO_FILE).read())
        return model_info[goal]
    else:
        #needs Keras and so GPU for the GPU version. GPU might not be available on DSS side
        return { "summary" : "Not Available before 1st run", "layers" : "Not Available before 1st run" }
