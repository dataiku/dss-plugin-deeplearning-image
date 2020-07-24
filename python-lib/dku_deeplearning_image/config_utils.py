#To move some utils function that does not needs Keras and so can be called from the backend without GPU installed and configured
import json
import os
import dku_deeplearning_image.constants as constants

def get_config(model_folder):
    return json.loads(model_folder.get_download_stream(constants.CONFIG_FILE).read())

def deactivate_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
def get_model_info(model_folder, goal):
    #1st check that the files exist at the root of the model_folder
    if '/'+constants.MODEL_INFO_FILE  in model_folder.list_paths_in_partition() : 
        model_info = json.loads(model_folder.get_download_stream( constants.MODEL_INFO_FILE).read())
        return model_info[goal]
    else:
        #needs Keras and so GPU for the GPU version. GPU might not be available on DSS side
        return { "summary" : "Not Available before 1st run", "layers" : "Not Available before 1st run" }
