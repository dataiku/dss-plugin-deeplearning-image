import dataiku
import pandas as pd
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from keras.models import Model
import dl_image_toolbox_utils as utils
import constants

def load_input_output(config):
    image_folder_input_name = get_input_names_for_role('image_folder')[0]
    config.image_folder = dataiku.Folder(image_folder_input_name)

    model_folder_input_name = get_input_names_for_role('model_folder')[0]
    config.model_folder = dataiku.Folder(model_folder_input_name)

    output_name = get_output_names_for_role('feature_dataset')[0]
    config.output_dataset = dataiku.Dataset(output_name)


def load_model(config):
    model_and_pp = utils.load_instantiate_keras_model_preprocessing(config.model_folder, goal=constants.SCORING)
    model = model_and_pp["model"]
    config.model = Model(inputs=model.input, outputs=model.layers[config.extract_layer_index].output)
    config.preprocessing = model_and_pp["preprocessing"]
    config.model_input_shape = utils.get_model_input_shape(config.model, config.model_folder)


def load_recipe_params(config):
    recipe_config = get_recipe_config()
    config.extract_layer_index = int(recipe_config['extract_layer_index'])
    config.should_use_gpu = recipe_config.get('should_use_gpu', False)

    # gpu
    config.gpu_options = utils.load_gpu_options(
        config.should_use_gpu, recipe_config['list_gpu'], recipe_config['gpu_allocation'])


def load_images_path(config):
    config.images_paths = config.image_folder.list_paths_in_partition()

def load_config():
    config = utils.AttributeDict()

    load_recipe_params(config)
    load_input_output(config)
    load_model(config)
    load_images_path(config)

    return config

###################################################################################################################
## EXTRACTING FEATURES
###################################################################################################################

# Helper for predicting
def get_predictions():
    batch_size = 100
    n = 0
    results = {"prediction": [], "error": []}
    num_images = len(images_paths)
    while True:
        if (n * batch_size) >= num_images:
            break

        next_batch_list = []
        error_indices = []
        for index_in_batch, i in enumerate(range(n*batch_size, min((n + 1)*batch_size, num_images))):
            img_path = images_paths[i]
            try:
                preprocessed_img = utils.preprocess_img(image_folder.get_download_stream(img_path), model_input_shape, preprocessing)
                next_batch_list.append(preprocessed_img)
            except IOError as e:
                print("Cannot read the image '{}', skipping it. Error: {}".format(img_path, e))
                error_indices.append(index_in_batch)
        next_batch = np.array(next_batch_list)

        prediction_batch = model.predict(next_batch).tolist()
        error_batch = [0] * len(prediction_batch)

        for err_index in error_indices:
            prediction_batch.insert(err_index, None)
            error_batch.insert(err_index, 1)

        results["prediction"].extend(prediction_batch)
        results["error"].extend(error_batch)
        n+=1
        print("{} images treated, out of {}".format(min(n * batch_size, num_images), num_images))
    return results

# Make the predictions
print("------ \n Info: Start predicting \n ------")
predictions = get_predictions()
print("------ \n Info: Finished predicting \n ------")

###################################################################################################################
## SAVING RESULTS
###################################################################################################################
 
# Prepare results
output = pd.DataFrame()
output["images"] = images_paths
print("------->" + str(output))
output["prediction"] = predictions["prediction"]
output["error"] = predictions["error"]

# Write to output dataset    
print("------ \n Info: Writing to output dataset \n ------")
output_dataset.write_with_schema(pd.DataFrame(output))
print("------ \n Info: END of recipe \n ------")