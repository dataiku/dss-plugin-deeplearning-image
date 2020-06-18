import dataiku
import pandas as pd
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
import constants
import dl_image_toolbox_utils as utils

def load_input_output(config):
    image_folder_input_name = get_input_names_for_role('image_folder')[0]
    config.image_folder = dataiku.Folder(image_folder_input_name)

    model_folder_input_name = get_input_names_for_role('model_folder')[0]
    config.model_folder = dataiku.Folder(model_folder_input_name)

    output_name = get_output_names_for_role('scored_dataset')[0]
    config.output_dataset = dataiku.Dataset(output_name)


def load_model(config):
    model_and_pp = utils.load_instantiate_keras_model_preprocessing(config.model_folder, goal=constants.SCORING)
    config.model = model_and_pp["model"]
    config.preprocessing = model_and_pp["preprocessing"]
    config.model_input_shape = utils.get_model_input_shape(config.model, config.model_folder)


def load_recipe_params(config):
    recipe_config = get_recipe_config()
    config.max_nb_labels = int(recipe_config['max_nb_labels'])
    config.min_threshold = float(recipe_config['min_threshold'])
    config.should_use_gpu = recipe_config.get('should_use_gpu', False)

    # gpu
    config.gpu_options = utils.load_gpu_options(
        config.should_use_gpu, recipe_config['list_gpu'], recipe_config['gpu_allocation'])


def load_labels_df(config):
    details_model_label = config.model_folder.get_path_details(constants.MODEL_LABELS_FILE)
    if details_model_label['exists'] and not details_model_label["directory"]:
        labels_path = config.model_folder.get_download_stream(constants.MODEL_LABELS_FILE)
        config.labels_df = pd.read_csv(labels_path, sep=",").set_index('id')
    else:
        print("------ \n Info: No csv file in the model folder, will not use class names. \n ------")
        config.labels_df = None


def load_images_path(config):
    config.images_paths = config.image_folder.list_paths_in_partition()


@utils.log_func(txt='config loading')
def load_config():
    config = utils.AttributeDict()

    load_recipe_params(config)
    load_input_output(config)
    load_model(config)
    load_labels_df(config)
    load_images_path(config)

    return config

def build_output_df(images_paths, predictions):
    output = pd.DataFrame()
    output["images"] = images_paths
    print("------->" + str(output))
    output["prediction"] = predictions["prediction"]
    output["error"] = predictions["error"]
    return output


@utils.log_func(txt='output dataset writing')
def write_output_dataset(output_dataset, output_df):
    output_dataset.write_with_schema(pd.DataFrame(output_df))


@utils.log_func(txt='recipe')
def run():
    config = load_config()
    predictions = utils.predict(config, config.max_nb_labels, config.min_threshold)
    output_df = build_output_df(config.images_paths, predictions)
    write_output_dataset(config.output_dataset, output_df)


run()