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


@utils.log_func(txt='config loading')
def load_config():
    config = utils.AttributeDict()

    load_recipe_params(config)
    load_input_output(config)
    load_model(config)
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
    predictions = utils.predict(config, labelize=False)
    output_df = build_output_df(config.images_paths, predictions)
    write_output_dataset(config.output_dataset, output_df)


run()
