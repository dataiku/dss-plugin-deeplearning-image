import pandas as pd
from dataiku.customrecipe import get_recipe_config

import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.config_handler import create_dku_config

from dku_deeplearning_image.recipes import ExtractRecipe
from dku_deeplearning_image.misc_objects import DkuFileManager


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    output_dataset = file_manager.get_output_dataset('feature_dataset')
    return image_folder, model_folder, output_dataset


@utils.log_func(txt='output dataset writing')
def write_output_dataset(output_dataset, image_folder, features):
    images_paths = image_folder.list_paths_in_partition()
    output_df = utils.build_prediction_output_df(images_paths, features)
    output_dataset.write_with_schema(pd.DataFrame(output_df))


@utils.log_func(txt='recipes')
def run():
    recipe_config = get_recipe_config()
    config = create_dku_config(recipe_config, constants.EXTRACT)

    image_folder, model_folder, output_dataset = get_input_output()
    recipe = ExtractRecipe(config)

    features = recipe.compute(image_folder, model_folder)

    write_output_dataset(output_dataset, image_folder, features)


run()
