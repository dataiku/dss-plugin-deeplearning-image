import pandas as pd
from dataiku.customrecipe import get_recipe_config

print(help('modules'))

import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.config_handler import create_dku_config

from dku_deeplearning_image.recipes import ScoreRecipe
from dku_deeplearning_image.misc_objects import DkuFileManager


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    output_dataset = file_manager.get_output_dataset('scored_dataset')
    return image_folder, model_folder, output_dataset


@utils.log_func(txt='output dataset writing')
def write_output_dataset(output_dataset, image_folder, classification):
    images_paths = image_folder.list_paths_in_partition()
    output_df = utils.build_prediction_output_df(images_paths, classification)
    output_dataset.write_with_schema(pd.DataFrame(output_df))


@utils.log_func(txt='recipes')
def run():
    recipe_config = get_recipe_config()
    config = create_dku_config(recipe_config, constants.GOAL.SCORE)

    image_folder, model_folder, output_dataset = get_input_output()
    recipe = ScoreRecipe(config)

    classification = recipe.compute(image_folder, model_folder)

    write_output_dataset(output_dataset, image_folder, classification)


run()
