import pandas as pd
import dku_deeplearning_image.utils as utils

from model.score_recipe import ScoreRecipe
from config.score_config import ScoreConfig
from utils.dku_file_manager import DkuFileManager
import dku_deeplearning_image.constants as constants


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


@utils.log_func(txt='recipe')
def run():
    image_folder, model_folder, output_dataset = get_input_output()
    config = ScoreConfig()
    recipe = ScoreRecipe(config)

    classification = recipe.compute(image_folder, model_folder)

    write_output_dataset(output_dataset, image_folder, classification)


run()
