from dataiku.customrecipe import get_recipe_config

import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.dku_constants as constants
from dku_deeplearning_image.config_handler import create_dku_config


from dku_deeplearning_image.recipes import RetrainRecipe
from dku_deeplearning_image.misc_objects import DkuFileManager


def format_label_df(label_dataset, col_filename, col_label):
    renaming_mapping = {
        col_filename: constants.FILENAME,
        col_label: constants.LABEL
    }
    label_df = label_dataset.get_dataframe().rename(columns=renaming_mapping)[list(renaming_mapping.values())]
    return label_df


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    label_dataset = file_manager.get_input_dataset('label_dataset')
    output_folder = file_manager.get_output_folder('model_output')
    return image_folder, label_dataset, model_folder, output_folder


def save_output_model(output_folder, model):
    output_model = model.deepcopy(folder=output_folder)
    output_model.save_to_folder()


@utils.log_func(txt='recipes')
def run():
    recipe_config = get_recipe_config()
    config = create_dku_config(recipe_config, constants.RETRAIN)
    image_folder, label_dataset, model_folder, output_folder = get_input_output()
    label_df = format_label_df(label_dataset, config.col_filename, config.col_label)
    recipe = RetrainRecipe(config)
    new_model = recipe.compute(image_folder, model_folder, label_df, output_folder)
    save_output_model(output_folder, new_model)


run()
