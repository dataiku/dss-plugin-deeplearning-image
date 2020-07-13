import dku_deeplearning_image.utils as utils

from recipe.retrain_recipe import RetrainRecipe
from config.retrain_config import RetrainConfig
from utils_objects.dku_file_manager import DkuFileManager

import pandas as pd

import dku_deeplearning_image.constants as constants


def format_label_df(label_dataset, col_filename, col_label):
    renaming_mapping = {
        col_filename: constants.FILENAME,
        col_label: constants.LABEL
    }
    label_df = label_dataset.get_dataframe().rename(columns=renaming_mapping)[renaming_mapping.values()]
    return label_df


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    label_dataset = file_manager.get_input_dataset('label_dataset')
    output_folder = file_manager.get_output_folder('model_output')
    return image_folder, label_dataset, model_folder, output_folder


def save_output_model(output_folder, model):
    utils.write_config(output_folder, model.jsonify_config())

    labels = model.get_distinct_labels()
    df_labels = pd.DataFrame({"id": range(len(labels)), "className": labels})
    with output_folder.get_writer(constants.MODEL_LABELS_FILE) as w:
        w.write((df_labels.to_csv(index=False)))

    # This copies a local file to the managed folder
    model_weights_path = model.get_weights_path()

    with open(model_weights_path) as f:
        output_folder.upload_stream(model_weights_path, f)
    # Computing recipe info
    utils.save_model_info(output_folder, model)


@utils.log_func(txt='recipe')
def run():
    config = RetrainConfig()

    image_folder, label_dataset, model_folder, output_folder = get_input_output()

    label_df = format_label_df(label_dataset, config.col_filename, config.col_label)
    recipe = RetrainRecipe(config)

    new_model = recipe.compute(image_folder, model_folder, label_df, output_folder)

    save_output_model(output_folder, new_model)


run()
