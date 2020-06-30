from dataiku.customrecipe import get_input_names_for_role
import dku_deeplearning_image.utils as utils

from model.retrain_model import RetrainModel
from config.retrain_config import RetrainConfig
from utils.dku_file_manager import DkuFileManager

import pandas as pd

import dku_deeplearning_image.constants as constants
import dataiku


def get_label_df(col_filename, col_label):
    label_dataset_input_name = get_input_names_for_role('label_dataset')[0]
    label_dataset = dataiku.Dataset(label_dataset_input_name)
    renaming_mapping = {
        col_filename: constants.FILENAME,
        col_label: constants.LABEL
    }
    label_df = label_dataset.get_dataframe().rename(columns=renaming_mapping)[renaming_mapping.values()]
    return label_df


def get_input_output(col_filename, col_label):
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    label_df = get_label_df(col_filename, col_label)
    output_folder = file_manager.get_output_folder('model_output')
    return image_folder, label_df, model_folder, output_folder


def save_output_model(output_folder, model):
    model.model_config[constants.RETRAINED] = True
    model.model_config[constants.TOP_PARAMS] = model.model_params
    utils.write_config(output_folder, model.model_config)

    df_labels = pd.DataFrame({"id": range(len(model.labels)), "className": model.labels})
    with output_folder.get_writer(constants.MODEL_LABELS_FILE) as w:
        w.write((df_labels.to_csv(index=False)))

    # This copies a local file to the managed folder
    model_weights_path = utils.get_weights_path(
        output_folder,
        config=model.model_config,
        suffix=constants.RETRAINED_SUFFIX,
        should_exist=False
    )

    with open(model_weights_path) as f:
        output_folder.upload_stream(model_weights_path, f)
    # Computing model info
    utils.save_model_info(output_folder)


@utils.log_func(txt='recipe')
def run():
    config = RetrainConfig()

    image_folder, label_df, model_folder, output_folder = get_input_output(
        col_filename=config.col_filename,
        col_label=config.col_label
    )

    model = RetrainModel(model_folder, label_df, config)
    model.retrain(image_folder, label_df, output_folder)

    save_output_model(output_folder, model)


run()
