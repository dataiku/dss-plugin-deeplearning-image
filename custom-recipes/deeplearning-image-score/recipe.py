import pandas as pd
import dku_deeplearning_image.utils as utils

from model.score_model import ScoreModel
from config.score_config import ScoreConfig
from utils.dku_file_manager import DkuFileManager
import dku_deeplearning_image.constants as constants


def get_label_df(model_folder):
    details_model_label = model_folder.get_path_details(constants.MODEL_LABELS_FILE)
    if details_model_label['exists'] and not details_model_label["directory"]:
        labels_path = model_folder.get_download_stream(constants.MODEL_LABELS_FILE)
        labels_df = pd.read_csv(labels_path, sep=",").set_index('id')
    else:
        print("------ \n Info: No csv file in the model folder, will not use class names. \n ------")
        labels_df = None
    return labels_df


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    label_df = get_label_df(model_folder)
    output_dataset = file_manager.get_output_dataset('scored_dataset')
    return image_folder, label_df, model_folder, output_dataset


@utils.log_func(txt='output dataset writing')
def write_output_dataset(output_dataset, images_paths, classification):
    output_df = utils.build_prediction_output_df(images_paths, classification)
    output_dataset.write_with_schema(pd.DataFrame(output_df))


@utils.log_func(txt='recipe')
def run():
    image_folder, label_df, model_folder, output_dataset = get_input_output()
    images_paths = image_folder.list_paths_in_partition()

    config = ScoreConfig()
    model = ScoreModel(model_folder, config)
    classification = model.classify(image_folder, label_df)

    write_output_dataset(output_dataset, images_paths, classification)


run()
