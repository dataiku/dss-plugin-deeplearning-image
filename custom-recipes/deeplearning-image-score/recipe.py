import pandas as pd
import dl_image_toolbox_utils as utils

from model.extract_model import ScoreModel
from config.extract_config import ScoreConfig
from utils.dku_file_manager import DkuFileManager


def get_input_output():
    file_manager = DkuFileManager()
    image_folder = file_manager.get_input_folder('image_folder')
    model_folder = file_manager.get_input_folder('model_folder')
    output_dataset = file_manager.get_output_dataset('scored_dataset')
    return image_folder, model_folder, output_dataset


@utils.log_func(txt='output dataset writing')
def write_output_dataset(output_dataset, images_paths, predictions):
    output_df = utils.build_output_df(images_paths, predictions)
    output_dataset.write_with_schema(pd.DataFrame(output_df))


@utils.log_func(txt='recipe')
def run():
    image_folder, model_folder, output_dataset = get_input_output()
    images_paths = image_folder.list_paths_in_partition()

    config = ScoreConfig()
    model = ScoreModel(config, model_folder)
    predictions = model.classify(image_folder)

    write_output_dataset(output_dataset, images_paths, predictions)


run()
