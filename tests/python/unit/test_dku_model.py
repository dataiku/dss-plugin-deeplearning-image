import pytest
from utils_objects import DkuModel, VirtualManagedFolder
import os
import dku_deeplearning_image.constants as constants


def create_base_model(output_folder, model_choice):
    architecture, trained_on = model_choice.split('::')
    config = {
        "architecture": architecture,
        "trained_on": trained_on,
        "extract_layer_default_index": -2
    }

    new_model = DkuModel(output_folder, is_empty=True)
    new_model.set_config(config)
    return new_model

class TestDkuModel:
    def setup_class(self):
        PATH = './model_resnet_unit_test'
        INFO_FILENAME = 'model_info.json'
        MODEL_CHOICE = 'resnet::imagenet'
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        model_folder = VirtualManagedFolder(PATH)
        if not os.path.exists(os.path.join(PATH, INFO_FILENAME)):
            new_model = create_base_model(model_folder, MODEL_CHOICE)
            new_model.download_from_web()

        self.dku_model = DkuModel(model_folder)
        self.dku_model.load_model({}, constants.SCORING)

    def test_init(self):
        return not not self.dku_model
