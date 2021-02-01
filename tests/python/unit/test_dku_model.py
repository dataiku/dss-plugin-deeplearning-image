import pytest
from utils_objects import DkuModel, VirtualManagedFolder
import os


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
        PATH = 'model_resnet_unit_test'
        MODEL_CHOICE = 'resnet::imagenet'
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        output_folder = VirtualManagedFolder(PATH)
        self.new_model = create_base_model(output_folder, MODEL_CHOICE)
        self.new_model.download_from_web()

    def test_init(self):
        return not not self.new_model
