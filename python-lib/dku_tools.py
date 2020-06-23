# -*- coding: utf-8 -*-

import dataiku
import pandas as pd
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role
from keras.models import Model
import dl_image_toolbox_utils as utils
import dku_deeplearning_image.constants as constants
import numpy as np

############################################################
# Config loading
############################################################
def extract__load_recipe_params(config):
    recipe_config = get_recipe_config()
    config.extract_layer_index = int(recipe_config['extract_layer_index'])
    config.should_use_gpu = recipe_config.get('should_use_gpu', False)

    # gpu
    config.gpu_options = utils.load_gpu_options(
        config.should_use_gpu, recipe_config['list_gpu'], recipe_config['gpu_allocation'])


def score__load_recipe_params(config):
    recipe_config = get_recipe_config()
    config.max_nb_labels = int(recipe_config['max_nb_labels'])
    config.min_threshold = float(recipe_config['min_threshold'])
    config.should_use_gpu = recipe_config.get('should_use_gpu', False)

    # gpu
    config.gpu_options = utils.load_gpu_options(
        config.should_use_gpu, recipe_config['list_gpu'], recipe_config['gpu_allocation'])


def retrain__load_recipe_config(config):
    recipe_config = get_recipe_config()

    config.should_use_gpu = recipe_config.get('should_use_gpu', False)
    config.list_gpu = recipe_config["list_gpu"]
    config.gpu_allocation = recipe_config["gpu_allocation"]
    config.train_ratio = float(recipe_config["train_ratio"])
    config.input_shape = (int(recipe_config["image_width"]), int(recipe_config["image_height"]), 3)
    config.batch_size = int(recipe_config["batch_size"])
    config.model_pooling = recipe_config["model_pooling"]
    config.model_reg = recipe_config["model_reg"]
    config.model_dropout = float(recipe_config["model_dropout"])
    config.layer_to_retrain = recipe_config["layer_to_retrain"]
    config.layer_to_retrain_n = int(recipe_config.get('layer_to_retrain_n', 0))
    config.optimizer = recipe_config["model_optimizer"]
    config.learning_rate = recipe_config["model_learning_rate"]
    config.custom_params_opti = recipe_config.get("model_custom_params_opti", [])
    config.nb_epochs = int(recipe_config["nb_epochs"])
    config.nb_steps_per_epoch = int(recipe_config["nb_steps_per_epoch"])
    config.nb_validation_steps = int(recipe_config["nb_validation_steps"])
    config.data_augmentation = recipe_config["data_augmentation"]
    config.n_augmentation = int(recipe_config["n_augmentation"])
    config.custom_params_data_augment = recipe_config.get("model_custom_params_data_augmentation", [])
    config.use_tensorboard = recipe_config["tensorboard"]
    config.random_seed = int(recipe_config["random_seed"])
    config.gpu_options = utils.load_gpu_options(config.should_use_gpu, config.list_gpu, config.gpu_allocation)
    config.n_gpu = config.gpu_options.get("n_gpu", 0)


def retrain__load_label_df(config):
    recipe_config = get_recipe_config()

    label_dataset_input_name = get_input_names_for_role('label_dataset')[0]
    config.label_dataset = dataiku.Dataset(label_dataset_input_name)
    renaming_mapping = {
        recipe_config["col_filename"]: constants.FILENAME,
        recipe_config["col_label"]: constants.LABEL
    }
    config.label_df = config.label_dataset.get_dataframe().rename(columns=renaming_mapping)[renaming_mapping.values()]
    config.labels = list(np.unique(config.label_df[constants.LABEL]))
    config.n_classes = len(config.labels)

def load_output(config, recipe):
    output_type = dataiku.Folder if recipe == constants.Recipe.RETRAIN else dataiku.Dataset
    output_name = get_output_names_for_role(recipe.value['output_role'])[0]
    config.output_model_folder = output_type(output_name)

def load_input(config, recipe):
    image_folder_input_name = get_input_names_for_role('image_folder')[0]
    config.image_folder = dataiku.Folder(image_folder_input_name)

    model_folder_input_name = get_input_names_for_role('model_folder')[0]
    config.model_folder = dataiku.Folder(model_folder_input_name)

    if recipe == constants.Recipe.RETRAIN:
        retrain__load_label_df(config)

def load_recipe_config(config, recipe):
    

def load_input_output(config, recipe):
    load_input(config, recipe)
    load_output(config, recipe)


def load_model(config):
    model_and_pp = utils.load_instantiate_keras_model_preprocessing(config.model_folder, goal=constants.SCORING)
    config.model = model_and_pp["model"]
    config.preprocessing = model_and_pp["preprocessing"]
    config.model_input_shape = utils.get_model_input_shape(config.model, config.model_folder)


def load_recipe_params(config):
    recipe_config = get_recipe_config()
    config.max_nb_labels = int(recipe_config['max_nb_labels'])
    config.min_threshold = float(recipe_config['min_threshold'])
    config.should_use_gpu = recipe_config.get('should_use_gpu', False)

    # gpu
    config.gpu_options = utils.load_gpu_options(
        config.should_use_gpu, recipe_config['list_gpu'], recipe_config['gpu_allocation'])


def load_labels_df(config):
    details_model_label = config.model_folder.get_path_details(constants.MODEL_LABELS_FILE)
    if details_model_label['exists'] and not details_model_label["directory"]:
        labels_path = config.model_folder.get_download_stream(constants.MODEL_LABELS_FILE)
        config.labels_df = pd.read_csv(labels_path, sep=",").set_index('id')
    else:
        print("------ \n Info: No csv file in the model folder, will not use class names. \n ------")
        config.labels_df = None


def load_images_path(config):
    config.images_paths = config.image_folder.list_paths_in_partition()


@utils.log_func(txt='config loading')
def load_config():
    config = utils.AttributeDict()

    load_recipe_params(config)
    load_input_output(config)
    load_model(config)
    load_labels_df(config)
    load_images_path(config)

    return config