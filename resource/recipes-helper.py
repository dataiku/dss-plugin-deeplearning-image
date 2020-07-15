import dataiku
import glob
import pandas as pd
import dku_deeplearning_image.config_utils as config_utils
# import tensorflow as tf
import dku_deeplearning_image.constants as constants
import os


# We deactivate GPU for this script, because all the methods only need to
# fetch information about recipe and do not make computation
# config_utils.deactivate_gpu()

def do(payload, config, plugin_config, inputs):
    if "method" not in payload:
        return {}

    client = dataiku.api_client()

    if payload["method"] == "get-info-scoring":
        return get_info_scoring(inputs)

    if payload["method"] == "get-info-about-model":
        return get_info_about_model(inputs)

    if payload["method"] == "get-info-retrain":
        return get_info_retrain(inputs)


def get_info_scoring(inputs):
    return add_can_use_gpu_to_resp({})


def get_info_about_model(inputs):
    model_folder = get_model_folder_path(inputs)

    model_info = config_utils.get_model_info(model_folder, goal=constants.SCORING)
    config = config_utils.get_config(model_folder)

    return add_can_use_gpu_to_resp({
        "layers": model_info["layers"],

        "summary": model_info["summary"],
        # "layers": "zfNA",
        # "summary": "NrfrerA",
        "default_layer_index": config["extract_layer_default_index"]
    })


def get_info_retrain(inputs):
    model_folder = get_model_folder_path(inputs)

    model_info = config_utils.get_model_info(model_folder, goal=constants.BEFORE_TRAIN)

    label_dataset = get_label_dataset(inputs)
    columns = [c["name"] for c in label_dataset.read_schema()]

    model_config = config_utils.get_config(model_folder)
    # return add_can_use_gpu_to_resp({"summary": "NA SUMMARY TO DISPLAY", "columns": columns, "model_config": model_config})
    # print(model_info)
    return add_can_use_gpu_to_resp({"summary": model_info["summary"], "columns": columns, "model_config": model_config})


def get_model_folder_path(inputs):
    # Retrieving model folder
    model_folder_full_name = get_input_name_from_role(inputs, "model_folder")
    model_folder = dataiku.Folder(model_folder_full_name)

    return model_folder


def get_label_dataset(inputs):
    label_dataset_full_name = get_input_name_from_role(inputs, "label_dataset")
    label_dataset = dataiku.Dataset(label_dataset_full_name)
    return label_dataset


def get_input_name_from_role(inputs, role):
    return [inp for inp in inputs if inp["role"] == role][0]["fullName"]


def add_can_use_gpu_to_resp(response):
    response["can_use_gpu"] = config_utils.can_use_gpu()
    return response
