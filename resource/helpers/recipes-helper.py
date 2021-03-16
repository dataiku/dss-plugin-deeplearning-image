import dataiku
import dku_deeplearning_image.constants as constants
import dku_deeplearning_image.utils as utils
import GPUtil
import json


def do(payload, config, plugin_config, inputs):
    response = {}
    if "method" not in payload:
        return response

    if payload["method"] == "get-info-scoring":
        response = get_info_scoring(inputs)

    if payload["method"] == "get-info-about-model":
        response = get_info_about_model(inputs)

    if payload["method"] == "get-info-retrain":
        response = get_info_retrain(inputs)
    response["gpu_info"] = get_info_gpu()
    add_plugin_id(response)
    return response


def get_gpu_list():
    return [{
        'label': 'GPU:{} - {}'.format(gpu.id, gpu.name, gpu.memoryTotal),
        'value': str(gpu.id)
    } for gpu in GPUtil.getGPUs()]


def get_info_scoring(inputs):
    return {}


def download_json(folder, path):
    return json.loads(folder.get_download_stream(path).read())


def get_model_config(model_folder):
    return download_json(model_folder, constants.CONFIG_FILE)


def get_model_info(model_folder, goal):
    if utils.is_path_in_folder(constants.MODEL_INFO_FILE, model_folder):
        return download_json(model_folder, constants.MODEL_INFO_FILE)[goal]
    else:
        return {"summary": "Not Available before 1st run", "layers": "Not Available before 1st run"}


def get_info_about_model(inputs):
    model_folder = get_model_folder_path(inputs)
    model_info = get_model_info(model_folder, goal=constants.SCORING)
    config = get_model_config(model_folder)

    return {
        "layers": model_info["layers"],
        "summary": model_info["summary"],
        "default_layer_index": config["extract_layer_default_index"]
    }


def get_info_retrain(inputs):
    model_folder = get_model_folder_path(inputs)
    model_info = get_model_info(model_folder, goal=constants.BEFORE_TRAIN)
    label_dataset = get_label_dataset(inputs)
    label_columns = [c["name"] for c in label_dataset.read_schema()]
    model_config = get_model_config(model_folder)
    return {
        "model_summary": model_info["summary"],
        "label_columns": label_columns,
        "model_config": model_config,
        "pooling_options": constants.POOLING_OPTIONS,
        "layers_options": constants.LAYERS_OPTIONS,
        "optimizer_options": constants.OPTIMIZER_OPTIONS}


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


def add_plugin_id(response):
    response['pluginId'] = constants.PLUGIN_ID


def get_info_gpu():
    response = {}
    response["gpu_list"] = get_gpu_list()
    response["can_use_gpu"] = len(response["gpu_list"]) > 0
    response["gpu_usage_choices"] = [
        {
            'label': 'Use all GPU(s)',
            'value': 'all'
        },
        {
            'label': 'Use a custom set of GPU(s)...',
            'value': 'custom'
        }
    ]
    response["gpu_memory_allocation_mode"] = [
        {
            'label': 'No limitation',
            'value': 'all'
        },
        {
            'label': 'Allocate memory automatically',
            'value': constants.GPU_MEMORY_GROWTH
        },
        {
            'label': 'Set a custom memory limit...',
            'value': constants.GPU_MEMORY_LIMIT
        }
    ]
    return response
