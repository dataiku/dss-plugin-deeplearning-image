from enum import Enum

# Plugin ID
PLUGIN_ID = 'deeplearning-image'


class MODEL(Enum):
    RESNET = "resnet"
    XCEPTION = "xception"
    INCEPTIONV3 = "inception_v3"
    VGG16 = "vgg16"
    INCEPTION_RESNET = "inception_resnet"
    MOBILENET = "mobilenet"
    DENSENET = "densenet"
    NASNET_LARGE = "nasnet_large"
    NASNET_MOBILE = "nasnet_mobile"


# Goals
class GOAL(Enum):
    SCORE = "score"
    RETRAIN = "retrain"
    EXTRACT = "extract"
    API_DESIGNER = "api_designer"
    BEFORE_TRAIN = "before_train"


# Options for custom form
class GPU_MEMORY(Enum):
    LIMIT = "memory_limit"
    GROWTH = "memory_growth"
    NO_LIMIT = "all"


POOLING_OPTIONS = [
    ["No pooling", "None"],
    ["Average", "avg"],
    ["Maximum", "max"]
]

LAYERS_OPTIONS = [
    ["Last layer", "last"],
    ["All layers", "all"],
    ["N last layers", "n_last"]
]

OPTIMIZER_OPTIONS = [
    ["Adam", "adam"],
    ["Adagrad", "adagrad"],
    ["SGD", "sgd"]
]


# Images banks on which model are pre-trained
IMAGENET = "imagenet"
IMAGENET_LABEL = "ImageNet"
IMAGENET_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

# Filenames
CONFIG_FILE = "config.json"
MODEL_INFO_FILE = "model_info.json"
CLASSES_MAPPING_FILE = "classes_mapping.json"
MODEL_LABELS_FILE = "model_labels.csv"
WEIGHT_FILENAME = 'model_weights'
TENSORBOARD_LOGS = "tensorboard_logs"

# Default values
PREDICTION_BATCH_SIZE = 100
COMPILE_LOSS_FUNCTION = 'categorical_crossentropy'
COMPILE_METRICS = ["accuracy"]

# Models Parameters
MOBILENET_ALPHA = "1.0"
MOBILENET_ROWS = "224"

# Other
NOTOP_SUFFIX = "_notop"
LABEL = "__dku__image_label"
FILENAME = "__dku__image_filename"
ENV_NAME = 'plugin_dl-image_api-node'
AUTOTUNE = -1
