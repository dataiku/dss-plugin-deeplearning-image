PLUGIN_ID = 'deeplearning-image-v2'

RESNET = "resnet"
RESNET_LABEL = "Resnet"
XCEPTION = "xception"
XCEPTION_LABEL = "Xception"
INCEPTIONV3 = "inceptionv3"
INCEPTIONV3_LABEL = "Inception V3"
VGG16 = "vgg16"
VGG16_LABEL = "VGG16"
IMAGENET = "imagenet"
IMAGENET_LABEL = "ImageNet"
CONFIG_FILE = "config.json"
MODEL_INFO_FILE = "model_info.json"
CLASSES_MAPPING_FILE = "classes_mapping.json"
SCORING = "scoring"
RETRAINING = "retraining"
BEFORE_TRAIN = "before_train"
NOTOP_SUFFIX = "_notop"
TENSORBOARD_LOGS = "tensorboard_logs"
LABEL = "__dku__image_label"
FILENAME = "__dku__image_filename"
MODEL_LABELS_FILE = "model_labels.csv"
ENV_NAME = 'plugin_dl-image_api-node'
WEIGHT_FILENAME = 'model_weights'
PREDICTION_BATCH_SIZE = 100
DEFAULT_PRED_LIMIT = 5
DEFAULT_PRED_MIN_THRESHOLD = 0
COMPILE_LOSS_FUNCTION = 'categorical_crossentropy'
COMPILE_METRICS = ["accuracy"]

GPU_MEMORY_LIMIT = "memory_limit"
GPU_MEMORY_GROWTH = "memory_growth"

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
