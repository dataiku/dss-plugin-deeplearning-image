from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocessing
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocessing
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocessing
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocessing

import dku_deeplearning_image.dku_constants as constants


APPLICATIONS = [{
        "name": constants.RESNET,
        "label": constants.RESNET_LABEL,
        "source": "keras",
        "model_func": ResNet50,
        "preprocessing": resnet50_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": constants.XCEPTION,
        "label": constants.XCEPTION_LABEL,
        "source": "keras",
        "model_func": Xception,
        "preprocessing": xception_preprocessing,
        "input_shape": (299, 299, 3),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            }
        }
    },
    {
        "name": constants.INCEPTIONV3,
        "label": constants.INCEPTIONV3_LABEL,
        "source": "keras",
        "model_func": InceptionV3,
        "preprocessing": inceptionv3_preprocessing,
        "input_shape": (299, 299, 3),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": constants.VGG16,
        "label": constants.VGG16_LABEL,
        "source": "keras",
        "model_func": VGG16,
        "preprocessing": vgg16_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            constants.IMAGENET: {
                "top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
}]