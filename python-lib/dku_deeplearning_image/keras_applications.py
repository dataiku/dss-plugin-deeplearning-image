from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocessing
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocessing
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocessing
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocessing

from dku_deeplearning_image.dku_constants import IMAGENET, MODEL


APPLICATIONS = [{
        "name": MODEL.RESNET,
        "label": "Resnet",
        "source": "keras",
        "model_func": ResNet50,
        "preprocessing": resnet50_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            IMAGENET: {
                "top": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": MODEL.XCEPTION,
        "label": "Xception",
        "source": "keras",
        "model_func": Xception,
        "preprocessing": xception_preprocessing,
        "input_shape": (299, 299, 3),
        "weights": {
            IMAGENET: {
                "top": "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            }
        }
    },
    {
        "name": MODEL.INCEPTIONV3,
        "label": "InceptionV3",
        "source": "keras",
        "model_func": InceptionV3,
        "preprocessing": inceptionv3_preprocessing,
        "input_shape": (299, 299, 3),
        "weights": {
            IMAGENET: {
                "top": "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": MODEL.VGG16,
        "label": "VGG16",
        "source": "keras",
        "model_func": VGG16,
        "preprocessing": vgg16_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            IMAGENET: {
                "top": "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    }
]
