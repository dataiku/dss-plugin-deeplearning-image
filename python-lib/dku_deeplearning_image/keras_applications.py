from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocessing
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocessing
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocessing
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocessing
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as MobileNet, preprocess_input as mobilenet_preprocessing
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet_preprocessing
from tensorflow.keras.applications.densenet import DenseNet201 as DenseNet, preprocess_input as densenet_preprocessing
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_large_preprocessing
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input as nasnet_mobile_preprocessing

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
    },
    {
        "name": constants.MOBILENET,
        "label": constants.MOBILENET_LABEL,
        "source": "keras",
        "model_func": MobileNet,
        "preprocessing": mobilenet_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            constants.IMAGENET: {
                "top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_{constants.MOBILENET_ALPHA}_{constants.MOBILENET_ROWS}_tf.h5",
                "no_top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_{constants.MOBILENET_ALPHA}_{constants.MOBILENET_ROWS}_tf_no_top.h5"
            }
        }
    },
    {
        "name": constants.INCEPTION_RESNET,
        "label": constants.INCEPTION_RESNET_LABEL,
        "source": "keras",
        "model_func": InceptionResNetV2,
        "preprocessing": inception_resnet_preprocessing,
        "input_shape": (299, 299, 3),
        "weights": {
            constants.IMAGENET: {
                "top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": constants.DENSENET,
        "label": constants.DENSENET_LABEL,
        "source": "keras",
        "model_func": DenseNet,
        "preprocessing": densenet_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            constants.IMAGENET: {
                "top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5",
                "no_top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
            }
        }
    },
    {
        "name": constants.NASNET_LARGE,
        "label": constants.NASNET_LARGE_LABEL,
        "source": "keras",
        "model_func": NASNetLarge,
        "preprocessing": nasnet_large_preprocessing,
        "input_shape": (331, 331, 3),
        "weights": {
            constants.IMAGENET: {
                "top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5",
                "no_top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large-no-top.h5"
            }
        }
    },
    {
        "name": constants.NASNET_MOBILE,
        "label": constants.NASNET_MOBILE_LABEL,
        "source": "keras",
        "model_func": NASNetMobile,
        "preprocessing": nasnet_mobile_preprocessing,
        "input_shape": (224, 224, 3),
        "weights": {
            constants.IMAGENET: {
                "top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile.h5",
                "no_top": f"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile-no-top.h5"
            }
        }
    }
]
