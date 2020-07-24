# -*- coding: utf-8 -*-

import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.constants as constants
from io import StringIO, BytesIO
import json
import pandas as pd
import numpy as np
import tables
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model

import copy as cp

class DkuModel:
    def __init__(self, folder):
        self.folder = folder
        if '/{}'.format(constants.CONFIG_FILE) in self.folder.list_paths_in_partition():
            self.load_config()

    def jsonify_config(self):
        return {
            'retrained': self.retrained,
            'trained_on': self.trained_on,
            'top_params': self.top_params,
            'extract_layer_default_index': self.extract_layer_default_index,
            'architecture': self.application.jsonify()
        }
    
    @tf.function
    def test_decorated_function(self, **kwargs):
        return self.application.model_func(**kwargs)

    def load_model(self, config, goal, use_gpu=False, n_gpu=None):
        strategy = tf.distribute.MirroredStrategy()
        include_top = goal == constants.SCORING and not self.retrained
        input_shape = config.get('input_shape', self.get_input_shape())
        self.model = self.application.model_func(
            weights=None,
            include_top=include_top,
            input_shape=input_shape
        )
        with strategy.scope():
            

            if goal == constants.RETRAINING:
                self.load_weights(with_top=include_top)
            
            if self.top_params or goal == constants.RETRAINING:
                self.enrich(
                    pooling=config.get('model_pooling'),
                    dropout=config.get('model_dropout'),
                    reg=config.get('model_reg')
                )

            if goal != constants.RETRAINING:
                self.load_weights(with_top=include_top)


            self.top_params['input_shape'] = input_shape

        if use_gpu and n_gpu:
            self.base_model = cp.deepcopy(self.model)
            #self.model = multi_gpu_model(self.model, n_gpu)

    def get_base_model(self):
        return self.getattr('base_model', self.model)

    def truncate_output(self, layer_index):
        self.model = Model(inputs=self.model.input, outputs=self.model.layers[layer_index].output)

    def load_weights(self, with_top=False):
        weights_path = self.get_weights_path(with_top=with_top)
        self.load_weights_to_local(weights_path)
        self.model.load_weights(weights_path)

    def get_model(self):
        assert self.model, "You must load the recipe before getting it. Killing process."
        return self.model

    def get_input_shape(self):
        return self.application.input_shape or self.model.input_shape or (224, 224, 3)

    def get_layers_as_list(self, base=False):
        model = self.get_base_model() if base else self.get_model()
        return [layer.__class__.__name__ for layer in model.layers]

    def get_model_summary(self, base=False):
        model = self.get_base_model() if base else self.get_model()
        summary_io = StringIO()
        model.summary(print_fn=lambda line: summary_io.write(line + "\n"))
        return summary_io.getvalue()

    def load_config(self):
        config = json.loads(self.folder.get_download_stream(constants.CONFIG_FILE).read())
        self.set_config(config)

    def set_config(self, config):
        self.retrained = config.get('retrained', False)
        self.trained_on = config.get('trained_on')
        self.top_params = config.get('top_params', {})
        self.architecture = config.get('architecture')
        self.extract_layer_default_index = config.get('extract_layer_default_index', -1)

        self.check_mandatory_attrs(['trained_on', 'architecture'])
        self.application = utils.get_application(self.architecture)

    def get_weights_url(self):
        return self.application.get_weights_url(self.trained_on)

    def save_config(self):
        with self.folder.get_writer(constants.CONFIG_FILE) as w:
            w.write(json.dumps(self.jsonify_config()))

    def get_info(self, base=False):
        return {
            'layers': self.get_layers_as_list(base),
            'summary': self.get_model_summary(base)
        }

    def save_info(self):
        with self.folder.get_writer(constants.MODEL_INFO_FILE) as w:
            w.write(json.dumps(self.get_info()))

    def get_or_load(self, attr, default):
        if not self.hasattr(attr):
            self.setattr(attr, default)
        return self.getattr(attr)

    def get_distinct_labels(self):
        label_df = self.get_label_df()
        return self.get_or_load('distinct_labels', list(np.unique(label_df[constants.LABEL])))

    def get_label_df(self):
        return self.get_or_load('label_df', self.build_label_df())

    def build_label_df(self):
        details_model_label = self.folder.get_path_details(constants.MODEL_LABELS_FILE)
        if details_model_label['exists'] and not details_model_label["directory"]:
            labels_path = self.folder.get_download_stream(constants.MODEL_LABELS_FILE)
            label_df = pd.read_csv(labels_path, sep=",").set_index('id').rename({'className': constants.LABEL}, axis=1)
        else:
            print("------ \n Info: No csv file in the recipe folder, will not use class names. \n ------")
            label_df = None
        return label_df

    def enrich(self, pooling=None, dropout=None, reg=None, n_classes=None):
        # Init params if not done before
        self.top_params['pooling'] = pooling or self.top_params.get('pooling')
        self.top_params['n_classes'] = n_classes or len(self.get_distinct_labels())
        x = self.get_base_model().layers[-1].output

        x = utils.add_pooling(x, self.top_params['pooling'])
        x = utils.add_dropout(x, dropout)

        regularizer = utils.get_regularizer(reg)

        predictions = Dense(self.top_params['n_classes'], activation='softmax', name='predictions',
                            kernel_regularizer=regularizer)(x)
        self.model = Model(input=self.model.input, output=predictions)

    def score(self, images_folder, limit=constants.DEFAULT_PRED_LIMIT, min_threshold=0, classify=True):
        batch_size = constants.PREDICTION_BATCH_SIZE
        images_paths = images_folder.list_paths_in_partition()
        n = 0
        results = {"prediction": [], "error": []}
        num_images = len(images_paths)
        while True:
            if (n * batch_size) >= num_images: break
            next_batch_list, error_indices = [], []
            for index_in_batch, i in enumerate(range(n * batch_size, min((n + 1) * batch_size, num_images))):
                img_path = images_paths[i]
                try:
                    preprocessed_img = utils.preprocess_img(
                        img_path=images_folder.get_download_stream(img_path),
                        img_shape=self.get_input_shape(),
                        preprocessing=self.application.preprocessing
                    )
                    next_batch_list.append(preprocessed_img)
                except IOError as e:
                    print("Cannot read the image '{}', skipping it. Error: {}".format(img_path, e))
                    error_indices.append(index_in_batch)
            next_batch = np.array(next_batch_list)

            prediction_batch = utils.get_predictions(
                model=self.get_model(),
                batch=next_batch,
                classify=classify,
                limit=limit,
                min_threshold=min_threshold,
                labels_df=self.get_label_df()
            )
            error_batch = [0] * len(prediction_batch)

            for err_index in error_indices:
                prediction_batch.insert(err_index, None)
                error_batch.insert(err_index, 1)

            results["prediction"].extend(prediction_batch)
            results["error"].extend(error_batch)
            n += 1
            print("{} images treated, out of {}".format(min(n * batch_size, num_images), num_images))
        return results

    def get_weights_path(self, with_top=False):
        weights_filename = utils.get_weights_filename(with_top)
        return weights_filename

    def load_weights_to_local(self, weights_path):
        model_weights_path = self.folder.get_download_stream(weights_path)
        # Hack to get the H5 stream coming from the folder API as a file
        # Hack inspired from https://stackoverflow.com/questions/16654251/can-h5py-load-a-file-from-a-byte-array-in-memory
        h5file = tables.open_file(weights_path + "_temp", driver="H5FD_CORE",
                                  driver_core_image=model_weights_path.read(),
                                  driver_core_backing_store=0)
        h5file.copy_file(weights_path, overwrite=True)

    def get_layers(self):
        return self.get_model().layers

    def print_summary(self):
        self.get_model().summary()

    def compile(self, **kwargs):
        self.get_model().compile(**kwargs)

    def fit_generator(self, **kwargs):
        self.model.fit_generator(**kwargs)

    def setattrs(self, d):
        for k, v in d.items():
            self.setattr(k, v)

    def hasattr(self, att):
        return hasattr(self, att)

    def setattr(self, name, val):
        setattr(self, name, val)

    def getattr(self, attr, default=None):
        return getattr(self, attr, default)

    def check_mandatory_attrs(self, mandatory_attrs):
        for attr in mandatory_attrs:
            if not self.hasattr(attr):
                raise IOError('Argument {} is missing. Killing process.'.format(str(attr)))
