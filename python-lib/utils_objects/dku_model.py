import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.constants as constants
from dku_deeplearning_image import APPLICATIONS

from io import StringIO, BytesIO
from utils_objects import DkuFileManager
from utils_objects import DkuApplication

import json
import pandas as pd
import numpy as np
import tables
from keras.layers import Dense
from keras.models import Model
import base64
import PIL
import time
import requests

import copy as cp


class DkuModel(object):
    def __init__(self, folder, is_empty=False):
        self.folder = folder
        files = self.folder.list_paths_in_partition()
        if not is_empty:
            if utils.is_path_in_folder(constants.CONFIG_FILE, self.folder):
                self.load_config()
            else:
                raise IOError(
                    "Error when creating DkuModel. {constants.CONFIG_FILE} should exist in the following list: {files}"
                )

    def jsonify_config(self):
        return {
            'retrained': self.retrained,
            'trained_on': self.trained_on,
            'top_params': self.top_params,
            'extract_layer_default_index': self.extract_layer_default_index,
            'architecture': self.application.jsonify()
        }

    def load_model(self, config, goal):
        strategy = utils.get_tf_strategy()
        include_top = goal == constants.SCORING and not self.retrained
        input_shape = config.get('input_shape', self.get_input_shape())
        self.base_model = self.application.model_func(
            weights=None,
            include_top=include_top,
            input_shape=input_shape
        )
        self.model = cp.deepcopy(self.base_model)
        with strategy.scope():
            self._load_weights_and_enrich(config, goal, include_top)
            self.top_params['input_shape'] = input_shape

    def _load_weights_and_enrich(self, config, goal, include_top):
        # Order of execution of load_weights() and enrich() has to change according to a condition.
        # This way of doing so is more readable like that, it is however not the best for a DRY standpoint
        enrich_kwargs = {
            "pooling": config.get('model_pooling'),
            "dropout": config.get('model_dropout'),
            "reg": config.get('model_reg')
        }
        load_weights_kwargs = {
            "with_top": include_top
        }
        if goal == constants.RETRAINING:
            self.load_weights(**load_weights_kwargs)
            self.enrich(**enrich_kwargs)
        else:
            if self.top_params:
                self.enrich(**enrich_kwargs)
            self.load_weights(**load_weights_kwargs)

    def deepcopy(self, **kwargs):
        new_model = cp.deepcopy(self)
        new_model.update_attributes(**kwargs)
        return new_model

    def update_attributes(self, **kwargs):
        for attr, value in kwargs.items():
            utils.dbg_msg(kwargs, 'kwargs')
            if self.hasattr(attr):
                self.setattr(attr, value)

    def save_label_df(self):
        labels = self.get_distinct_labels()
        df_labels = pd.DataFrame({"id": range(len(labels)), "className": labels})
        DkuFileManager.write_to_folder(
            folder=self.folder,
            file_path=constants.MODEL_LABELS_FILE,
            content=df_labels.to_csv(index=False))

    def save_weights(self):
        # This copies a local file to the managed folder
        model_weights_path = self.get_weights_path()
        with open(model_weights_path, 'rb') as f:
            self.folder.upload_stream(model_weights_path, f)

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
        self.application = self.get_application()

    def get_weights_url(self):
        return self.application.get_weights_url(self.trained_on)

    def save_config(self):
        DkuFileManager.write_to_folder(
            folder=self.folder,
            file_path=constants.CONFIG_FILE,
                content=json.dumps(self.jsonify_config()))

    def get_info(self, base=False):
        return {
            'layers': self.get_layers_as_list(base),
            'summary': self.get_model_summary(base)
        }

    def save_info(self):
        model_info = {
            constants.SCORING: self.get_info(),
            constants.BEFORE_TRAIN: self.get_info(base=True)
        }
        DkuFileManager.write_to_folder(
            folder=self.folder,
            file_path=constants.MODEL_INFO_FILE,
            content=json.dumps(model_info))

    def save_to_folder(self):
        utils.log_info("Starting model saving...")
        self.save_config()
        self.save_label_df()
        self.save_weights()
        self.save_info()
        utils.log_info("Model has been successfully saved.")

    def get_application(self):
        dku_application_params = list(filter(lambda x: x['name'] == self.architecture, APPLICATIONS))
        if not dku_application_params:
            available_apps = [x['name'] for x in APPLICATIONS]
            raise IOError("The application you asked for is not available. Available are : {}.".format(available_apps))
        return DkuApplication(**dku_application_params[0])

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
            utils.log_info("------ \n Info: No csv file in the recipe folder, will not use class names. \n ------")
            label_df = None
        return label_df

    def enrich(self, pooling=None, dropout=None, reg=None, n_classes=None):
        # Init params if not done before
        self.top_params['pooling'] = pooling or self.top_params.get('pooling')
        self.top_params['n_classes'] = n_classes or len(self.get_distinct_labels())

        x = self.model.layers[-1].output
        x = utils.add_pooling(x, self.top_params['pooling'])
        x = utils.add_dropout(x, dropout)

        regularizer = utils.get_regularizer(reg)

        predictions = Dense(self.top_params['n_classes'], activation='softmax', name='predictions',
                            kernel_regularizer=regularizer)(x)
        self.model = Model(input=self.model.input, output=predictions)

    def score_b64_image(self, img_b64, **kwargs):
        img_b64_decode = base64.b64decode(img_b64)
        image = BytesIO(img_b64_decode)
        return self.score([image], **kwargs)

    def score_image_folder(self, images_folder, **kwargs):
        images_paths = images_folder.list_paths_in_partition()
        images = []
        for path in images_paths:
            try:
                images.append(images_folder.get_download_stream(path))
            except IOError as e:
                utils.log_warning("Cannot read the image '{}', skipping it. Error: {}".format(path, e))
                images.append(None)
        return self.score(images, **kwargs)

    def score(self, images, limit=constants.DEFAULT_PRED_LIMIT,
              min_threshold=constants.DEFAULT_PRED_MIN_THRESHOLD, classify=True):
        batch_size = constants.PREDICTION_BATCH_SIZE
        n = 0
        results = {"prediction": [], "error": []}
        num_images = len(images)
        while True:
            if (n * batch_size) >= num_images: break
            next_batch_list, error_indices = [], []
            for index_in_batch, i in enumerate(range(n * batch_size, min((n + 1) * batch_size, num_images))):
                image = images[i]
                preprocessed_img = utils.preprocess_img(
                    img_path=image,
                    img_shape=self.get_input_shape(),
                    preprocessing=self.application.preprocessing
                ) if image else None
                if preprocessed_img is None:
                    error_indices.append(index_in_batch)
                else:
                    next_batch_list.append(preprocessed_img)

            next_batch = np.array(next_batch_list)

            prediction_batch = self.get_predictions_for_batch(next_batch, classify, limit, min_threshold)
            error_batch = [0] * len(prediction_batch)

            for err_index in error_indices:
                prediction_batch.insert(err_index, None)
                error_batch.insert(err_index, 1)

            results["prediction"].extend(prediction_batch)
            results["error"].extend(error_batch)
            n += 1
            utils.log_info("{} images treated, out of {}".format(min(n * batch_size, num_images), num_images))
        return results

    def get_predictions_for_batch(self, batch, classify, limit, min_threshold):
        if not batch.size:
            return []
        return utils.get_predictions(
                model=self.get_model(),
                batch=batch,
                classify=classify,
                limit=limit,
                min_threshold=min_threshold,
                labels_df=self.get_label_df()
        )

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

    def download_from_web(self, cb):
        # Downloading weights
        url_to_weights = self.get_weights_url()

        def update_percent(percent, last_update_time):
            new_time = time.time()
            if (new_time - last_update_time) > 3:
                cb(percent)
                return new_time
            else:
                return last_update_time

        def download_files_to_managed_folder(output_f, files_info, chunk_size=8192):
            total_size = 0
            bytes_so_far = 0
            for file_info in files_info:
                response = requests.get(file_info["url"], stream=True)
                total_size += int(response.headers.get('content-length'))
                file_info["response"] = response
            update_time = time.time()
            for file_info in files_info:
                with output_f.get_writer(file_info["filename"]) as f:
                    for content in file_info["response"].iter_content(chunk_size=chunk_size):
                        bytes_so_far += len(content)
                        # Only scale to 80% because needs to compute model summary after download
                        percent = int(float(bytes_so_far) / total_size * 80)
                        update_time = update_percent(percent, update_time)
                        f.write(content)

        if self.trained_on == constants.IMAGENET:
            # Downloading mapping id <-> name for imagenet classes
            # File used by Keras in all its 'decode_predictions' methods
            # Found here : https://github.com/keras-team/keras/blob/2.1.1/keras/applications/imagenet_utils.py
            class_mapping_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        else:
            class_mapping_url = ''

        files_to_dl = [
            {"url": url_to_weights["top"], "filename": self.get_weights_path(with_top=True)},
            {"url": url_to_weights["no_top"], "filename": self.get_weights_path(with_top=False)}
        ]

        if class_mapping_url:
            files_to_dl.append({"url": class_mapping_url, "filename": constants.CLASSES_MAPPING_FILE})

        self.folder.upload_data(constants.CONFIG_FILE, json.dumps(self.jsonify_config()).encode('utf-8'))
        download_files_to_managed_folder(self.folder, files_to_dl)

        if class_mapping_url:
            mapping_df = pd.read_json(self.folder.get_download_stream(constants.CLASSES_MAPPING_FILE), orient="index")
            mapping_df = mapping_df.reset_index()
            mapping_df = mapping_df.rename(columns={"index": "id", 1: "className"})[["id", "className"]]
            DkuFileManager.write_to_folder(
                folder=self.folder,
                file_path=constants.MODEL_LABELS_FILE,
                content=mapping_df.to_csv(index=False, sep=","))
            self.folder.delete_path(constants.CLASSES_MAPPING_FILE)

        self.load_model({}, constants.SCORING)
        self.save_info()

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
