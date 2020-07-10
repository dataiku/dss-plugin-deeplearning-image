import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.constants as constants
import StringIO
import json
import pandas as pd
import numpy as np
import tables

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout, Dense
from keras.models import Model
from keras import regularizers

class DkuModel:
    def __init__(self, folder):
        self.folder = folder

    def jsonify_config(self):
        return {
            'retrained': self.retrained,
            'trained_on': self.trained_on,
            'top_params': self.top_params,
            'extract_layer_default_index': self.extract_layer_default_index,
            'application': self.application.jsonify()
        }

    def load_model(self, **kwargs):
        self.model = self.application.model_func(**kwargs)

    def get_model(self):
        assert self.model, "You must load the model before getting it. Killing process."
        return self.model

    def get_input_shape(self):
        return self.application.input_shape or self.model.input_shape or (224, 224)

    def get_layers_as_list(self):
        return [layer.__class__.__name__ for layer in self.get_model().layers]

    def get_model_summary(self):
        summary_io = StringIO.StringIO()
        self.model.summary(print_fn=lambda line: summary_io.write(line + "\n"))
        return summary_io.getvalue()

    def load_config(self):
        config = json.loads(self.folder.get_download_stream(constants.CONFIG_FILE).read())
        self.setattrs(config)
        self.check_mandatory_attrs(['trained_on', 'architecture'])
        self.application = utils.get_application(config.get('architecture'))

    def save_config(self):
        with self.folder.get_writer(constants.CONFIG_FILE) as w:
            w.write(json.dumps(self.jsonify_config()))

    def get_info(self):
        return {
            'layers': self.get_layers_as_list(),
            'summary': self.get_model_summary()
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
            self.label_df = pd.read_csv(labels_path, sep=",").set_index('id')
        else:
            print("------ \n Info: No csv file in the model folder, will not use class names. \n ------")
            self.label_df = None

    def enrich(self, pooling=None, dropout=None, reg=None, verbose=False):
        # Init params if not done before
        pooling = self.getattr('pooling', pooling)
        n_classes = len(self.get_distinct_labels())
        x = self.model.layers[-1].output

        x = utils.add_pooling(x, pooling)
        x = utils.add_dropout(x, dropout)
        regularizer = utils.get_regularizer(reg)

        predictions = Dense(n_classes, activation='softmax', name='predictions', kernel_regularizer=regularizer)(x)
        self.model = Model(input=self.model.input, output=predictions)

    def get_weights(self):
        weights_filename = utils.get_weights_filename()

    def load_weights_to_local(self):
        weights_filename = utils.get_weights_filename()
        model_weights_path = self.folder.get_download_stream(weights_filename)
        # Hack to get the H5 stream coming from the folder API as a file
        # Hack inspired from https://stackoverflow.com/questions/16654251/can-h5py-load-a-file-from-a-byte-array-in-memory
        h5file = tables.open_file(weights_filename + "_temp", driver="H5FD_CORE",
                                  driver_core_image=model_weights_path.read(),
                                  driver_core_backing_store=0)
        h5file.copy_file(weights_filename, overwrite=True)

    def setattrs(self, d):
        for k, v in d.enumerate():
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
