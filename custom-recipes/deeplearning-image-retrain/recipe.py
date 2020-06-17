from dataiku.customrecipe import *
import dl_image_toolbox_utils as utils
import config_utils as config_utils
from sklearn.model_selection import train_test_split
from keras import optimizers, initializers, metrics, regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dropout, Dense
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
import constants
import math
import shutil
import numpy as np
import dataiku


###################################################################################################################
## LOADING ALL REQUIRED INFO AND 
##      SETTING VARIABLES
###################################################################################################################
def load_recipe_config(config):
    recipe_config = get_recipe_config()

    config.should_use_gpu = recipe_config.get('should_use_gpu', False)
    config.list_gpu = recipe_config["list_gpu"]
    config.gpu_allocation = recipe_config["gpu_allocation"]
    config.train_ratio = float(recipe_config["train_ratio"])
    config.input_shape = (int(recipe_config["image_width"]), int(recipe_config["image_height"]), 3)
    config.batch_size = int(recipe_config["batch_size"])
    config.model_pooling = recipe_config["model_pooling"]
    config.model_reg = recipe_config["model_reg"]
    config.model_dropout = float(recipe_config["model_dropout"])
    config.layer_to_retrain = recipe_config["layer_to_retrain"]
    config.layer_to_retrain_n = int(recipe_config.get('layer_to_retrain_n', 0))
    config.optimizer = recipe_config["model_optimizer"]
    config.learning_rate = recipe_config["model_learning_rate"]
    config.custom_params_opti = recipe_config.get("model_custom_params_opti", [])
    config.nb_epochs = int(recipe_config["nb_epochs"])
    config.nb_steps_per_epoch = int(recipe_config["nb_steps_per_epoch"])
    config.nb_validation_steps = int(recipe_config["nb_validation_steps"])
    config.data_augmentation = recipe_config["data_augmentation"]
    config.n_augmentation = int(recipe_config["n_augmentation"])
    config.custom_params_data_augment = recipe_config.get("model_custom_params_data_augmentation", [])
    config.tensorboard = recipe_config["tensorboard"]
    config.random_seed = int(recipe_config["random_seed"])
    config.gpu_options = utils.load_gpu_options(config.should_use_gpu, config.list_gpu, config.gpu_allocation)
    config.n_gpu = config.gpu_options.get("n_gpu", 0)


def load_input_output(config):
    image_folder_input_name = get_input_names_for_role('image_folder')[0]
    config.image_folder = dataiku.Folder(image_folder_input_name)

    model_folder_input_name = get_input_names_for_role('model_folder')[0]
    config.model_folder = dataiku.Folder(model_folder_input_name)

    output_model_folder_name = get_output_names_for_role('model_output')[0]
    config.output_model_folder = dataiku.Folder(output_model_folder_name)


def load_label_df(config):
    recipe_config = get_recipe_config()

    label_dataset_input_name = get_input_names_for_role('label_dataset')[0]
    config.label_dataset = dataiku.Dataset(label_dataset_input_name)
    renaming_mapping = {
        recipe_config["col_filename"]: constants.FILENAME,
        recipe_config["col_label"]: constants.LABEL
    }
    config.label_df = config.label_dataset.get_dataframe().rename(columns=renaming_mapping)[renaming_mapping.values()]
    config.labels = list(np.unique(config.label_df[constants.LABEL]))
    config.n_classes = len(config.labels)


def display_gpu_device():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


def load_config():
    config = utils.AttributeDict()

    load_recipe_config(config)
    load_input_output(config)
    load_label_df(config)

    display_gpu_device()

    return config


def load_model_config(model_folder):
    # Model config
    utils.save_model_info(model_folder)
    model_config = config_utils.get_config(model_folder)
    return model_config


###################################################################################################################
## BUILD TRAIN/TEST SETS
###################################################################################################################
def build_train_test_set(label_df, train_ratio, random_seed):
    train_df, test_df = train_test_split(label_df, stratify=label_df[constants.LABEL], train_size=train_ratio,
                                         random_state=random_seed)
    return train_df, test_df


###################################################################################################################
## LOAD MODEL
###################################################################################################################

# Loading pre-trained model
def get_model_and_pp(config):
    model_and_pp = utils.load_instantiate_keras_model_preprocessing(
        config.model_folder,
        goal=constants.RETRAINING,
        input_shape=config.input_shape,
        pooling=config.model_pooling,
        reg=config.model_reg,
        dropout=config.model_dropout,
        n_classes=config.n_classes)
    return model_and_pp


def set_trainable_layers(model, layer_to_retrain, layer_to_retrain_n=None):
    # CHOOSING LAYER TO RETRAIN
    print("Will Retrain layer(s) with mode: {}".format(layer_to_retrain))
    if layer_to_retrain == "all":
        for lay in model.layers:
            lay.trainable = True

    elif layer_to_retrain == "last":
        for lay in model.layers[:-1]:
            lay.trainable = False
        lay = model.layers[-1]
        lay.trainable = True

    elif layer_to_retrain == "n_last":
        n_last = layer_to_retrain_n
        for lay in model.layers[:-n_last]:
            lay.trainable = False
        for lay in model.layers[-n_last:]:
            lay.trainable = True


def load_model_with_gpu(config):
    with tf.device('/cpu:0'):
        config.model_and_pp = get_model_and_pp(config)
    config.model_and_pp['model'] = multi_gpu_model(config.model_and_pp['model'], config.n_gpu)


def load_model_without_gpu(config):
    config.model_and_pp = get_model_and_pp(config)


def load_model(config):
    load_model_with_gpu(config) if config.should_use_gpu and config.n_gpu > 1 else load_model_without_gpu(config)
    set_trainable_layers(config.model_and_pp['model'], config.layer_to_retrain, config.layer_to_retrain_n)
    config.model_and_pp['model'].summary()


###################################################################################################################
## BUILD GENERATORS
## Info: Generators must loop infinitely, each loop yielding the batches of preprocessed data.
##       It will be used at each epoch, hence the infinite loop.
###################################################################################################################


@utils.threadsafe_generator
def images_generator(image_df, image_folder, batch_size, input_shape, labels, preprocessing, n_classes,
                     use_augmentation=False, extra_images_gen=None, n_augmentation=None):
    n_images = image_df.shape[0]
    batch_size_adapted = int(batch_size / n_augmentation) if use_augmentation else batch_size
    n_batch = int(math.ceil(n_images * 1.0 / batch_size_adapted))

    while True:

        for num_batch in range(n_batch):

            images_df_batch = image_df.iloc[num_batch * batch_size_adapted: (num_batch + 1) * batch_size_adapted, :]
            n_images_batch = images_df_batch.shape[0]

            X_batch_list = []
            y_batch_list = []

            for num_img in range(n_images_batch):
                row = images_df_batch.iloc[num_img, :]
                img_filename = row[constants.FILENAME]
                label = row[constants.LABEL]
                label_index = labels.index(label)
                try:
                    x = utils.preprocess_img(
                        utils.get_cached_file_from_folder(image_folder, img_filename), input_shape, preprocessing)
                    if use_augmentation:
                        x = np.tile(x, (n_augmentation, 1, 1, 1))
                        X_batch = next(extra_images_gen.flow(x, batch_size=n_augmentation))
                        y_batch = [label_index] * n_augmentation
                    else:
                        X_batch = [x]
                        y_batch = [label_index]
                    X_batch_list.extend(X_batch)
                    y_batch_list.extend(y_batch)
                except IOError as e:
                    print("Cannot read the image '{}', skipping it. Error: {}".format(img_filename, e))

            X_batch = np.array(X_batch_list)

            actual_batch_size = X_batch.shape[0]
            y_batch = np.zeros((actual_batch_size, n_classes))
            y_batch[range(actual_batch_size), y_batch_list] = 1

            yield X_batch, y_batch


def load_train_test_generator(train_df, test_df, config):
    extra_images_gen = None
    if config.data_augmentation:
        print("Using data augmentation with {} images generated per training image\n".format(config.n_augmentation))
        params_data_augment = utils.clean_custom_params(
            custom_params=config.custom_params_data_augment,
            params_type="Data Augmentation"
        )
        extra_images_gen = ImageDataGenerator(**params_data_augment)
    get_images_gen = lambda images_df: images_generator(
        images_df=images_df,
        images_folder=config.image_folder,
        batch_size=config.batch_size,
        input_shape=config.input_shape,
        labels=config.labels,
        preprocessing=config.model_and_pp['preprocessing'],
        n_classes=config.n_classes,
        use_augmentation=config.data_augmentation,
        extra_images_gen=extra_images_gen,
        n_augmentation=config.n_augmentation
    )
    train_gen = get_images_gen(train_df)
    test_gen = get_images_gen(test_df)
    return train_gen, test_gen


###################################################################################################################
## COMPILE MODEL
###################################################################################################################


if optimizer == "adam":
    model_opti_class = optimizers.Adam
elif optimizer == "adagrad":
    model_opti_class = optimizers.Adagrad
elif optimizer == "sgd":
    model_opti_class = optimizers.SGD

# Cleaning custom parameters
params_opti = utils.clean_custom_params(custom_params_opti)
params_opti["lr"] = learning_rate

model_opti = model_opti_class(**params_opti)
model.compile(optimizer=model_opti, loss='categorical_crossentropy', metrics=['accuracy'])

callback_list = []

###################################################################################################################
## BUILD MODEL CHECKPOINT
###################################################################################################################

model_weights_path = utils.get_weights_path(output_model_folder, model_config, suffix=constants.RETRAINED_SUFFIX,
                                            should_exist=False)
should_save_weights_only = utils.should_save_weights_only(model_config)

if should_use_gpu and n_gpu > 1:
    mcheck = MultiGPUModelCheckpoint(model_weights_path, base_model, monitor="val_loss", save_best_only=True,
                                     save_weights_only=should_save_weights_only)
else:
    mcheck = ModelCheckpoint(model_weights_path, monitor="val_loss", save_best_only=True,
                             save_weights_only=should_save_weights_only)

callback_list.append(mcheck)

###################################################################################################################
## TENSORBOARD
###################################################################################################################

if tensorboard:
    log_path = utils.get_file_path(config.output_model_folder.get_path(), constants.TENSORBOARD_LOGS)

    # If already folder at loger_path, delete it
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)

    tsboard = TensorBoard(log_dir=log_path, write_graph=True)
    callback_list.append(tsboard)

###################################################################################################################
## TRAIN MODEL
###################################################################################################################


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_steps_per_epoch,
    epochs=nb_epochs,
    validation_data=test_generator,
    validation_steps=nb_validation_steps,
    callbacks=callback_list,
    shuffle=False,
    verbose=2)

###################################################################################################################
## SAVING NEW CONFIG AND LABELS
###################################################################################################################

model_config[constants.RETRAINED] = True
model_config[constants.TOP_PARAMS] = model_params
utils.write_config(output_model_folder, model_config)

df_labels = pd.DataFrame({"id": range(n_classes), "className": labels})
with output_model_folder.get_writer(constants.MODEL_LABELS_FILE) as w:
    w.write((df_labels.to_csv(index=False)))
# df_labels.to_csv(utils.get_file_path(output_model_folder_path, constants.MODEL_LABELS_FILE), index=False)


# This copies a local file to the managed folder
with open(model_weights_path) as f:
    output_model_folder.upload_stream(model_weights_path, f)
# Computing model info
utils.save_model_info(output_model_folder)


###############################################################
## MODEL CHECKPOINT FOR MULTI GPU
## When using multiple GPUs, we need to save the base model,
## not the one defined by multi_gpu_model
## see example: https://keras.io/utils/#multi_gpu_model
## Therefore, to save the model after each epoch by leveraging 
## ModelCheckpoint callback, we need to adapt it to save the 
## base model. To do so, we pass the base model to the callback.
## Inspired by: 
##   https://github.com/keras-team/keras/issues/8463#issuecomment-345914612
###############################################################

class MultiGPUModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__(filepath,
                                                      monitor=monitor,
                                                      verbose=verbose,
                                                      save_best_only=save_best_only,
                                                      save_weights_only=save_weights_only,
                                                      mode=mode,
                                                      period=period)
        self.base_model = base_model

    def on_epoch_end(self, epoch, logs=None):
        # Must behave like ModelCheckpoint on_epoch_end but save base_model instead

        # First retrieve model
        model = self.model

        # Then switching model to base model
        self.model = self.base_model

        # Calling super on_epoch_end
        super(MultiGPUModelCheckpoint, self).on_epoch_end(epoch, logs)

        # Resetting model afterwards
        self.model = model
