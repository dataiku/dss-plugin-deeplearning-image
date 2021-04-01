from .dku_recipe import DkuRecipe
from dku_deeplearning_image.misc_objects import DkuModel
import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.dku_constants as constants
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import shutil
import numpy as np

import logging
logger = logging.getLogger(__name__)


class RetrainRecipe(DkuRecipe):
    def __init__(self, config):
        super(RetrainRecipe, self).__init__(config)

    def load_dku_model(self, model_folder, label_df):
        self.dku_model = DkuModel(model_folder)
        self.dku_model.label_df = label_df
        self.dku_model.load_model(self.config, constants.GOAL.RETRAIN)
        self._set_trainable_layers()

    def _set_trainable_layers(self):
        logger.info("Will Retrain layer(s) with mode: {}".format(self.config.layer_to_retrain))
        layers = self.dku_model.get_layers()
        if self.config.layer_to_retrain == "all":
            n_last = len(layers)
        elif self.config.layer_to_retrain == "last":
            n_last = 1
        elif self.config.layer_to_retrain == "n_last":
            n_last = self.config.layer_to_retrain_n
        else:
            raise ValueError("Error in # layers to retrain. You must retrain at least one layer.")

        for i, lay in enumerate(layers):
            lay.trainable = i >= (len(layers) - n_last)

    def _build_train_test_sets(self, label_df):
        train_df, test_df = train_test_split(
            label_df,
            stratify=label_df[constants.LABEL],
            train_size=self.config.train_ratio,
            random_state=self.config.random_seed)
        return train_df, test_df

    def _get_tf_image_data_gen(self):
        logger.info("Using data augmentation with {} images generated per training image\n".format(
            self.config.n_augmentation))
        params_data_augment = utils.clean_custom_params(
            custom_params=self.config.custom_params_data_augment,
            params_type="Data Augmentation"
        )
        return ImageDataGenerator(**params_data_augment)

    def _get_optimizer_class(self):
        if self.config.optimizer == "adam":
            model_opti_class = optimizers.Adam
        elif self.config.optimizer == "adagrad":
            model_opti_class = optimizers.Adagrad
        elif self.config.optimizer == "sgd":
            model_opti_class = optimizers.SGD
        else:
            logger.info("Optimizer not supporter: {}. Applying adam.".format(self.config.optimizer))
            model_opti_class = optimizers.Adam
        return model_opti_class

    def _get_model_checkpoint(self, model_weights_path):
        return ModelCheckpoint(
            filepath=model_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True
        )

    def _get_tensorboard(self, output_model_folder):
        log_path = utils.get_file_path(output_model_folder.get_path(), constants.TENSORBOARD_LOGS)
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)
        return TensorBoard(log_dir=log_path, write_graph=True)

    def _get_callbacks(self, output_model_folder, model_weights_path):
        callback_list = []
        callback_list.append(self._get_model_checkpoint(model_weights_path))
        if self.config.use_tensorboard:
            callback_list.append(self._get_tensorboard(output_model_folder))
        return callback_list

    def compile(self):
        model_opti_class = self._get_optimizer_class()

        # Cleaning custom parameters
        params_opti = utils.clean_custom_params(self.config.custom_params_opti)
        params_opti["learning_rate"] = self.config.learning_rate

        model_opti = model_opti_class(**params_opti)
        self.dku_model.compile(
            optimizer=model_opti,
            loss=constants.COMPILE_LOSS_FUNCTION,
            metrics=constants.COMPILE_METRICS
        )

    def _retrain(self, train_generator, test_generator, callback_list):
        self.dku_model.fit(
            x=train_generator,
            steps_per_epoch=self.config.nb_steps_per_epoch,
            epochs=self.config.nb_epochs,
            validation_data=test_generator,
            validation_steps=self.config.nb_validation_steps,
            callbacks=callback_list,
            shuffle=False,
            verbose=2
        )

    def _get_augmented_images(self, image, extra_images_gen):
        def _run_image_augm(im, augm_gen):
            image_augm = np.tile(im, (self.config.data_augmentation, 1, 1, 1))
            return next(augm_gen.flow(image_augm, batch_size=self.config.data_augmentation))

        return tf.numpy_function(
            func=lambda x: _run_image_augm(x, extra_images_gen),
            inp=[image],
            Tout=tf.float32)

    def _add_data_augmentation(self, X_tfds, y_values):
        extra_images_gen = self._get_tf_image_data_gen()
        X_tfds = X_tfds.map(map_func=lambda x: self._get_augmented_images(x, extra_images_gen),
                            num_parallel_calls=constants.AUTOTUNE)
        X_tfds = X_tfds.flat_map(map_func=lambda x: tf.data.Dataset.from_tensor_slices(x))
        y_values = np.repeat(y_values, self.config.n_augmentation, axis=0)
        return X_tfds, y_values

    def _build_tfds(self, pddf, images_folder, ignore_augm=False):
        use_augm = self.config.data_augmentation and not ignore_augm
        X_tfds = utils.read_images_to_tfds(
            images_folder=images_folder,
            np_images=pddf[constants.FILENAME].values)
        X_tfds = utils.apply_preprocess_image(
            tfds=X_tfds,
            input_shape=self.config.input_shape,
            preprocessing=self.dku_model.application.preprocessing)
        y_values = utils.convert_target_to_np_array(pddf[constants.LABEL].values)["remapped"]
        if use_augm:
            X_tfds, y_values = self._add_data_augmentation(X_tfds, y_values)

        y_tfds = tf.data.Dataset.from_tensor_slices(y_values)
        tfds = tf.data.Dataset.zip((X_tfds, y_tfds)).batch(self.config.batch_size, drop_remainder=True).repeat()
        optim_tfds = tfds.prefetch(constants.AUTOTUNE)
        return optim_tfds

    def compute(self, image_folder, model_folder, label_df, output_folder):
        self.load_dku_model(model_folder, label_df)
        self.compile()

        train_df, test_df = self._build_train_test_sets(label_df)

        train_tfds = self._build_tfds(train_df, image_folder)
        test_tfds = self._build_tfds(test_df, image_folder, ignore_augm=True)

        callbacks = self._get_callbacks(
            output_model_folder=output_folder,
            model_weights_path=self.dku_model.get_weights_path()
        )
        self._retrain(
            train_generator=train_tfds,
            test_generator=test_tfds,
            callback_list=callbacks
        )

        self.dku_model.retrained = True
        return self.dku_model
