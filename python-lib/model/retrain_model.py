from .dku_model import DkuModel
import dl_image_toolbox_utils as utils
from .dku_image_generator import DkuImageGenerator
import constants
from sklearn.model_selection import train_test_split
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import shutil
import numpy as np


class RetrainModel(DkuModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None

    def _load_model_and_pp(self):
        super().load(
            mf_path=self.config.model_folder,
            goal=constants.RETRAINING,
            input_shape=self.config.input_shape,
            pooling=self.config.model_pooling,
            reg=self.config.model_reg,
            dropout=self.config.model_dropout,
            n_classes=self.config.n_classes)

    def _set_trainable_layers(self):
        print("Will Retrain layer(s) with mode: {}".format(self.config.layer_to_retrain))
        layers = self.model.layers
        if self.config.layer_to_retrain == "all":
            n_last = len(layers)
        elif self.config.layer_to_retrain == "last":
            n_last = 1
        elif self.config.layer_to_retrain == "n_last":
            n_last = self.config.layer_to_retrain_n
        else:
            n_last = 0

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
        print("Using data augmentation with {} images generated per training image\n".format(
            self.config.n_augmentation))
        params_data_augment = utils.clean_custom_params(
            custom_params=self.config.custom_params_data_augment,
            params_type="Data Augmentation"
        )
        return ImageDataGenerator(**params_data_augment)

    def _get_optimizer_class(self, optimizer):
        if optimizer == "adam":
            model_opti_class = optimizers.Adam
        elif optimizer == "adagrad":
            model_opti_class = optimizers.Adagrad
        elif optimizer == "sgd":
            model_opti_class = optimizers.SGD
        else:
            print("Optimizer not supporter: {}. Applying adam.".format(optimizer))
            model_opti_class = optimizers.Adam
        return model_opti_class

    def _get_model_checkpoint(self, model_config, model_weights_path):
        should_save_weights_only = utils.should_save_weights_only(model_config)

        if self.config.use_gpu:
            mcheck = utils.MultiGPUModelCheckpoint(
                filepath=model_weights_path,
                base_model=self.base_model,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=should_save_weights_only
            )
        else:
            mcheck = ModelCheckpoint(
                filepath=model_weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=should_save_weights_only
            )
        return mcheck

    def _get_tensorboard(self, output_model_folder):
        log_path = utils.get_file_path(output_model_folder.get_path(), constants.TENSORBOARD_LOGS)

        if os.path.isdir(log_path):
            shutil.rmtree(log_path)

        return TensorBoard(log_dir=log_path, write_graph=True)

    def _get_callbacks(self, output_model_folder, model_config, model_weights_path):
        callback_list = []
        callback_list.append(self._get_model_checkpoint(model_config, model_weights_path))
        if self.config.use_tensorboard:
            callback_list.append(self._get_tensorboard(output_model_folder))
        return callback_list

    def load(self):
        self._load_model_and_pp()
        self.model = multi_gpu_model(self.base_model, self.config.n_gpu) if self.config.use_gpu else self.base_model
        self._set_trainable_layers()
        self.model.summary()

    def compile(self):
        model_opti_class = self._get_optimizer_class(self.config.optimizer)

        # Cleaning custom parameters
        params_opti = utils.clean_custom_params(self.config.custom_params_opti)
        params_opti["lr"] = self.config.learning_rate

        model_opti = model_opti_class(**params_opti)
        self.model.compile(optimizer=model_opti, loss='categorical_crossentropy', metrics=['accuracy'])

    def _retrain(self, train_generator, test_generator, callback_list):
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.config.nb_steps_per_epoch,
            epochs=self.config.nb_epochs,
            validation_data=test_generator,
            validation_steps=self.config.nb_validation_steps,
            callbacks=callback_list,
            shuffle=False,
            verbose=2
        )

    def retrain(self, images_folder, label_df, model_folder, output_model_folder):
        self.compile()
        train_df, test_df = self._build_train_test_sets(label_df)
        extra_images_gen = self._get_tf_image_data_gen() if self.config.data_augmentation else None

        dku_generator = DkuImageGenerator(
            images_folder=images_folder,
            labels=list(np.unique(label_df[constants.LABEL])),
            input_shape=self.config.input_shape,
            batch_size=self.config.batch_size,
            preprocessing=self.preprocessing,
            use_augmentation=self.config.data_augmentation,
            extra_images_gen=extra_images_gen,
            n_augm=self.config.n_augmentation
        )
        train_gen, test_gen = dku_generator.load(train_df), dku_generator.load(test_df)

        model_config = utils.get_model_config_from_file(model_folder)
        model_weights_path = utils.get_weights_path(
            output_model_folder,
            model_config,
            suffix=constants.RETRAINED_SUFFIX,
            should_exist=False
        )
        callbacks = self._get_callbacks(
            output_model_folder=output_model_folder,
            model_config=model_config,
            model_weights_path=model_weights_path
        )
        self._retrain(
            train_generator=train_gen,
            test_generator=test_gen,
            callback_list=callbacks
        )

