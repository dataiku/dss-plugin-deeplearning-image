from .dku_recipe import DkuRecipe
from utils_objects import DkuModel
import dku_deeplearning_image.utils as utils
from utils_objects import DkuImageGenerator
import dku_deeplearning_image.constants as constants
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import shutil


class RetrainRecipe(DkuRecipe):
    def __init__(self, config):
        super(RetrainRecipe, self).__init__(config)

    def load_dku_model(self, model_folder, label_df):
        self.dku_model = DkuModel(model_folder)
        self.dku_model.label_df = label_df
        self.dku_model.load_model(self.config, constants.RETRAIN)
        self._set_trainable_layers()
        self.dku_model.print_summary()

    def _set_trainable_layers(self):
        utils.log_info("Will Retrain layer(s) with mode: {}".format(self.config.layer_to_retrain))
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
        utils.log_info("Using data augmentation with {} images generated per training image\n".format(
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
            utils.log_info("Optimizer not supporter: {}. Applying adam.".format(self.config.optimizer))
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
        self.dku_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=self.config.nb_steps_per_epoch,
            epochs=self.config.nb_epochs,
            validation_data=test_generator,
            validation_steps=self.config.nb_validation_steps,
            callbacks=callback_list,
            shuffle=False,
            verbose=2
        )

    def compute(self, image_folder, model_folder, label_df, output_folder):
        self.load_dku_model(model_folder, label_df)
        self.compile()

        train_df, test_df = self._build_train_test_sets(label_df)
        extra_images_gen = self._get_tf_image_data_gen() if self.config.data_augmentation else None

        dku_generator = DkuImageGenerator(
            images_folder=image_folder,
            labels=self.dku_model.get_distinct_labels(),
            input_shape=self.config.input_shape,
            batch_size=self.config.batch_size,
            preprocessing=self.dku_model.application.preprocessing,
            use_augmentation=self.config.data_augmentation,
            extra_images_gen=extra_images_gen,
            n_augm=self.config.n_augmentation
        )
        train_gen, test_gen = dku_generator.load(train_df), dku_generator.load(test_df)

        model_weights_path = self.dku_model.get_weights_path()

        callbacks = self._get_callbacks(
            output_model_folder=output_folder,
            model_weights_path=model_weights_path
        )
        self._retrain(
            train_generator=train_gen,
            test_generator=test_gen,
            callback_list=callbacks
        )

        self.dku_model.retrained = True
        return self.dku_model