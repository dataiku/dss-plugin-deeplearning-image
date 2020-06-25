import numpy as np
import dku_deeplearning_image.utils as utils
import dku_deeplearning_image.constants as constants
import math


class DkuImageGenerator:
    def __init__(self, images_folder, labels, input_shape, batch_size, preprocessing,
                 use_augmentation, extra_images_gen=None, n_augm=None):
        self.images_folder = images_folder
        self.labels = labels
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.use_augmentation = use_augmentation
        self.extra_images_gen = extra_images_gen
        self.n_augm = n_augm

    def _get_batch_size_adapted(self):
        return int(self.batch_size / self.n_augm) if self.use_augmentation else self.batch_size

    def _preprocess_img(self, images_folder, img_filename):
        image = utils.get_cached_file_from_folder(images_folder, img_filename)
        return utils.preprocess_img(image, self.input_shape, self.preprocessing)

    def _get_augmented_images(self, image):
        augm_image = np.tile(image, (self.n_augm, 1, 1, 1))
        return next(self.extra_images_gen.flow(augm_image, batch_size=self.n_augm))

    def _process_one_image(self, row):
        img_filename = row[constants.FILENAME]
        label = row[constants.LABEL]
        label_index = self.labels.index(label)
        try:
            image = self._preprocess_img(self.images_folder, img_filename)
            if self.use_augmentation:
                X_batch = self._get_augmented_images(image)
                y_batch = [label_index] * self.n_augm
            else:
                X_batch = [image]
                y_batch = [label_index]
        except IOError as e:
            print("Cannot read the image '{}', skipping it. Error: {}".format(img_filename, e))
            X_batch, y_batch = [], []
        return X_batch, y_batch

    def _get_batch_features(self, batch_df):
        X_batch_list = []
        y_batch_list = []

        for index, row in batch_df.iterrows():
            X_batch, y_batch = self._process_one_image(self.images_folder)
            X_batch_list.extend(X_batch)
            y_batch_list.extend(y_batch)

        X_batch = np.array(X_batch_list)

        actual_batch_size = X_batch.shape[0]
        y_batch = np.zeros((actual_batch_size, len(self.labels)))
        y_batch[range(actual_batch_size), y_batch_list] = 1

        return X_batch, y_batch

    @utils.threadsafe_generator
    def load(self, image_df):
        n_images = image_df.shape[0]
        batch_size_adapted = self._get_batch_size_adapted()
        n_batch = int(math.ceil(n_images * 1.0 / batch_size_adapted))
        while True:
            for num_batch in range(n_batch):
                batch_df = image_df.iloc[num_batch * batch_size_adapted: (num_batch + 1) * batch_size_adapted, :]
                yield self._get_batch_features(batch_df)