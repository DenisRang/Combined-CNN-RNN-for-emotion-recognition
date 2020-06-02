"""
Generate the large dataset on multiple cores in real time and feed it right away to deep learning model.
We use the similar data generator as in https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
They use it in training of deep learning model on 3D images of protein structure in medical purpose to predict enzymatic
function via their amino acid composition.
"""
import os

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras_vggface import utils


from src.config import RANDOM_STATE, PROCESSED_SEQUENCES_DATA_DIR, RNN_WINDOW_SIZE
from src.processor import process_image


class DataGenerator(keras.utils.Sequence):
    """
    Generates frame batches for CNN
    """

    def __init__(self, list_IDs, targets, train_test, batch_size=128, dim=(96, 96), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.targets = targets
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.train_test = train_test
        self.on_epoch_end()
        if train_test == 'train':
            self.image_gen = ImageDataGenerator(rotation_range=15,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range=0.01,
                                                zoom_range=[0.9, 1.25],
                                                horizontal_flip=True,
                                                vertical_flip=False,
                                                fill_mode='reflect',
                                                data_format='channels_last',
                                                brightness_range=[0.5, 1.5])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            preprocessed_image = process_image(ID, self.dim)
            augmented_image = preprocessed_image
            if self.train_test == 'train':
                augmented_image = self.image_gen.random_transform(preprocessed_image, seed=RANDOM_STATE)
            augmented_image = (augmented_image / 255.).astype(np.float32)
            X[i,] = augmented_image

            # Store target
            y[i] = self.targets[ID]

        # X = utils.preprocess_input(X, version=1)

        return X, y


class SequenceDataGenerator(keras.utils.Sequence):
    """
    Generates batches of sequences of feature vectors from frames for RNN
    """

    def __init__(self, dataset, train_test, window_size=RNN_WINDOW_SIZE, batch_size=128, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.dataset = dataset
        self.list_IDs = dataset.get_sequence_partition(train_test)
        self.train_test = train_test
        self.window_size = window_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.sequences = self.load_sequences()
        print(f'Number of sequences for {train_test} generator: {len(self.list_IDs)}')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_sequences(self):
        sequences = {}
        sequence_filenames = os.listdir(PROCESSED_SEQUENCES_DATA_DIR)
        for sequence_filename in sequence_filenames:
            video_filename = sequence_filename[:-13]
            #             sequences[video_filename] = (np.load(os.path.join(PROCESSED_SEQUENCES_DATA_DIR, sequence_filename))*1000)
            sequences[video_filename] = np.load(os.path.join(PROCESSED_SEQUENCES_DATA_DIR, sequence_filename))

        return sequences

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.window_size, 300), dtype=np.float32)
        y = np.empty((self.batch_size, 2), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            video_filename, frame_idx, frame_position = ID
            X[i,] = self.sequences[video_filename][frame_position:frame_position + self.window_size]
            y[i,] = self.dataset.get_sequence_target(self.train_test, ID)

        return X, y
