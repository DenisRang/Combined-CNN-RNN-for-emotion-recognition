"""
Train different CNN models on frames
"""
import datetime
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from src.config import MODEL_CHECKPOINTS_DIR
from src.data import DataSet
from src.data_generator import DataGenerator
from src.models import ResearchModels


def get_generators(model_name):
    def get_main_params(train_test):
        dataset = DataSet()
        list_IDs, targets = dataset.get_partition(train_test, balanced=True)

        return list_IDs, targets, train_test

    params = {'dim': (96, 96),
              'batch_size': 128,
              'n_channels': 1,
              'shuffle': True}

    # Generators
    train_generator = DataGenerator(*get_main_params("train"), **params)
    valid_generator = DataGenerator(*get_main_params("test"), **params)

    return train_generator, valid_generator


def train_model(research_model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    research_model.model.fit_generator(
        train_generator,
        #         workers=4,
        steps_per_epoch=1000,
        #         use_multiprocessing=True,
        validation_data=validation_generator,
        epochs=nb_epoch,
        callbacks=callbacks)


def main():
    model_name = 'simple_cnn_ccc_loss'

    if not os.path.exists(MODEL_CHECKPOINTS_DIR):
        os.makedirs(MODEL_CHECKPOINTS_DIR)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINTS_DIR, model_name + '.{epoch:04d}-{val_loss:.4f}.hdf5'),
        verbose=1,
        save_best_only=True,
        period=1)
    logdir = os.path.join("data/logs", model_name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch=0)

    # saved_model_path='/home/rangulov/EmotionRecognition/data/checkpoints/vggface_ccc_loss.0002-0.9685.hdf5'
    saved_model_path = None
    research_model = ResearchModels(model_name, saved_model=saved_model_path)

    generators = get_generators('simple-cnn')
    train_model(research_model, 100, generators, [checkpointer, tensorboard_callback])


if __name__ == '__main__':
    main()
