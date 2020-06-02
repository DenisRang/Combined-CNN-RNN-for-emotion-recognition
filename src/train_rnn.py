"""
Train different RNN models on saved feature vectors
"""
import datetime
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from src.config import MODEL_CHECKPOINTS_DIR
from src.data import DataSet
from src.data_generator import SequenceDataGenerator
from src.models import ResearchModels


def get_generators():
    dataset = DataSet()

    params = {'batch_size': 256,
              'shuffle': True}

    # Generators
    train_generator = SequenceDataGenerator(dataset, "train", **params)
    valid_generator = SequenceDataGenerator(dataset, "test", **params)

    return train_generator, valid_generator


def train_model(research_model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    research_model.model.fit_generator(
        train_generator,
        #         steps_per_epoch=1000,
        #         workers=4,
        #         use_multiprocessing=True,
        validation_data=validation_generator,
        epochs=nb_epoch,
        callbacks=callbacks)


def main():
    #     model_name = 'simple_rnn_tanh_3_layers_ccc_loss'
    #     model_name = 'gru_ccc_loss'
    #     model_name = 'gru'
    #     model_name = 'gru_1_layer_ccc_loss'
    #     model_name = 'simple_rnn'
    model_name = 'simple_rnn_ccc_loss'

    if not os.path.exists(MODEL_CHECKPOINTS_DIR):
        os.makedirs(MODEL_CHECKPOINTS_DIR)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINTS_DIR,
                              model_name + '-epoch:{epoch:02d}-loss:{val_loss:.4f}-ccc_v:{val_ccc_v:.4f}-ccc_a:{val_ccc_a:.4f}.hdf5'),
        verbose=1,
        save_best_only=True,
        period=1)
    logdir = os.path.join("data/logs", model_name + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch=0)

    # saved_model_path='/home/rangulov/EmotionRecognition/data/checkpoints/simple-cnn.0001-0.1362.hdf5'
    research_model = ResearchModels(model_name, saved_model=None)

    generators = get_generators()
    train_model(research_model, 100, generators, [checkpointer, tensorboard_callback])


if __name__ == '__main__':
    main()
