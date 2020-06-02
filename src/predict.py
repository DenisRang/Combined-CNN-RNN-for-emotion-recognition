"""
Make predictions using saved CNN and RNN models
"""
import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm

from src import metrics
from src.config import PREDICTIONS, RNN_WINDOW_SIZE, \
    SAVED_CNN_EXTRACTOR_MODEL, SAVED_RNN_MODEL
from src.data import DataSet
from src.extractor import Extractor
from src.utils import VideoHelper


def prepare_sequence_for_rnn(sequence):
    rnn_input = []
    for r in range(RNN_WINDOW_SIZE, len(sequence)):
        l = r - RNN_WINDOW_SIZE
        window = np.asarray(sequence[l:r])
        rnn_input.append(window)
    return np.asarray(rnn_input)


cnn_extractor_model = Extractor(SAVED_CNN_EXTRACTOR_MODEL)
rnn_model = load_model(SAVED_RNN_MODEL, custom_objects={'ccc_loss': metrics.ccc_loss,
                                                        'rmse': metrics.rmse,
                                                        'rmse_v': metrics.rmse_v,
                                                        'rmse_a': metrics.rmse_a,
                                                        'cc_v': metrics.cc_v,
                                                        'cc_a': metrics.cc_a,
                                                        'ccc_v': metrics.ccc_v,
                                                        'ccc_a': metrics.ccc_a})

video_helper = VideoHelper()
private_test_video_filenames = video_helper.get_private_test_video_filenames()

if not os.path.exists(PREDICTIONS):
    os.makedirs(PREDICTIONS)

vbar = tqdm(total=len(private_test_video_filenames))
for video_filename in private_test_video_filenames:
    prediction_path = os.path.join(PREDICTIONS, video_filename + '.txt')

    # Check if we already have it.
    if os.path.isfile(prediction_path + '.txt'):
        vbar.update(1)
        continue

    num_frames = video_helper.get_num_frames(video_filename)
    predictions = np.full((num_frames, 2), -5, dtype=np.float32)
    sequence = []
    fbar = tqdm(total=num_frames)
    for frame_idx in range(num_frames):
        frame_path = DataSet.get_frame_path(video_filename, frame_idx)
        if os.path.isfile(frame_path):
            feature_vector = cnn_extractor_model.extract(frame_path)
            sequence.append(feature_vector)
        elif len(sequence) > 0:
            # Uncomment to predict first less than 'RNN_WINDOW_SIZE' frames using CNN
            # num_cnn_predictions = min(len(sequence), RNN_WINDOW_SIZE)
            # x = np.asarray(sequence[:num_cnn_predictions])
            # prediction = cnn_extractor_model.predict(x)
            # predictions[frame_idx - len(sequence):frame_idx - len(sequence) + len(prediction)] = prediction
            if len(sequence) > RNN_WINDOW_SIZE:
                x = prepare_sequence_for_rnn(sequence)
                prediction = rnn_model.predict(x)
                predictions[frame_idx - len(prediction):frame_idx] = prediction
            sequence = []
        if frame_idx == num_frames - 1 and len(sequence) > 0:
            # Uncomment to predict first less than 'RNN_WINDOW_SIZE' frames using CNN
            # num_cnn_predictions = min(len(sequence), RNN_WINDOW_SIZE)
            # x = np.asarray(sequence[:num_cnn_predictions])
            # prediction = cnn_extractor_model.predict(x)
            # predictions[frame_idx - len(sequence) + 1:frame_idx - len(sequence) + 1 + len(prediction)] = prediction
            if len(sequence) > RNN_WINDOW_SIZE:
                x = prepare_sequence_for_rnn(sequence)
                prediction = rnn_model.predict(x)
                predictions[frame_idx - len(prediction) + 1:frame_idx + 1] = prediction
        fbar.update(1)
    fbar.close()

    predictions = np.asarray(predictions)
    predictions = pd.DataFrame({'valence': predictions[:, 0], 'arousal': predictions[:, 1]})
    predictions.to_csv(prediction_path, sep=',', index=False)
    vbar.update(1)
vbar.close()
