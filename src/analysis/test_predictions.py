"""
Calculate CCC for some video of test folder
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src import metrics
from src.config import PREDICTIONS
from src.data import DataSet

video_filename = "79-30-960x720"

path = os.path.join(PREDICTIONS, video_filename + '.txt')  # numpy will auto-append .npy
pred_df = pd.read_csv(path, sep=",")
pred_df[pred_df['valence'] == -5] = np.nan
# pred_df = pred_df.interpolate(method='linear', axis=0).fillna(-5)
# pred_df = pred_df.interpolate(method='linear', axis=0).fillna(0)
pred_df = pred_df.interpolate(method='linear', axis=0).ffill().bfill()
pred_df = pred_df.ex
pred = pred_df[['valence', 'arousal']].values

true = DataSet.get_targets('test', video_filename)

r = len(true) if len(pred) > len(true) else len(pred)
pred = tf.convert_to_tensor(pred[:r], np.float32)
true = tf.convert_to_tensor(true[:r], np.float32)
ccc_v = metrics.ccc_v(true, pred)
ccc_a = metrics.ccc_a(true, pred)

print(f'ccc_v: {ccc_v}, ccc_a: {ccc_a}')
