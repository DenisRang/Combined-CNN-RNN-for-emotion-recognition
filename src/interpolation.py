"""
Interpolate predictions to fill -5,-5 values of valence and arousal with approximation of values of nearest neighborhoods
"""
import os
import pandas as pd
import numpy as np
from src.config import PREDICTIONS, INTERPOLATION_PREDICTIONS

if not os.path.exists(INTERPOLATION_PREDICTIONS):
    os.makedirs(INTERPOLATION_PREDICTIONS)

for video_filename in os.listdir(PREDICTIONS):
    path = os.path.join(PREDICTIONS, video_filename)
    pred_df = pd.read_csv(path, sep=",")
    pred_df[pred_df['valence'] == -5] = np.nan
    pred_df = pred_df.interpolate(method='linear', axis=0).ffill().bfill()

    interpolation_path = os.path.join(INTERPOLATION_PREDICTIONS, video_filename)
    pred_df.to_csv(interpolation_path, sep=',', index=False)
