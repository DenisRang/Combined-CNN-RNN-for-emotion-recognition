"""
Find frames of every video with edge values of valence and arousal
"""
import os
import pandas as pd

from src.config import RAW_ANNOTATIONS_DATA_DIR


def print_edge_frames(video_filename, folder):
    print(video_filename)

    a_path = os.path.join(RAW_ANNOTATIONS_DATA_DIR, folder, video_filename)
    a_df = pd.read_csv(a_path)
    a_df['min_v_min_a'] = - a_df['valence'] - a_df['arousal']
    a_df['min_v_max_a'] = - a_df['valence'] + a_df['arousal']
    a_df['max_v_min_a'] = a_df['valence'] - a_df['arousal']
    a_df['max_v_max_a'] = a_df['valence'] + a_df['arousal']

    min_v_min_a = a_df['min_v_min_a'].idxmax()
    min_v_max_a = a_df['min_v_max_a'].idxmax()
    max_v_min_a = a_df['max_v_min_a'].idxmax()
    max_v_max_a = a_df['max_v_max_a'].idxmax()

    print(
        f'min_v_min_a  -  ind: {min_v_min_a}, value: {a_df["min_v_min_a"][min_v_min_a]}, v: {a_df["valence"][min_v_min_a]}, a: {a_df["arousal"][min_v_min_a]}\n'
        f'min_v_max_a  -  ind: {min_v_max_a}, value: {a_df["min_v_max_a"][min_v_max_a]}, v: {a_df["valence"][min_v_max_a]}, a: {a_df["arousal"][min_v_max_a]}\n'
        f'max_v_min_a  -  ind: {max_v_min_a}, value: {a_df["max_v_min_a"][max_v_min_a]}, v: {a_df["valence"][max_v_min_a]}, a: {a_df["arousal"][max_v_min_a]}\n'
        f'max_v_max_a  -  ind: {max_v_max_a}, value: {a_df["max_v_max_a"][max_v_max_a]}, v: {a_df["valence"][max_v_max_a]}, a: {a_df["arousal"][max_v_max_a]}\n')


folder = 'train'
folder_path = os.path.join(RAW_ANNOTATIONS_DATA_DIR, folder)
video_filenames = os.listdir(folder_path)

for video_filename in video_filenames:
    print_edge_frames(video_filename, folder)
