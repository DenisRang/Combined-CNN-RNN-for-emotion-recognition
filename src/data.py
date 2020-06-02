import os.path

import pandas as pd
from matplotlib.colors import LogNorm
from pylab import *

from src.config import RAW_ANNOTATIONS_DATA_DIR, \
    PROCESSED_FRAMES_DATA_DIR, RNN_WINDOW_SIZE


class DataSet():
    """
    Class for managing our data (meta information from annotation and frames).
    """

    def __init__(self):
        self.data = {}
        self.extract_metadata()

    def extract_metadata(self):
        folders = ['train', 'test']
        for folder in folders:
            video_filenames = self.get_video_filenames(folder)
            self.extract_video_file_metadata(folder, video_filenames)

    def get_video_filenames(self, folder):
        folder_path = os.path.join(RAW_ANNOTATIONS_DATA_DIR, folder)
        video_filenames = os.listdir(folder_path)
        video_filenames = map(lambda filename: filename[:-4], video_filenames)  # Remove extension

        return video_filenames

    def extract_video_file_metadata(self, folder, video_filenames):
        videos_folder_dict = {}
        for video_filename in video_filenames:
            annotations = self.get_targets(folder, video_filename)
            frames = self.get_frames_filenames(video_filename)
            frames = self.clean_frames(annotations, frames)
            videos_folder_dict[video_filename] = (annotations, frames)
        self.data[folder] = videos_folder_dict

    @staticmethod
    def get_targets(folder, video_filename):
        annotation_path = os.path.join(RAW_ANNOTATIONS_DATA_DIR, folder, video_filename + '.txt')
        annotation_df = pd.read_csv(annotation_path, sep=",")
        return annotation_df[['valence', 'arousal']].values

    def get_frames_filenames(self, video_filename):
        frames_path = os.path.join(PROCESSED_FRAMES_DATA_DIR, video_filename)
        return sorted(os.listdir(frames_path))

    def get_frames_paths(self, train_test, video_filename):
        return map(lambda frame_idx: self.get_frame_path(video_filename, frame_idx),
                   self.data[train_test][video_filename][1])

    def clean_frames(self, annotations, frames):
        clean_frames = []
        num_calculated_targets = len(annotations)
        for frame in frames:
            if frame == '.DS_Store':
                continue
            frame_idx = self.get_frame_idx(frame)
            if frame_idx < num_calculated_targets and (
                    annotations[frame_idx][0] != -5 and annotations[frame_idx][1] != -5):
                clean_frames.append(frame_idx)
        return np.asarray(clean_frames)

    @staticmethod
    def get_frame_idx(frame_filename):
        return int(frame_filename[:-4]) - 1

    @staticmethod
    def get_frame_path(video_filename, frame_idx):
        frame_filename = f'{frame_idx + 1:05d}.jpg'
        frame_path = os.path.join(PROCESSED_FRAMES_DATA_DIR, video_filename, frame_filename)
        return frame_path

    def get_partition(self, train_test, balanced=False):
        x = []
        y = {}
        data = self.data[train_test]
        video_filenames = list(data.keys())
        density, bins = self.get_density()
        max_class = (np.mean(density) + np.max(density)) / 2
        #         max_class = np.mean(density)
        bins = list(bins)
        for video_filename in video_filenames:
            targets = data[video_filename][0]
            frame_idxs = data[video_filename][1]
            for frame_idx in frame_idxs:
                target = targets[frame_idx]

                if train_test == 'train' and balanced:
                    v = target[0]
                    a = target[1]
                    if (v == 0 and a == 0):
                        continue

                    v_bin_idx = np.digitize(v, bins, right=True) - 1
                    a_bin_idx = np.digitize(a, bins, right=True) - 1

                    u = np.random.random()
                    k = density[v_bin_idx, a_bin_idx]
                    p = 1 - (k / max_class)
                    if u < p:
                        frame_filename = f'{frame_idx + 1:05d}.jpg'
                        frame_path = os.path.join(PROCESSED_FRAMES_DATA_DIR, video_filename, frame_filename)
                        x.append(frame_path)
                        y[frame_path] = target
                else:
                    frame_filename = f'{frame_idx + 1:05d}.jpg'
                    frame_path = os.path.join(PROCESSED_FRAMES_DATA_DIR, video_filename, frame_filename)
                    x.append(frame_path)
                    y[frame_path] = target
        return x, y

    def get_sequence_partition(self, train_test, window_size=RNN_WINDOW_SIZE):
        complex_idxs = []
        data = self.data[train_test]
        video_filenames = list(data.keys())
        for video_filename in video_filenames:
            frame_idxs = data[video_filename][1]

            # Check if video is shorter than size of window
            if (len(frame_idxs) < window_size):
                continue

            for i, frame_idx in enumerate(frame_idxs[:-window_size]):
                # if i % 100 != 0:
                #     continue
                # Check if there is gap between frames
                if frame_idxs[i + window_size] != frame_idx + window_size:
                    continue
                complex_idx = (video_filename, frame_idx, i)
                complex_idxs.append(complex_idx)
        return complex_idxs

    def get_sequence_targets(self, train_test, complex_idx, window_size=RNN_WINDOW_SIZE):
        video_filename, frame_idx, frame_position = complex_idx
        video_metadata = self.data[train_test][video_filename]
        video_targets = video_metadata[0]
        sequence_targets = video_targets[frame_idx:frame_idx + window_size]
        return sequence_targets

    def get_sequence_target(self, train_test, complex_idx, window_size=RNN_WINDOW_SIZE):
        video_filename, frame_idx, frame_position = complex_idx
        video_metadata = self.data[train_test][video_filename]
        video_targets = video_metadata[0]
        sequence_targets = video_targets[frame_idx + window_size]
        return sequence_targets

    def get_val_ar(self, balancing_mode='all', max_mode='max_mean', train_test='train'):
        valences = []
        arousals = []
        data = self.data[train_test]
        video_filenames = list(data.keys())
        if balancing_mode == 'all':
            for video_filename in video_filenames:
                targets = data[video_filename][0]
                current_valences = targets[:, 0]
                current_arousals = targets[:, 1]
                for v, a in zip(current_valences, current_arousals):
                    if (v == -5 or a == -5):
                        continue
                    valences.append(v)
                    arousals.append(a)
        elif balancing_mode == 'balanced':
            density, bins = self.get_density()
            if max_mode == 'max':
                max_class = np.max(density)
            elif max_mode == 'mean':
                max_class = np.mean(density)
            elif max_mode == 'max_mean':
                max_class = (np.mean(density) + np.max(density)) / 2
            bins = list(bins)
            for video_filename in video_filenames:
                targets = data[video_filename][0]
                current_valences = targets[:, 0]
                current_arousals = targets[:, 1]
                for v, a in zip(current_valences, current_arousals):
                    if (v == -5 or a == -5) or (v == 0 and a == 0):
                        continue

                    v_bin_idx = np.digitize(v, bins, right=True) - 1
                    a_bin_idx = np.digitize(a, bins, right=True) - 1

                    u = np.random.random()
                    k = density[v_bin_idx, a_bin_idx]
                    p = 1 - (k / max_class)
                    if u < p:
                        valences.append(v)
                        arousals.append(a)

        return valences, arousals

    def get_density(self):
        valences, arousals = self.get_val_ar(balancing_mode='all')
        res_hist = hist2d(valences, arousals, bins=40, cmap=cm.jet, norm=LogNorm())
        density = res_hist[0]
        bins = res_hist[1]
        return density, bins
