import os

import cv2
import pandas as pd

from src.config import PROCESSED_FRAMES_DATA_DIR, RAW_VIDEO_DATA_DIR
from src.data import DataSet


class VideoHelper():
    """
    Get meta-information from videos. Private methods are used only locally because we don't save videos on the server
    """
    def __init__(self):
        video_lengths_df = pd.read_csv('data/video_lengths.csv', sep=',')
        video_lengths = {}
        for filename, length in video_lengths_df.values:
            video_lengths[filename] = length
        self.video_lengths = video_lengths

    def get_num_frames(self, video_filename):
        """
        Get number of frames in particular video

        :param video_filename: video filename without extension
        :return: number of frames
        """
        return self.video_lengths[video_filename]

    def get_private_test_video_filenames(self):
        """
        Get test video filenames that have not annotations

        :return: list of video filenames without extension
        """
        return list(self.video_lengths.keys())

    @staticmethod
    def extract_video_lengths():
        """
        Extract number of frames from videos and write to 'video_lengths.csv'
        """
        private_test_video_filenames = VideoHelper._extract_private_test_video_filenames()

        video_lengths = {}
        for video_filename in private_test_video_filenames:
            video_lengths[video_filename] = VideoHelper._extract_num_frames(video_filename)

        video_lengths_df = pd.DataFrame(video_lengths.items())
        video_lengths_df.to_csv('data/video_lengths.csv', sep=',', index=False)

    @staticmethod
    def _clear_video_filename(video_filename):
        video_filename = video_filename.replace('_right', '')
        video_filename = video_filename.replace('_left', '')
        return video_filename

    @staticmethod
    def _extract_num_frames(video_filename):
        video_filename = VideoHelper._clear_video_filename(video_filename)
        video_path = os.path.join(RAW_VIDEO_DATA_DIR, video_filename + '.mp4')
        if not os.path.isfile(video_path):
            video_path = os.path.join(RAW_VIDEO_DATA_DIR, video_filename + '.avi')
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return length

    @staticmethod
    def _extract_private_test_video_filenames():
        all_video_filenames = set(os.listdir(PROCESSED_FRAMES_DATA_DIR))

        dataset = DataSet()
        train_test_video_filenames = set(dataset.data['train'].keys()) | set(dataset.data['test'].keys())

        private_test_video_filenames = all_video_filenames - train_test_video_filenames

        private_test_video_filenames = [video_filename for video_filename in private_test_video_filenames if
                                        not video_filename[0] == '.']
        return private_test_video_filenames
