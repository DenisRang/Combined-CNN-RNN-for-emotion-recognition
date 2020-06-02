"""
Generates extracted features for each video, which other
models make use of.
"""
import os.path

import numpy as np
from tqdm.notebook import tqdm

from src.config import PROCESSED_SEQUENCES_DATA_DIR, SAVED_CNN_EXTRACTOR_MODEL
from src.data import DataSet
from src.extractor import Extractor


def extract_features():
    # Get the dataset.
    data = DataSet()

    # get the model.
    model = Extractor(SAVED_CNN_EXTRACTOR_MODEL)

    if not os.path.exists(PROCESSED_SEQUENCES_DATA_DIR):
        os.makedirs(PROCESSED_SEQUENCES_DATA_DIR)

    # Loop through data.
    folders = ['train', 'test']
    #     folders = ['train']
    for folder in folders:
        print(f'Extracting features from {folder} videos...')
        video_filenames = list(data.data[folder].keys())
        #         video_filenames=['171']
        pbar = tqdm(total=len(video_filenames))
        for video_filename in video_filenames:

            # Get the path to the sequence for this video.
            path = os.path.join(PROCESSED_SEQUENCES_DATA_DIR,
                                video_filename + '-features')  # numpy will auto-append .npy

            # Check if we already have it.
            if os.path.isfile(path + '.npy'):
                pbar.update(1)
                continue

            # Get the frames for this video.
            frames = data.get_frames_paths(folder, video_filename)

            # Now loop through and extract features to build the sequence.
            sequence = []
            for image in frames:
                features = model.extract(image)
                sequence.append(features)

            # Save the sequence.
            np.save(path, sequence)

            pbar.update(1)

        pbar.close()


def main():
    extract_features()


if __name__ == '__main__':
    main()
