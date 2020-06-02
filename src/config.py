"""
Constant values of different configuration setting variables for different environments
"""
env = 'local'
if env == 'colab':
    RAW_ANNOTATIONS_DATA_DIR = "/content/annotations"
    PROCESSED_FRAMES_DATA_DIR = "/content/cropped_aligned"
    PROCESSED_SEQUENCES_DATA_DIR = "/content/processed_dataset/sequences"
    MODEL_CHECKPOINTS_DIR = "data/checkpoints"
elif env == 'local':
    RAW_ANNOTATIONS_DATA_DIR = "/Users/denisrangulov/Fake Colab/aff_wild2/annotations"
    RAW_VIDEO_DATA_DIR = "/Users/denisrangulov/Fake Colab/aff_wild2_videos"
    PROCESSED_FRAMES_DATA_DIR = "/Users/denisrangulov/Fake Colab/cropped_aligned"
    PROCESSED_SEQUENCES_DATA_DIR = "/Users/denisrangulov/Fake Colab/sequences"
    PREDICTIONS = '/Users/denisrangulov/predictions/submission-1'
    INTERPOLATION_PREDICTIONS = '/Users/denisrangulov/predictions/interpolated/submission-1'
    MODEL_CHECKPOINTS_DIR = "/Users/denisrangulov/Fake Colab/model_checkpoints"
    SAVED_CNN_EXTRACTOR_MODEL = '/Users/denisrangulov/Google Drive/EmotionRecognition/data/checkpoints/' \
                                'simple_cnn_ccc_loss.0001-0.7792.hdf5'
    SAVED_RNN_MODEL = '/Users/denisrangulov/Google Drive/EmotionRecognition/data/checkpoints/' \
                      'gru_ccc_loss-epoch:17-loss:0.6913-ccc_v:0.2252-ccc_a:0.3922.hdf5'

elif env == 'vm':
    RAW_ANNOTATIONS_DATA_DIR = "/home/rangulov/annotations"
    RAW_VIDEO_DATA_DIR = ""  # dont store videos on server
    PROCESSED_FRAMES_DATA_DIR = "/home/rangulov/cropped_aligned"
    PROCESSED_SEQUENCES_DATA_DIR = "/home/rangulov/sequences_simple_cnn_ccc_loss.0001-0.7792"
    PREDICTIONS = '/home/rangulov/predictions/gru_ccc_loss-epoch:17-loss:0.6913-ccc_v:0.2252-ccc_a:0.3922.hdf5'
    MODEL_CHECKPOINTS_DIR = "/home/rangulov/EmotionRecognition/data/checkpoints"
    SAVED_CNN_EXTRACTOR_MODEL = '/home/rangulov/EmotionRecognition/data/checkpoints/' \
                                'simple_cnn_ccc_loss.0001-0.7792.hdf5'
    SAVED_RNN_MODEL = '/home/rangulov/EmotionRecognition/data/checkpoints/' \
                      'gru_ccc_loss-epoch:17-loss:0.6913-ccc_v:0.2252-ccc_a:0.3922.hdf5'

RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.2
RNN_WINDOW_SIZE = 100
