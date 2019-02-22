import os

# Wave path
TRAIN_WAV_DIR = '/home/admin/Desktop/read_25h_2/train'
TEST_WAV_DIR = 'test_wavs'

# Feature path
TRAIN_FEAT_DIR = 'feat_MRCG_nfilt96/train'
TEST_FEAT_DIR = 'feat_MRCG_nfilt96/test'

LABEL_DIR = 'sohn_result'

# Context window size
NUM_PREVIOUS_FRAME = 20 #30
NUM_NEXT_FRAME = 20 #10
NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME

# Settings for feature extraction
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 96 # 96 for MRCG, 40 for FBANK

# Settings for feature normalization
USE_GLOBAL_NORM = True
MEAN_STD_PATH = os.path.join(os.getcwd(), "train_mean_and_var")
MEAN_PATH = os.path.join(MEAN_STD_PATH, "train_mean.txt")
STD_PATH = os.path.join(MEAN_STD_PATH, "train_std.txt")