# yh computer
import os

TRAIN_WAVROOT_DIR = '/home/dudans/Desktop/DB/robot_VAD_DB/'
TRAIN_DATAROOT_DIR = '/home/dudans/Desktop/DB/robot_feat/dist_train_MRCG_nfilt96'
#TRAIN_DATAROOT_DIR = '/home/dudans/Desktop/DB/robot_feat/dist_train_logfbank_nfilt40'

#TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/3M0D/kaist_10h2w'
#TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/1M0D/kaist_10h3c'
#TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/SRC/kaist_10h3c'
#TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/1M0D/kaist_10h_refmotor_snr103c'
TEST_WAVROOT_DIR = '/home/dudans/Desktop/DB/robot_VAD_DB/kaist_10h_DB/1M0D/kaist_10h_ssuljeon1_snr103c'
TEST_DATAROOT_DIR = '/home/dudans/Desktop/DB/robot_feat/test_1M0D_ssuljeon1_snr103c_MRCG_nfilt96'
#TEST_DATAROOT_DIR = '/data/DB/Speaker_robot_test_DB/3M0D/kaist_10h_ssuljeon1_snr103c'

FEAT_DIR = '/home/dudans/Desktop/DB'
#FEAT_DIR = '/home/administrator/Desktop/Speaker_recognition_robot'

LABEL_DIR = '/home/dudans/Desktop/DB/sohn_result'

NUM_PREVIOUS_FRAME = 20 #30
#NUM_PREVIOUS_FRAME = 13
NUM_NEXT_FRAME = 20 #10

NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 96 # 96 for MRCG, 40 for FBANK

USE_GLOBAL_NORM = True
MEAN_STD_PATH = os.path.join(os.getcwd(), "train_mean_and_var")
MEAN_PATH = os.path.join(MEAN_STD_PATH, "train_mean.txt")
STD_PATH = os.path.join(MEAN_STD_PATH, "train_std.txt")