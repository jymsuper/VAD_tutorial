# yh computer (VAD), python2
"""
Modification of the function 'DBspeech_wav_reader.py' of the deep-speaker created by philipperemy 
Working on python 2
Input : DB path
Output : 1) Make DB structure using pd.DataFrame which has 3 columns (file id, file path, speaker id, DB id)
            => 'read_DB_structure' function
         2) Read a wav file from DB structure
            => 'read_audio' function

<corpus root> Speaker_robot_DB
    |
    .- S1_light_700_mod/
    |        |
    |        .- C001F2/
    |        |     |
    |        |     .- C001F2INDE001.txt
    |        |     |
    |        |     .- C001F2INDE001.wav
    |        |     |
    |        |     .- C001F2INDE002.txt
    |        |     |
    |        |     ...
    |        |
    |        .- C001M3/
    |              |
    |              ...
    |
    .- Etri_readsent/
    |        | ...

"""
import logging
import os
#from glob import glob
import glob2 # for python2

import librosa
import numpy as np
import pandas as pd
import configure as c
from configure import SAMPLE_RATE

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_files(directory, pattern='**/*.p'):
    """Recursively finds all feature files matching the pattern."""
    return glob2.glob(os.path.join(directory, pattern)) # for python2

def find_waves(directory, pattern='**/*.wav'):
    """Recursively finds all feature files matching the pattern."""
    return glob2.glob(os.path.join(directory, pattern)) # for python2

def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

def convert_filename_to_labelpath(filename):
    if 'read_25h' in filename:
        label_folder = os.path.join(c.LABEL_DIR, 'read_25h') # ex) /home/dudans/Desktop/DB/sohn_result/read_25h
        spk_and_feat_name = ('/').join(filename.split('/')[-2:]).replace('.p','.mat') # ex) 322F3065/SNR322F3MIC065188.mat
        label_path = os.path.join(label_folder, spk_and_feat_name)
    if 'spon_22h' in filename:
        label_folder = os.path.join(c.LABEL_DIR, 'spon_22h') # ex) /home/dudans/Desktop/DB/sohn_result/spon_22h
        spk_and_feat_name = ('/').join(filename.split('/')[-2:]).replace('.p','.mat') # ex) 322F3065/SNR322F3MIC065188.mat
        label_path = os.path.join(label_folder, spk_and_feat_name)
    if 'kaist_10h' in filename:
        label_folder = os.path.join(c.LABEL_DIR, 'kaist_10h') # ex) /home/dudans/Desktop/DB/sohn_result/kaist_10h
        spk_and_feat_name = ('/').join(filename.split('/')[-2:]).replace('.p','.mat') # ex) 322F3065/SNR322F3MIC065188.mat
        label_path = os.path.join(label_folder, spk_and_feat_name)
    return label_path

def read_wav_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_waves(directory) # filename
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def read_DB_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_files(directory) # filename (feature)
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['label_path'] = DB['filename'].apply(lambda x: convert_filename_to_labelpath(x)) # label path
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) # dataset folder name
    #num_speakers = len(DB['speaker_id'].unique())
    #logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def test():
    DB_dir = '/home/dudans/Desktop/DB/robot_feat/dist_train_logfbank_nfilt40/1M0D_read_25h_2_noisy/'
    DB = read_DB_structure(DB_dir)
    #test_wav = read_audio(DB[0:1]['filename'].values[0])
    return DB


if __name__ == '__main__':
    DB, test_wav = test()