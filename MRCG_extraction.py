from glob import glob
from feature_extraction_py.MRCG_extraction import MRCG

import os
import numpy as np
import librosa
import re
import h5py 
import scipy.io
import _pickle as pickle
#import cPickle as pickle
#import json

"""
* Training DB
1. Generate 32h noisy wave files using <matlab> with 100 types noises (+2 second silence padding) 
    => 'Au4_16k_noisy_wav_35h_sil' folder
2. Generate new labels after silence padding using <matlab>
    => 'Au4_label_silence_padded' folder
3. Feature extraction using <python> (MRCG or log-mel FB) 
    => 'Au4_16k_MRCGfeature_96dim_NI_35h_sil_py' folder

* Test DB
1. Generate wave files for each noise condition using <matlab> without silence padding
    => ex) 'robot_2Y_16k_noisy_wave/1M0D/aircon_ch01_SNR0' folders
2. Generate labels without silence padding using <matlab>
    => 'sohn_result/dist_degree/kaist_10h/' folder
3. Feature extraction using python (MRCG or log-mel FB)
    => 'robot_2Y_16k_MRCGfeature_96dim_test_py' folder

# For small test data set
fs=16000
MRCG_dim = 96 
file_name = 'jazz-mirinae-mh-3s'
test_wav_file = '/home/administrator/Desktop/robot_VAD/robot_DB_test/'+ file_name + '.wav'
MRCG_test_path = '/home/administrator/Desktop/robot_VAD/robot_DB_test/'+ file_name + '.p'
MRCG_tot_feat = load_and_feat_extraction(test_wav_file, fs)
with open(MRCG_test_path, 'w') as fp:
    pickle.dump(MRCG_tot_feat, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""

fs = 16000
MRCG_dim = 96 

# Training DB config
current_path = os.getcwd() # '/home/administrator/Desktop/robot_VAD_pytorch'
train_folder = '/data/DB/Speaker_robot_train_DB_dist' # read and spon noisy wav path
train_DB_type = '1M0D_read_25h_2_noisy'
train_path = os.path.join(train_folder, train_DB_type)

train_label_path = '/data/DB/sohn_result' # read, spon ,kaist_10h label
train_MRCG_folder = '/data/DB/robot_VAD_MRCG_96dim_py' # where to save MRCG feature
train_MRCG_path = os.path.join(train_MRCG_folder, train_DB_type)

# Test DB config
dist_degree = '1M0D' # '1M0D' or '3M0D'
noise_type = ['happytogether1','iphonebell1','cleaner','aircon','ssuljeon1','refmotor','samsungbell1','LGbell1','jazz']
SNR = [-5,0,5,10,15]
mic_chan = '01'
robot_path = '/home/administrator/Desktop/robot_VAD/robot_2Y_16k_noisy_wave/' + dist_degree
robot_MRCG_path = current_path + '/robot_2Y_16k_MRCGfeature_96dim_test_py'
robot_label_path = current_path + '/sohn_result/' + dist_degree + '/kaist_10h'

def load_and_feat_extraction(file_name, fs):
    x, sr = librosa.load(file_name, sr=fs)
    MRCG_feat, MRCG_d_feat, MRCG_dd_feat = MRCG(x,fs=sr,total_dim=MRCG_dim)
    MRCG_tot_feat = np.concatenate((MRCG_feat, MRCG_d_feat, MRCG_dd_feat), axis=-1)
    # MRCG_tot_feat's dim should be equal to MRCG_dim
    assert (MRCG_tot_feat.shape[-1] == MRCG_dim), "MRCG dimension is wrong!"
    
    return MRCG_tot_feat

def wavpath_to_p(file_name, pattern):
    mc = pattern.findall(file_name)
    mc = mc[0].replace("/","") # ex) '/5620_40io030g.wav' -> '5620_40io30g.wav'
    mc = mc.replace(".wav",".p") # ex) '5620_40io030g.wav' -> '5620_40io30g.p'     
    return mc

# ex) '~/5620_20cc0104.wav' -> '20cc0104.mat'    
def wavpath_to_mat(file_name, pattern, dataset):
    mc = pattern.findall(file_name)
    if dataset == "train":
        mc = mc[1].replace("_","") # wave file's name (without directory). ex) '20cc0104.wav'
    elif dataset == "test":
        mc = mc[0].replace("/","")
    mc = mc.replace(".wav",".mat") # change ".wav" to ".mat"     
    return mc

def get_label(file_name, dataset):
    if dataset == "train":
        label_file = Au4_label_path + "/train_folder/" + file_name
        label = scipy.io.loadmat(label_file)
        label = label['final_label']
    elif dataset == "test":
        label = scipy.io.loadmat(file_name)
        label = label['hangover_result']
    return label

def make_num_f_num_l_same(feature, label):
    num_f = feature.shape[0]
    num_l = label.shape[0]
    
    # To make num_f == num_l (fit to num_l)
    if num_f > num_l:
        feature = feature[0:num_l,:]
        assert (feature.shape[0] == label.shape[0]), "num_f is not equal to num_l !"
    elif num_f < num_l:
        feature = np.concatenate((feature, feature[num_f-(num_l-num_f):,:]), axis=0)
        assert (feature.shape[0] == label.shape[0]), "num_f is not equal to num_l !"
    return feature


def feature_extractor(dataset):
    # Feature extraction of training set
    if dataset == 'train':
        #train_wav_files = glob(os.path.join(train_path, "*.wav"))
        pattern_mat = re.compile("_\w+.wav")
        pattern_p = re.compile("/\d+_\w+.wav")
        count = 0
        
        if not os.path.exists(train_MRCG_path):
            os.makedirs(train_MRCG_path)
        
        for (path, dir, files) in os.walk(train_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.wav':
                    tot_filepath = os.path.join(path, filename)
                    MRCG_tot_feat = load_and_feat_extraction(tot_filepath, fs)
        
        for file in sorted(train_wav_files):
            MRCG_tot_feat = load_and_feat_extraction(file, fs)
            file_name_mat = wavpath_to_mat(file, pattern_mat, dataset) # ex) '~/5620_20cc0104.wav' -> '20cc0104.mat'
            file_name_p = wavpath_to_p(file, pattern_p) # ex) ''~/5620_20cc0104.wav' -> '5620_20cc0104.p'
            Au4_label = get_label(file_name_mat, dataset) # ex) '20cc0104.mat' -> Au4_label:(T,1) column vector
            MRCG_tot_feat = make_num_f_num_l_same(MRCG_tot_feat, Au4_label) # To make num_f == num_l (fit to num_l)
            assert (MRCG_tot_feat.shape[0] == Au4_label.shape[0]), "num_f is not equal to num_l !"
            MRCG_train_feat_and_label = {'MRCG_train_feature':MRCG_tot_feat, 'MRCG_train_label':Au4_label} # MRCG feature and label 
            MRCG_train_path = Au4_MRCG_path + '/' + file_name_p # where to save
            
            if os.path.isfile(MRCG_train_path) == 1:
                print("\"" + file_name_p + "\"" + " file already extracted!")
                continue
            
            with open(MRCG_train_path, 'w') as fp:
                pickle.dump(MRCG_train_feat_and_label, fp, protocol=pickle.HIGHEST_PROTOCOL)
            count += 1
            print("MRCG feature extraction (training DB). step : %d, file : \"%s\"" %(count,file_name_p))
    
    # Feature extraction of test set
    elif dataset == 'test':
        test_wav_folders = glob(os.path.join(robot_path, "*")) # each subfolders in main folder (ex. '~/LGbell1_ch01_SNR-5')
        pattern_mat = re.compile("/\d+_\w+.wav")
        pattern_p = re.compile("/\d+_\w+.wav")
        pattern_just_folder = re.compile("/\d\d\d")
        count = 0
        
        for folder in sorted(test_wav_folders):
            if os.path.isdir(folder):
                test_wav_files = glob(os.path.join(folder, "*.wav")) # wav files per each folder (ex. '~/001_Hot_Word01_ch01.wav') 
                just_folder = os.path.basename(folder) # ex) LGbell1_ch01_SNR-5
                MRCG_test_folder = robot_MRCG_path + '/' + dist_degree + '/' + just_folder
                if not os.path.exists(MRCG_test_folder):
                    os.makedirs(MRCG_test_folder)
                for wav_file in sorted(test_wav_files):
                    if "Hot_Word" not in wav_file: # Only select the "Hot word" test file 
                        continue
                    MRCG_tot_feat = load_and_feat_extraction(wav_file, fs)
                    file_name_mat = wavpath_to_mat(wav_file, pattern_mat, dataset) # ex) '~/001_Hot_Word01_ch01.wav' -> '001_Hot_Word01_ch01.mat'
                    file_name_p = wavpath_to_p(wav_file, pattern_p) # ex) '001_Hot_Word01_ch01.p'
                    label_just_folder = pattern_just_folder.findall(wav_file)[0] # ex) '/001'
                    robot_label = get_label(robot_label_path + label_just_folder + "/" + file_name_mat, dataset) # (T,1) column vector, dtype=unit8
                    MRCG_tot_feat = make_num_f_num_l_same(MRCG_tot_feat, robot_label) # To make num_f == num_l (fit to num_l)
                    assert (MRCG_tot_feat.shape[0] == robot_label.shape[0]), "num_f is not equal to num_l !"
                    MRCG_test_feat_and_label = {'MRCG_test_feature':MRCG_tot_feat, 'MRCG_test_label':robot_label} # MRCG feature and label 
                    MRCG_test_path = MRCG_test_folder + '/' + file_name_p
                    
                    if os.path.isfile(MRCG_test_path) == 1:
                        print("\"" + file_name_p + "\"" + " file already extracted!")
                        continue            
                    with open(MRCG_test_path, 'w') as fp:
                        pickle.dump(MRCG_test_feat_and_label, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    count += 1                        
                    print("MRCG feature extraction (test DB). step : %d, file : \"%s\"" %(count,file_name_p))
               
        pass
        
def main():
    #feature_extractor('train')
    feature_extractor('test')
        
if __name__ == '__main__':
    main()
