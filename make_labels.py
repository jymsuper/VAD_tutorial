# yh computer. Not completed 
from glob import glob
#from feature_extraction_py.MRCG_extraction import MRCG

import os
import numpy as np
import librosa
import re
#import h5py 
import scipy.io
import _pickle as pickle

"""
1. Load the sohn result for each feature (train, test)
2. Make the length of two files same
3. Save the vad label
"""

fs = 16000
MRCG_dim = 96 

# Training DB config
current_path = os.getcwd() # '/home/dudans/Desktop/robot_VAD_pytorch'
train_feat_folder = '/home/dudans/Desktop/DB/robot_feat/dist_train_logfbank_nfilt40' # read and spon noisy wav path
train_feat_type = '1M0D_read_25h_2_noisy'
train_path = os.path.join(train_folder, train_feat_type)
label_path = '/home/dudans/Desktop/DB/sohn_result' # read, spon ,kaist_10h label
train_label_path = '/home/dudans/Desktop/DB/robot_label/'

# Test DB config
dist_degree = '1M0D' # '1M0D' or '3M0D'
noise_type = ['happytogether1','iphonebell1','cleaner','aircon','ssuljeon1','refmotor','samsungbell1','LGbell1','jazz']
SNR = [-5,0,5,10,15]
mic_chan = '01'
robot_path = '/home/administrator/Desktop/robot_VAD/robot_2Y_16k_noisy_wave/' + dist_degree
robot_MRCG_path = current_path + '/robot_2Y_16k_MRCGfeature_96dim_test_py'
robot_label_path = current_path + '/sohn_result/' + dist_degree + '/kaist_10h'

def wavpath_to_p(file_name, pattern):
    mc = pattern.findall(file_name)
    mc = mc[0].replace("/","") # ex) '/5620_40io030g.wav' -> '5620_40io30g.wav'
    mc = mc.replace(".wav",".p") # ex) '5620_40io030g.wav' -> '5620_40io30g.p'     
    return mc

def get_label(file_name, dataset):
    if dataset == "train":
        label_sub_folder = '_'.join(train_feat_type.split('_')[1:3]) # ex) 'read_25h' 
        label_file = label_path + label_sub_folder + file_name
        label = scipy.io.loadmat(label_file)
        label = label['final_label']
    elif dataset == "test":
        label = scipy.io.loadmat(file_name)
        label = label['hangover_result']
    return label

def make_num_f_num_l_same(feature, label):
"""
input : feature, label (probably with different lengths), size: (T1, dim), (T2,)
output : feature (with the same length as the label), size : (T1, dim)
"""
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
    
def label_generator(dataset):
    # Feature extraction of training set
    if dataset == 'train':
        #train_wav_files = glob(os.path.join(train_path, "*.wav"))
        pattern_mat = re.compile("_\w+.wav")
        pattern_p = re.compile("/\d+_\w+.wav")
        count = 0
        
        if not os.path.exists(train_label_path):
            os.makedirs(train_label_path)
        
        for (path, dir, files) in os.walk(train_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.p':
                    tot_filepath = os.path.join(path, filename)
                    # Load the feature using tot_filepath
                    with open(tot_filepath, 'rb') as f:
                        feat_and_spk = pickle.load(f)
                        feat = feat_and_spk['feat']
                        
                        
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
    label_generator('train')
    #label_generator('test')
        
if __name__ == '__main__':
    main()
