import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os
import configure as c
from VAD_Dataset import read_MFB

def eer(label,pred):
    FAR, TPR, threshold = roc_curve(label, pred, pos_label=1)
    MR = 1-TPR
    EER = FAR[np.nanargmin(np.absolute((MR - FAR)))]
    return FAR, MR, EER

def global_feature_normalize(feature, train_mean, train_std):
    mu = train_mean
    sigma = train_std
    return (feature-mu)/sigma

def train_mean_std(train_DB):
    
    print("Start to calculate the global mean and std of train DB")
    """ Calculate the global mean and std of train DB """
    n_files = len(train_DB)
    train_mean = 0.
    train_std = 0.
    n_frames = 0.
    # Calculate the global mean of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        label_path = train_DB['label_path'][i]
        inputs, targets = read_MFB(filename, label_path) # input shape : (n_frames, n_dim)
        temp_n_frames = len(inputs) # number of frames
        train_mean += np.sum(inputs, axis=0, keepdims=1) # shape : (1, n_dim)
        n_frames += temp_n_frames
    train_mean = train_mean/n_frames
    # Calculate the global std of train DB
    for i in range(n_files):
        filename = train_DB['filename'][i]
        label_path = train_DB['label_path'][i]
        inputs, targets = read_MFB(filename, label_path) # input shape : (n_frames, n_dim)
        deviation = np.sum((inputs - train_mean)**2, axis=0, keepdims=1) # shape : (1, n_dim)
        train_std += deviation
    train_std = train_std/(n_frames-1)
    train_std = np.sqrt(train_std)
    return train_mean, train_std

def calc_global_mean_std(mean_path, std_path, train_DB):
    try:
        mean = np.loadtxt(mean_path, delimiter='\n')
        mean = np.expand_dims(mean,0)
        std = np.loadtxt(std_path, delimiter='\n')
        std = np.expand_dims(std,0)
        #print("The global mean and std of train DB are loaded from saved files")
        return mean,std
    except:
        mean, std = train_mean_std(train_DB)
        np.savetxt(mean_path, mean, delimiter='\n')
        np.savetxt(std_path, std, delimiter='\n')
        print("The global mean and std of train DB are saved")
        return mean,std
        
def get_global_mean_std(train_DB):
    if not os.path.exists(c.MEAN_STD_PATH):
        os.makedirs(c.MEAN_STD_PATH)
    train_mean, train_std = calc_global_mean_std(c.MEAN_PATH, c.STD_PATH, train_DB)
    return train_mean, train_std