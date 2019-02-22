import torch
import torch.nn.functional as F

import os
import librosa
import time

import numpy as np
import configure as c
import matplotlib.pyplot as plt
from model.model import DNN
from DB_reader import read_DB_structure

from VAD_Dataset import TruncatedInputfromMRCGtest, ToTensorInputTest
from utils import read_MRCG, global_feature_normalize, calc_global_mean_std


def FSM_hangover(vad_result, min_s, min_ns):
    # vad_result should be a column vector
    min_s_frame = int(min_s/0.01) # If VAD=0 is maintained for 100ms, then set state = 0 
    min_ns_frame = int(min_ns/0.01) # % If VAD=1 is maintained for 40ms, then set state = 1
    
    state = 0 # 0 for non-speech, 1 for speech
    VAD = 0 # First hangover result is inactive
    D = 0 # parameter for transition
    tmp_result = np.array(vad_result) # or "tmp_result = vad_result[:]" 
    
    for i in range(1,len(tmp_result)):
        if (state == 0) and (D != min_ns_frame):
            if tmp_result[i] == 0:
                state = 0
                D = 0
                tmp_result[i] = 0
            elif tmp_result[i] == 1:
                D = D+1
                tmp_result[i] = 0
            
        if (state == 1) and (D != min_s_frame):
            if tmp_result[i] == 0:
                D = D+1
                tmp_result[i] = 1
            elif tmp_result[i] == 1:
                state = 1
                D = 0
                tmp_result[i] = 1
            
        if (state == 0) and (D == min_ns_frame):
            state = 1
            D = 0
            # tmp_result[i-2*min_ns_frame+1:i+1] = 1
            try:
                tmp_result[i-2*min_ns_frame+1:i+1] = 1
            except:
                tmp_result[i-min_ns_frame+1:i+1] = 1
            
        if (state == 1) and (D == min_s_frame):
            state = 0
            D = 0
            tmp_result[i] = 0
            
    return tmp_result

def vad_from_softmax(softmax_out, thres, min_s, min_ns):
    # tmp_result = softmax_output[:,1]/softmax_output[:,0] # shape : (T,)
    #tmp_result = np.divide(softmax_out[:,1], softmax_out[:,0]) # shape : (T,)
    #vad_result = np.multiply((tmp_result > thres),1)
    vad_result = np.multiply((softmax_out[:,1].data.numpy() > thres),1)
    hangover_result = FSM_hangover(vad_result, min_s, min_ns)
    return vad_result, hangover_result

def vad_plot(audio, vad_result, hangover_result, fs=16000.):
    # For audio
    t = range(0, len(audio))
    t = [x/fs for x in t]
    y = audio/np.max(np.abs(audio))
    
    # For vad result
    vad_temp = [x for x in vad_result for r in range(int(fs*0.01))]
    t1 = range(0, len(vad_temp))
    t1 = [x/fs for x in t1]
    
    # For hangover result
    hangover_temp = [x for x in hangover_result for r in range(int(fs*0.01))]
    t2 = range(0, len(hangover_temp))
    t2 = [x/fs for x in t2]
    
    plt.ylim(-1,2.5)
    plt.plot(t,y, label='Speech')
    plt.hold(True)
    plt.plot(t1,vad_temp, label='VAD_result')
    plt.plot(t2, [1.5*x for x in hangover_temp], 'b', label='hangover_result')
    #plt.hold(False)
    plt.xlabel('time(s)')
    plt.legend(loc='upper right')
    plt.show()
    pass

def load_audio_feat(filename):
    wav_name = os.path.join(c.TEST_WAV_DIR, filename.split('/')[-2], filename.split('/')[-1].replace('.p','.wav'))
    audio, sr = librosa.load(wav_name, sr=c.SAMPLE_RATE, mono=True)

    feat, _ = read_MRCG(filename)
    train_mean, train_std = calc_global_mean_std(c.MEAN_PATH, c.STD_PATH,[])
    total_feat = global_feature_normalize(feat, train_mean, train_std)
    
    #TI = truncatedinputfromMFB_test()
    TI = TruncatedInputfromMRCGtest()
    TT = ToTensorInputTest()
    dummy_labels = np.zeros((len(total_feat),1))
    #dummy_labels = torch.zeros(len(total_feat))
    total_feat, _ = TI(total_feat, dummy_labels) # size : (n_frames, n_win, n_dim)
    total_feat, _ = TT(total_feat, dummy_labels) # size : (n_frames, n_win x n_dim)
    return audio, total_feat
    
def load_model(use_cuda, log_dir, cp_num, hidden_size):
    input_size = c.FILTER_BANK * c.NUM_FRAMES
    model = DNN(input_size=input_size, hidden_size=hidden_size, num_classes=2)
    if use_cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
    
def one_utt_test(use_cuda, model, filename, thres, min_s, min_ns):

    audio, feature = load_audio_feat(filename)
    
    if use_cuda:
        feature = feature.cuda()
        
    out = model(feature) # (n_T-n_win, 2)
    softmax_out = F.softmax(out, dim=-1) # size : (n_T-n_win, 2)
    softmax_out = softmax_out.cpu()
    vad_result, hangover_result = vad_from_softmax(softmax_out, thres, min_s, min_ns)
    vad_plot(audio=audio, vad_result=vad_result, hangover_result=hangover_result, fs=16000.)
    return hangover_result

def main():
    
    log_dir = 'model_saved' # Where the checkpoints are saved
    test_dir = 'feat_MRCG_nfilt96/test/' # Where test features are saved

    # Settings
    use_cuda = True # Use cuda or not
    hidden_size = 512 # Dimension of speaker embeddings
    cp_num = 6 # Which checkpoint to use?

    # Load model from checkpoint
    model = load_model(use_cuda, log_dir, cp_num, hidden_size)

    # Get the dataframe for test DB
    test_DB = read_DB_structure(c.TEST_FEAT_DIR)
    
    # Set test file
    filename = test_DB['filename'][0] # choose one test file from 0 to 19 (20 test files)
    
    thres = 0.4 # Threshold for VAD decision
    min_s = 0.2 # hyperparam for hangover
    min_ns = 0.04 # hyperparam for hangover
    one_utt_test(use_cuda, model, filename, thres, min_s, min_ns)

if __name__ == '__main__':
    main()