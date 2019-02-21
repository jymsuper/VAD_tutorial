import os
import librosa
import time
import numpy as np
import argparse
import configure as c
import torch.nn.functional as F
import matplotlib.pyplot as plt
from feature_extraction import normalize_frames
from feature_extraction_py.MRCG_extraction import MRCG
from model.model import DNN
from VAD_Dataset import *
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition Enrollment and Test')

parser.add_argument('--log-dir', default='./data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: adam)')
parser.add_argument('--kernel-size', type=int, default=3, metavar='ES',
                    help='kernel size in CNN')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='ES',
                    help='number of hidden nodes')
# Test settings
parser.add_argument('--cp-num', type=int, default=25, metavar='ES',
                    help='Which check point to load')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

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

def mk_feat(filename, feat_type):
    audio, sr = librosa.load(filename, sr=c.SAMPLE_RATE, mono=True)
    if feat_type == 'MRCG':
        #MRCG_dim = 96 
        MRCG_feat, MRCG_d_feat, MRCG_dd_feat = MRCG(audio,fs=sr,total_dim=c.FILTER_BANK)
        total_feat = np.concatenate((MRCG_feat, MRCG_d_feat, MRCG_dd_feat), axis=-1)
        # features's dim should be equal to MRCG_dim
        assert (total_feat.shape[-1] == c.FILTER_BANK), "MRCG dimension is wrong!"
        total_feat = normalize_frames(total_feat, Scale=c.USE_SCALE)
    else:
        raise NotImplementedError
    if c.USE_GLOBAL_NORM:
        train_mean, train_std = calc_global_mean_std(c.MEAN_PATH, c.STD_PATH,[])
        total_feat = global_feature_normalize(total_feat, train_mean, train_std)
    #TI = truncatedinputfromMFB_test()
    TI = truncatedinputfromMFB_test_one_utt()
    TT = totensor_DNN_input_test()
    dummy_labels = np.zeros((len(total_feat),1))
    #dummy_labels = torch.zeros(len(total_feat))
    total_feat, _ = TI(total_feat, dummy_labels) # size : (n_frames, n_win, n_dim)
    total_feat, _ = TT(total_feat, dummy_labels) # size : (n_frames, n_win x n_dim)
    return audio, total_feat

def load_model():
    LOG_DIR = args.log_dir + '/dnn-run-optim_{}-lr{}-wd{}-k{}-hidden{}'\
    .format(args.optimizer, args.lr, args.wd, args.kernel_size, args.hidden_size)
    input_size = c.FILTER_BANK * c.NUM_FRAMES
    model = DNN(input_size=input_size, hidden_size=args.hidden_size, num_classes=2)
    checkpoint = torch.load(LOG_DIR + '/checkpoint_' + str(args.cp_num) + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def one_utt_test(one_utt_test_path, feat_type, thres, min_s, min_ns, device):
    t0 = time.time()
    audio, feature = mk_feat(one_utt_test_path, feat_type)
    print("running on %s for feature extraction %.4f s" %(device, time.time()-t0))
    model = load_model()
    if device == 'gpu':
        feature = feature.cuda()
        model.cuda()
    out = model(feature) # (n_T-n_win, 2)
    softmax_out = F.softmax(out, dim=-1) # size : (n_T-n_win, 2)
    t1 = time.time()
    softmax_out = softmax_out.cpu()
    vad_result, hangover_result = vad_from_softmax(softmax_out, thres, min_s, min_ns)
    print("running on %s %.4f s" %(device, t1-t0))
    vad_plot(audio=audio, vad_result=vad_result, hangover_result=hangover_result, fs=16000.)
    return hangover_result

def main():
    one_utt_test_dir = '/home/dudans/Desktop/robot_VAD_pytorch_py2.py/demo_utt/'
    """
    comp-alphabot-ym-4s.wav
    comp-alphabot-mh-4s.wav
    jazz-mirinae-ym-3s.wav
    jazz-mirinae-mh-3s.wav
    cleaner-mirinae-ym-4s.wav
    cleaner-mirinae-mh-4s.wav
    ssuljeon-alphabot-ym-6s.wav
    ssuljeon-alphabot-mh-6s.wav
    """
    one_utt_test_name = 'ssuljeon-alphabot-mh-6s.wav'
    one_utt_test_path = os.path.join(one_utt_test_dir, one_utt_test_name)
    feat_type = 'MRCG'
    device = 'cpu' # 'gpu' or 'cpu'
    thres = 0.4
    min_s = 0.2
    min_ns = 0.04 # 0.04
    one_utt_test(one_utt_test_path, feat_type, thres, min_s, min_ns, device)

if __name__ == '__main__':
    main()