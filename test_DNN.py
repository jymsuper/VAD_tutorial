# yh computer
from __future__ import print_function
import argparse
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tabulate import tabulate
from utils import *
#import sys
#sys.path.append('/home/administrator/Desktop/Speaker_recognition_robot')
#from utils import *
from VAD_Dataset import *
from model.model import DNN
#from model.model import background_res_DEPnet
#from model.model_deepten4096_fc512 import background_res_DeepTEN # 30 best

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
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
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

torch.cuda.set_device(int(args.gpu_id))

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def test_input_load(filename, label_path):
    inputs, targets = read_MFB(filename, label_path)  # input size :(n_frames, dim)
    if c.USE_GLOBAL_NORM:
        train_mean, train_std = calc_global_mean_std(c.MEAN_PATH, c.STD_PATH,[])
        inputs = global_feature_normalize(inputs, train_mean, train_std)
    #TI = truncatedinputfromMFB_CNN_test()
    #TT = totensor_CNN_test_input()
    TI = truncatedinputfromMFB_test()
    TT = totensor_DNN_input_test()
    inputs, targets = TI(inputs, targets) # size : (n_frames, 1, 40, 40)
    inputs, targets = TT(inputs, targets)
    #input = Variable(input, volatile=True)  # volatile option is really important for enrollment and test!!
    with torch.no_grad():
        inputs = Variable(inputs)
        targets = Variable(targets)
    return inputs, targets

def select_test_DB(test_dataroot_dir, DB_list):
    test_DB_all = read_DB_structure(test_dataroot_dir)
    test_DB = pd.DataFrame()
    for i in DB_list:
        test_DB = test_DB.append(test_DB_all[test_DB_all['dataset_id'] == i], ignore_index=True)
    return test_DB

def test(model, DB, criterion):
    n_files = len(DB)
    n_frames = 0
    n_correct = 0.
    n_total = 0.
    mean_cost = 0.
    mean_accuracy = 0.
    mean_AUC = 0.
    mean_EER = 0.
    for i in range(n_files):
        filename = DB['filename'][i]
        label_path = DB['label_path'][i]
        inputs, targets = test_input_load(filename, label_path)  # size : (n_frames, dim x win_size)
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        out = model(inputs)  # size : (n_frames, 2)
        softmax_out = F.softmax(out, dim=-1) # size : (n_frames, 2)
        temp_cost = (criterion(out, targets).data).cpu().numpy().item() # Covert to scalar value
        n_correct += (torch.max(out, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_frames += inputs.size(0) # batch size
        np_targets = (targets.data).cpu().numpy()
        np_softmax_out = (softmax_out.data).cpu().numpy()
        temp_AUC = roc_auc_score(np_targets, np_softmax_out[:,1])
        _,_, temp_EER = eer(np_targets, np_softmax_out[:,1])
        mean_cost += temp_cost/n_files
        mean_AUC += temp_AUC/n_files
        mean_EER += temp_EER/n_files
    mean_accuracy = 100. * n_correct/n_frames
    
    print("\nPerformance for \"%s\""  %DB['dataset_id'][0])
    print(tabulate([['Averaged Acc (%)', mean_accuracy], ['Averaged AUC', mean_AUC],\
                    ['Averaged EER', mean_EER], ['Averaged cost', mean_cost]],\
                    tablefmt="fancy_grid"))
    #print("------ Test for \"%s\" ------" % c.TEST_DATAROOT_DIR)
    #print("****** Averaged Acc (%) : %.6f " % mean_accuracy)
    #print("****** Averaged AUC : %.6f " % mean_AUC)
    #print("****** Averaged EER : %.6f " % mean_EER)
    #print("****** Averaged cost : %.6f " % mean_cost)
    return mean_accuracy, mean_AUC, mean_EER, mean_cost


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        cudnn.benchmark = True

    # Get test DB
    test_DB = read_DB_structure(c.TEST_DATAROOT_DIR)
    # Load model from checkpoint
    LOG_DIR = args.log_dir + '/dnn-run-optim_{}-lr{}-wd{}-k{}-hidden{}' \
        .format(args.optimizer, args.lr, args.wd, args.kernel_size, args.hidden_size)
    input_size = c.FILTER_BANK * c.NUM_FRAMES
    print('\nInput dimension for training:\n{}\n'.format(input_size))
    # instantiate model and initialize weights
    model = DNN(input_size=input_size, hidden_size=args.hidden_size, num_classes=2)

    if args.cuda:
        model.cuda()
    print('=> loading checkpoint')
    # original saved file with DataParallel
    checkpoint = torch.load(LOG_DIR + '/checkpoint_' + str(args.cp_num) + '.pth')
    
    ## create new OrderedDict that does not contain `module.`
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint['state_dict'].items():
    #    name = k[7:]  # remove `module.`
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    #model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu_id)
    test(model, test_DB, criterion)

if __name__ == '__main__':
    main()