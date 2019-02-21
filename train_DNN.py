# yh computer, python2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
import time
import shutil
import warnings

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from logger import Logger
#import sys
#sys.path.append('/home/administrator/Desktop/Speaker_recognition_robot')
#from utils import *

import configure as c
import pandas as pd
from DB_wav_reader import read_DB_structure
from VAD_Dataset import *
#from model.model import backgroundCNN
#from model.model import background_resnet
from model.model import DNN
from utils import *
#from CenterLoss import CenterLoss

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Voice activity detection')
# Model options
parser.add_argument('--log-dir', default='./data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    #default='./data/pytorch_speaker_logs/cnn_run_optim_adam-lr0.0001-wd0.0-k3-embeddings512-hidden256/checkpoint_9.pth',
                    default=None,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--model-types', type=int, default=1, metavar='MT',
                    help='DNN:1, CNN:2')
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='shuffle or not')
parser.add_argument('--kernel-size', type=int, default=3, metavar='ES',
                    help='kernel size in CNN')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='HS',
                    help='number of hidden nodes')
parser.add_argument('--batch-size', type=int, default=256, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=8, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

parser.add_argument('--min-softmax-epoch', type=int, default=2, metavar='MINEPOCH',
                    help='minimum epoch for initial parameter using softmax (default: 2')

parser.add_argument('--lsoftmax-margin', type=int, default=4, metavar='M',
                    help='the margin for the l-softmax formula (m=1, 2, 3, 4)')

parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=0, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: adam)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=200, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--mfb', action='store_true', default=True,
                    help='start from MFB file')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def select_train_DB(train_dataroot_dir, DB_list, kaist_10h_spk_list):
    train_DB_all = read_DB_structure(train_dataroot_dir)
    train_DB = pd.DataFrame()
    for i in DB_list:
        if 'kaist_10h_' in i: # select only kaist_10h noisy train DB
            for j in kaist_10h_spk_list:
                ith_train_DB = train_DB_all[train_DB_all['dataset_id'] == i]
                train_DB = train_DB.append(ith_train_DB[ith_train_DB['speaker_id'] == j], ignore_index=True)
        else:
            train_DB = train_DB.append(train_DB_all[train_DB_all['dataset_id'] == i], ignore_index=True)
    return train_DB

def get_kaist_10h_spk_list():
    """
    Get a kaist_10h speaker list in which the speakers are not used in the enrollment and test
    """
    noisy_enroll_test_DB = read_DB_structure(c.TEST_DATAROOT_DIR)
    any_filename = noisy_enroll_test_DB['filename'][0]
    any_spk_folder,_= convert_wav_to_feat_name(any_filename, 'test')
    data_dir = os.path.dirname(any_spk_folder)
    data_dir = os.path.dirname(data_dir) # '/home/..../test_1M0D_refmotor_snr103c_logfbank_nfilt40'
    enroll_test_spk_list = select_feat_folders_by_num_utts(data_dir,78) # speakers which have more than threshold utts
    total_list = [str(item).zfill(3) for item in list(range(1, 105+1))]
    train_list = [item for item in total_list if item not in enroll_test_spk_list]
    return train_list

def main():
    # set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
    # order to prevent any memory allocation on unused GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    np.random.seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.cuda:
        cudnn.benchmark = True
    LOG_DIR = args.log_dir + '/dnn-run-optim_{}-lr{}-wd{}-k{}-hidden{}' \
        .format(args.optimizer, args.lr, args.wd, args.kernel_size, args.hidden_size)
    # create logger
    logger = Logger(LOG_DIR)
    
    DB_list = ['1M0D_spon_22h_2_noisy_sil','3M0D_spon_22h_2_noisy_sil','SRC_spon_22h_noisy_sil',\
               '1M0D_read_25h_2_noisy_sil','3M0D_read_25h_2_noisy_sil','SRC_read_25h_noisy_sil']
    #kaist_10h_spk_list = get_kaist_10h_spk_list()
    train_DB = select_train_DB(c.TRAIN_DATAROOT_DIR, DB_list, kaist_10h_spk_list=[])
    
    if c.USE_GLOBAL_NORM:
        train_mean, train_std = get_global_mean_std(train_DB)
    ## select train DB
    #train_DB = pd.DataFrame()
    #train_DB = train_DB.append(train_DB_all[train_DB_all['dataset_id']=='1M0D_read_25h_2'], ignore_index=True)
    if args.makemfb:
        train_feat_extraction(args.dataroot)
        print("Complete convert for training")
    if args.mfb:
        transform = VAD_Compose([
            #MyJointOp(),
            truncatedinputfromMFB(), #(1, n_win=40, dim=40)
            totensor_DNN_input()
        ])
        file_loader = read_MFB
    else:
        transform = transforms.Compose([
            truncatedinput(),
            toMFB(),
            totensor(),
            # tonormal()
        ])
        file_loader = read_audio
    
    VAD_train_dataset = VAD_Dataset(DB=train_DB, loader=file_loader, transform=transform)
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    input_size = c.FILTER_BANK * c.NUM_FRAMES
    print('\nInput dimension for training:\n{}\n'.format(input_size))
    # instantiate model and initialize weights
    model = DNN(input_size=input_size, hidden_size=args.hidden_size, num_classes=2)
    #model = backgroundCNN(kernel_size=args.kernel_size, hidden_size=args.hidden_size,
    #                      embedding_size=args.embedding_size, num_classes=n_classes)

    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    model = nn.DataParallel(model)

    if args.cuda:
        model.cuda()
    start = args.start_epoch
    end = start + args.epochs
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu_id)
    optimizer = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
    VAD_train_loader = torch.utils.data.DataLoader(
        dataset=VAD_train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
        num_workers=args.workers)
    for epoch in range(start, end):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(VAD_train_loader, model, criterion, optimizer, epoch)
        # do checkpointing
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # train_acc = AverageMeter()
    n_correct, n_total = 0, 0
    # switch to train mode
    model.train()

    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data # target size:(batch size,1), input size:(batch size, dim x win)
        #print(inputs.shape)
        #print(targets.shape)
        current_sample = inputs.size(0) # batch size
        # delete the channel axis for DNN input
        #inputs = inputs.resize_(1 * current_sample, inputs.size(2)) # size:(batch size, dim=1600)
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = Variable(inputs)
        targets = Variable(targets)

        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        out = model(inputs) # out size:(batch size, #classes=2)
        #print(out.shape)
        #_, out = model(inputs, targets)  # out size:(batch size, #classes), for lsoftmax

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(out, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc = 100. * n_correct/n_total
        # train_acc.update(train_acc_temp, inputs.size(0))
        #print(targets.size())
        targets = targets.squeeze_()
        #print(targets.size())
        #cross entropy
        #Input: (N,C) where C = number of classes
        #Target: (N) where each value is 0=<targets[i]=<C-1
        loss = criterion(out, targets)
        losses.update(loss.item(), inputs.size(0))
        # loss = temp_loss / current_sample # average the loss by minibatch
        inf = float("inf")
        loss_value = loss.data.item()
        if loss_value == inf or loss_value == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        # avg_loss += loss_value
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {train_acc:.4f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, train_acc=train_acc))
        # len(train_loader.dataset) : # of utterances in all DB = 216753
        # len(train_loader) : # of batches = 216753/batchsize(128) = 1694
        # len(inputs) : batch size = 128


if __name__ == '__main__':
    main()