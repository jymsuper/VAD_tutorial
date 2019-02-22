import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import time
import os
import pandas as pd
import numpy as np

import configure as c
from DB_reader import read_DB_structure
from model.model import DNN

from utils import get_global_mean_std
from VAD_Dataset import *


def load_dataset():
    # Load training set and validation set
    
    # Percentage of validation set
    val_ratio = 10 
    
    # Split training set into training set and validation set according to "val_ratio"
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) 
    
    file_loader = read_MFB # numpy array:(n_frames, n_dims)
     
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
   
    
    speaker_list = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx)
    
    return train_dataset, valid_dataset

def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_DB_structure(train_feat_dir)
    total_len = len(train_valid_DB) # 148642
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    # Split the DB into train and valid set
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    # Reset the index
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len))
    
    return train_DB, valid_DB

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # Load training set and validation set
    
    # Percentage of validation set
    val_ratio = 10 

    # Split training set into training set and validation set according to "val_ratio"
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) 
    
    # Calculate global mean and standard deviation for normalization 
    train_mean, train_std = get_global_mean_std(train_DB)
        
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