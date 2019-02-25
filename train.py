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

from utils import get_global_mean_std, read_MRCG
from VAD_Dataset import VAD_Dataset, VAD_Compose, TruncatedInputfromMRCG, ToTensorInput
import matplotlib.pyplot as plt

def load_dataset(val_ratio):
    # Load training set and validation set
    
    # Split training set into training set and validation set according to "val_ratio"
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) 

    # Calculate global mean and standard deviation for normalization 
    train_mean, train_std = get_global_mean_std(train_DB)

    file_loader = read_MRCG # numpy array:(n_frames, n_dims)
     
    transform = VAD_Compose([
        TruncatedInputfromMRCG(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(n_dims*n_frames)
    ])
    #transform_T = ToTensorDevInput()
    
    train_dataset = VAD_Dataset(DB=train_DB, loader=file_loader, transform=transform)
    valid_dataset = VAD_Dataset(DB=valid_DB, loader=file_loader, transform=transform)
    
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

def main():
    
    # Set hyperparameters
    use_cuda = True # use gpu or cpu
    val_ratio = 10 # Percentage of validation set
    hidden_size = 512
    start = 1 # Start epoch
    n_epochs = 15 # How many epochs?
    end = start + n_epochs # Last epoch

    lr = 1e-2 # Initial learning rate
    wd = 1e-4 # Weight decay (L2 penalty)
    optimizer_type = 'sgd' # ex) sgd, adam, adagrad

    batch_size = 128 # Batch size for training 
    valid_batch_size = 32 # Batch size for validation
    use_shuffle = True # Shuffle for training or not

    # Load dataset
    train_dataset, valid_dataset = load_dataset(val_ratio)
    
    log_dir = 'model_saved' # where to save checkpoints

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    input_size = c.FILTER_BANK * c.NUM_FRAMES
    print('\nInput dimension for training:\n{}\n'.format(input_size))
    
    # instantiate model and initialize weights
    model = DNN(input_size=input_size, hidden_size=hidden_size, num_classes=2)
    
    if use_cuda:
        model.cuda()
    
    # define loss function (criterion), optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_type, model, lr, wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, min_lr=1e-4, verbose=1)
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=use_shuffle)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=valid_batch_size,
                                                       shuffle=False)
                               
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    for epoch in range(start, end):
        
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, use_cuda, epoch)
        
        # evaluate on validation set
        valid_loss = validate(valid_loader, model, criterion, use_cuda, epoch)
        
        scheduler.step(valid_loss, epoch)
        
        # calculate average loss over an epoch
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        # do checkpointing
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    '{}/checkpoint_{}.pth'.format(log_dir, epoch))
                    
    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    print('Lowest validation loss at epoch %d' %minposs)
    
    # visualize the loss and learning rate as the network trained
    visualize_the_losses(avg_train_losses, avg_valid_losses)

def train(train_loader, model, criterion, optimizer, use_cuda, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    log_interval = 35
    
    # switch to train mode
    model.train()

    end = time.time()
    # pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data # target size:(batch size,1), input size:(batch size, dim x win)
        targets = targets.view(-1) # target size:(batch size)
        current_sample = inputs.size(0) # batch size
        
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        output = model(inputs) # out size:(batch size, #classes=2)
        
        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, inputs.size(0))
        
        loss = criterion(output, targets)
        losses.update(loss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'Acc {train_acc.avg:.4f}'.format(
                     epoch, batch_idx * len(inputs), len(train_loader.dataset),
                     100. * batch_idx / len(train_loader), 
                     batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg
    
def validate(val_loader, model, criterion, use_cuda, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()
    
    n_correct, n_total = 0, 0
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            targets = targets.view(-1) # target size:(batch size)
            current_sample = inputs.size(0)  # batch size
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # compute output
            output = model(inputs)
            
            # measure accuracy and record loss
            n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
            n_total += current_sample
            val_acc_temp = 100. * n_correct / n_total
            val_acc.update(val_acc_temp, inputs.size(0))
            
            loss = criterion(output, targets)
            losses.update(loss.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        print('  * Validation: '
                  'Loss {loss.avg:.4f}\t'
                  'Acc {val_acc.avg:.4f}'.format(
                  loss=losses, val_acc=val_acc))
    
    return losses.avg

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

def create_optimizer(optimizer, model, new_lr, wd):
    # setup optimizer
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  weight_decay=wd)
    return optimizer
    
def visualize_the_losses(train_loss, valid_loss):
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.2) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')

if __name__ == '__main__':
    main()