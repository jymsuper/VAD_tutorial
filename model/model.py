import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import resnet as resnet
#import encoding
#from lsoftmax import LSoftmaxLinear

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1_drop = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2_drop = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3_drop = nn.Dropout(p=0.2)

        #self.fc4 = nn.Linear(hidden_size, hidden_size)
        #self.bn4 = nn.BatchNorm1d(hidden_size)
        #self.fc4_drop = nn.Dropout(p=0.2)
        # self.fc5 = nn.Linear(hidden_size, embedding_size)
        # self.bn5 = nn.BatchNorm1d(embedding_size)
        # self.fc5_drop = nn.Dropout(p=0.2)
        self.last = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))
        #out = self.fc1_drop(F.relu(self.bn4((self.fc4(out)))))
        # out = self.fc4_drop(F.relu(self.bn4((self.fc4(out)))))
        out = self.last(out)
        #print(out.size())
        return out
        
class CNN(nn.Module):
    def __init__(self, kernel_size, hidden_size, num_classes=2):
        super(CNN, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),  # 40x40 => 20x20
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),  # 20x20 => 10x10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),  # 10x10 => 5x5
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2))
        # self.conv3 = nn.Sequential(
            # nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2)),  # 5x5 => 3x3
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.MaxPool2d((2,2), stride=2))
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(3840, hidden_size)
        self.bn0 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.last = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        #print(x.size())
        #x = x.view(x.size(0),x.size(3),x.size(1),x.size(2))
        # now x: minibatch x 12XX x 128
        # print(x.size())
        out = self.conv0(x)
        #print(out.size())
        out = self.conv1(out)
        out = self.conv2(out)
        #print(out.size())
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.bn0(self.fc0(out)))
        out = F.relu(self.bn1(self.fc1(out)))
        out = self.last(out)
        return out
        
class RNN(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, 3, bidirectional=True, dropout=0.25)
        self.linear = nn.Linear(2*64, num_classes)  # 2 for bidirection

    def forward(self, x):
        output, hidden = self.lstm(x, None)\

        #print("Input size", x.size()) # (seq_len, batch_size, n_dim)
        #print("RNN output size", output.size()) # (seq_len, batch_size, 2*cell_size)
        #output = self.linear(output[-1])
        output = self.linear(output[-1,:,:])
        #print("FC output size", output.size()) # (batch_size, num_classes)

        return output