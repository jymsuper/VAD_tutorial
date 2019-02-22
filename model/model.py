import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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
        
        self.last = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = F.relu(self.bn1((self.fc1(x))))
        out = F.relu(self.bn2((self.fc2(out))))
        out = F.relu(self.bn3((self.fc3(out))))
        
        out = self.last(out)
        
        return out
