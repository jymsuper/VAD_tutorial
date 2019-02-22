import torch
#import cPickle as pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import pickle # For python3 
import numpy as np
import scipy.io
import configure as c
from DB_reader import read_DB_structure

def convert_wav_to_feat_name(filename, mode, feat_type):
    """
    Converts the wav dir (in DB folder) to feat dir(in feat folder)
    ex) input : '/data/DB/Speaker_robot_train_DB/Etri_sponspeech/0271M3/0271M30FC204.wav'
    output_foldername : '/home/administrator/.../data/train_logfbank_nfilt40/Etri_sponspeech/0271M3/'
    output_filename : '/home/administrator/.../data/train_logfbank_nfilt40/Etri_sponspeech/0271M3/0271M30FC204.p'
    """
    filename_only = filename.split('/')[-1].replace('.wav','.p') # ex) 0271M30FC204.p (pickle format)
    speaker_folder = filename.split('/')[-2] # ex) 0271M3
    dataset_folder = filename.split('/')[-3] # ex) Etri_sponspeech
    if feat_type == 'fbank':
        if c.USE_LOGSCALE == True:
            feat_type = 'logfbank'
    
    if mode == 'train':
        # ex) data/train_logfbank_nfilt40
        # "data_src" for source file or "data_dist" for distant file
        if c.USE_DELTA == True:
            root_folder = 'robot_feat/' + 'dist_' + 'train_' + feat_type + '_nfilt' + str(c.FILTER_BANK) + '_del2'
        else:
            root_folder = 'robot_feat/' + 'dist_' + 'train_' + feat_type + '_nfilt' + str(c.FILTER_BANK)
    elif mode == 'test':
        dist_type = filename.split('/')[-4] + '_' # ex) 1M0D
        noise_type = dataset_folder.split('_')[-2] + '_' # ex) refri
        SNR_type = dataset_folder.split('_')[-1] + '_' # ex) snr15

        # ex) data/test_1M0D_refri_snr15_logfbank_nfilt40
        if c.USE_DELTA == True:
            root_folder = 'robot_feat/' + 'test_' + dist_type + noise_type + SNR_type + feat_type + '_nfilt' + str(c.FILTER_BANK) + '_del2'
        else:
            root_folder = 'robot_feat/' + 'test_' + dist_type + noise_type + SNR_type + feat_type + '_nfilt' + str(c.FILTER_BANK)
    output_foldername = os.path.join(c.FEAT_DIR, root_folder, dataset_folder, speaker_folder)
    output_filename = os.path.join(output_foldername, filename_only)
    return output_foldername, output_filename

# For loader
def read_MFB(feat_path, label_path):
    with open(feat_path, 'rb') as f:
        feat_and_label = pickle.load(f, encoding='latin1') 
    feature = feat_and_label['feat'] # size : (n_frames, dim=40)
    
    label = scipy.io.loadmat(label_path)
    label = label['vad_result'] # size : (n_frames, 1)
    # feature and label should be same length! (change feature length)
    if len(feature)!=len(label):
        feature = feature[0:len(label)]
    
    return feature, label

class truncatedinputfromMFB(object):
    def __init__(self, input_per_file=1):
        super(truncatedinputfromMFB, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features, labels):
        network_inputs = []
        num_frames = len(frames_features)
        import random
        
        for i in range(self.input_per_file):
            
            j = random.randrange(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME)
            if not j:
                frames_slice = np.zeros(c.NUM_FRAMES, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                if c.NUM_PREVIOUS_FRAME == c.NUM_NEXT_FRAME == 0 :
                    frames_slice = frames_features
                else:
                    frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
            network_inputs.append(frames_slice)
        # input size : (n_frames, dim=40)
        # output size : (1, n_win=40, dim=40) => need to remove first dimension for DNN input
        return np.array(network_inputs), labels[j]

class truncatedinputfromMFB_test(object):
    """
    input size - feat: (#T, dim), label:(#T,1)
    output size - feat:(#T-n_win,n_win,dim), label:(#T-n_win,)
    """
    def __init__(self, input_per_file=1):
        super(truncatedinputfromMFB_test, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features, labels):
        network_inputs = []
        network_targets = []
        num_frames = len(frames_features)
        
        for i in range(self.input_per_file):
            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                # network_inputs.append(np.reshape(frames_slice, (32, 20, 3)))
                network_inputs.append(frames_slice)
                network_targets.append(labels[j])
        return np.array(network_inputs), np.array(network_targets).squeeze(1)

class truncatedinputfromMFB_test_one_utt(object):
    """
    input size - feat: (#T, dim), label:(#T,1)
    output size - feat:(#T ,n_win,dim), label:(#T,)
    """
    def __init__(self, input_per_file=1):
        super(truncatedinputfromMFB_test_one_utt, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features, labels):
        network_inputs = []
        network_targets = []
        num_frames = len(frames_features)
        
        for i in range(self.input_per_file):
            for j in range(0, num_frames):
                if j < c.NUM_PREVIOUS_FRAME:
                    frames_slice = np.zeros_like(frames_features[0:c.NUM_PREVIOUS_FRAME+c.NUM_NEXT_FRAME])
                    frames_slice[-(j+c.NUM_NEXT_FRAME):] = frames_features[:j+c.NUM_NEXT_FRAME]
                elif j > num_frames-c.NUM_NEXT_FRAME:
                    frames_slice = np.zeros_like(frames_features[0:c.NUM_PREVIOUS_FRAME+c.NUM_NEXT_FRAME])
                    frames_slice[:c.NUM_PREVIOUS_FRAME+num_frames-j] = frames_features[-(num_frames-j+c.NUM_PREVIOUS_FRAME):]
                else:
                    frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                # network_inputs.append(np.reshape(frames_slice, (32, 20, 3)))
                network_inputs.append(frames_slice)
                network_targets.append(labels[j])
        return np.array(network_inputs), np.array(network_targets).squeeze(1)

class truncatedinputfromMFB_CNN_test(object):
    def __init__(self, input_per_file=1):
        super(truncatedinputfromMFB_CNN_test, self).__init__()
        self.input_per_file = input_per_file
    def __call__(self, frames_features, labels):
        network_inputs = []
        network_targets = []
        num_frames = len(frames_features)
        for i in range(self.input_per_file):
            for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
                frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
                #network_inputs.append(np.reshape(frames_slice, (-1, c.NUM_PREVIOUS_FRAME+c.NUM_NEXT_FRAME, c.FILTER_BANK)))
                network_inputs.append(frames_slice)
                network_targets.append(labels[j])
        # network inputs:(n_frames, n_win, n_dim)
        network_inputs = np.expand_dims(network_inputs, axis=1) # (n_frames, 1, n_win, n_dim)
        assert network_inputs.ndim == 4, 'Data is not a 4D tensor. size:%s' % (np.shape(network_inputs),)
        return np.array(network_inputs), np.array(network_targets).squeeze(1)

class totensor_DNN_input(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature, label):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor. size:(1,T,dim)
        Returns:
            torch tensor: Converted feature. size:(1,Txdim)
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            # ten_feature = torch.from_numpy(np_feature) # output type => torch.DoubleTensor
            # ten_feature = torch.FloatTensor(np_feature) # output type => torch.FloatTensor, but slow
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            label = torch.from_numpy(label).long()
            #in_size = ten_feature.size(0)
            #ten_feature = ten_feature.view(in_size,-1)
            ten_feature = ten_feature.view(-1)
            #return img.float()
            # feature = torch.FloatTensor(feature.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return ten_feature, label
            #img = torch.from_numpy(pic)
            # backward compatibility

class totensor_DNN_input_test(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature, label):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor. size:(1,T,dim)
        Returns:
            torch tensor: Converted feature. size:(1,Txdim)
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            # ten_feature = torch.from_numpy(np_feature) # output type => torch.DoubleTensor
            # ten_feature = torch.FloatTensor(np_feature) # output type => torch.FloatTensor, but slow
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            label = torch.from_numpy(label).long()
            in_size = ten_feature.size(0)
            ten_feature = ten_feature.view(in_size,-1)
            #ten_feature = ten_feature.view(-1)
            #return img.float()
            # feature = torch.FloatTensor(feature.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return ten_feature, label
            #img = torch.from_numpy(pic)
            # backward compatibility

class totensor_CNN_input(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature, label):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            # ten_feature = torch.from_numpy(np_feature) # output type => torch.DoubleTensor
            # ten_feature = torch.FloatTensor(np_feature) # output type => torch.FloatTensor, but slow
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() # output type => torch.FloatTensor, fast
            label = torch.from_numpy(label).long()
            # feature = torch.FloatTensor(feature.transpose((0, 2, 1)))
            # input size : (1, n_win=40, dim=40)
            # output size : (1, dim=40, n_win=40)
            return ten_feature, label

class totensor_CNN_test_input(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, np_feature, label):
        """
        Args:
            feature (numpy.ndarray): feature to be converted to tensor.
        Returns:
            Tensor: Converted feature.
        """
        if isinstance(np_feature, np.ndarray):
            # handle numpy array
            # ten_feature = torch.from_numpy(np_feature) # output type => torch.DoubleTensor
            # ten_feature = torch.FloatTensor(np_feature) # output type => torch.FloatTensor, but slow
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() # output type => torch.FloatTensor, fast
            label = torch.from_numpy(label).long()
            #return img.float()
            # feature = torch.FloatTensor(feature.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            # input size : (1, n_win=40, dim=40)
            # output size : (1, dim=40, n_win=40)
            return ten_feature, label

class VAD_Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, feat, label):
        for t in self.transforms:
            feat, label = t(feat, label)
        return feat, label
          
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

from utils import *

class VAD_Dataset(data.Dataset):
    def __init__(self, DB, loader, transform=None, *arg, **kw):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        #self.spk_to_idx = spk_to_idx
        if c.USE_GLOBAL_NORM:
            self.train_mean, self.train_std = calc_global_mean_std(c.MEAN_PATH, c.STD_PATH,[])
    
    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        label_path = self.DB['label_path'][index]
        feature, label = self.loader(feat_path, label_path)
        # label is not a constant. It is a vector which corresponds to each feature
        #seed = np.random.randint(2147483647) # make a seed with numpy generator 
        #random.seed(seed) # apply this seed to feat tranfsorms
        if c.USE_GLOBAL_NORM:
            feature = global_feature_normalize(feature, self.train_mean, self.train_std)
            
        if self.transform:
            feature, label = self.transform(feature, label)
        #random.seed(seed) # apply this seed to target tranfsorms
        #if self.transform:
        #    target = self.target_transform(target)
        
        return feature, label
    
    def __len__(self):
        return self.len
        
def main():
    train_DB = read_DB_structure(c.TRAIN_DATAROOT_DIR)
    transform = transforms.Compose([
        truncatedinputfromMFB(),
        totensor_DNN_input()
    ])
    file_loader = read_MFB
    batch_size = 64
    VAD_train_dataset = VAD_Dataset(DB=train_DB, loader=file_loader, transform=transform)
    VAD_train_loader = torch.utils.data.DataLoader(dataset=VAD_train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

if __name__ == '__main__':
    main()