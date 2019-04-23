# VAD_tutorial
A pytorch implementation of full-connected DNN based voice activity detection (VAD).  
All the features for training and testing are uploaded.  
Korean manual is included ("190225_LG-AI_VAD.pdf").

## Requirements
python 3.5+  
pytorch 1.0.0  
pandas 0.23.4  
numpy 1.13.3  
pickle 4.0  
matplotlib 2.1.0  
sklearn 0.20.2

## Datasets
We used the dataset collected through the following task.
- No. 10063424, 'development of distant speech recognition and multi-task dialog processing technologies for in-door conversational robots'

Specification
- Korean read speech corpus (ETRI read speech)
- Clean speech at a distance of 1m and a direction of 0 degrees
- 16kHz, 16bits  

We uploaded multi-resolution cochleagram (MRCG) features extracted from the above dataset.  
[python based MRCG feature extraction toolkit](https://github.com/zouxinghao/MRCG) is used.

### * Train
10000 utterances, 100 folders (100 speakers)  
Size : 4.4GB  
```feat_MRCG_nfilt96 - train```

### * Test
20 utterances, 10 folders (10 speakers)  
Size : 18MB  
```feat_MRCG_nfilt96 - test```

## Usage
### 1. Training
```python train.py```  

### 2. Testing
```python test.py```  


## Author
Youngmoon Jung (dudans@kaist.ac.kr) at KAIST, South Korea
