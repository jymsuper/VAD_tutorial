# VAD_tutorial [미완성]

LG전자 서초R&D캠퍼스 강의 - voice activity detection (VAD) 튜토리얼

## 필요 환경
python 3.5+  
pytorch 1.0.0  
pandas 0.23.4  
numpy 1.13.3  
pickle 4.0  
matplotlib 2.1.0  

## * DB
원거리 음성대화 로봇과제 2차년도 수집 DB
- SNS단문 낭독 음성 DB (ETRI 낭독체)
- 원거리, 무잡음 음성, 1m 거리, 0도 방향, 16kHz, 16bits  

위의 DB를 이용하여 추출한 multi-resolution cochleagram (MRCG) feature를 업로드하였습니다.

### 1. 훈련 DB
10000개 파일, 100개 폴더(100명 화자)  
feat_MRCG_nfilt96 - train

### 2. 테스트 DB
20개 파일, 10개 폴더(10명 화자)  
feat_MRCG_nfilt96 - test

## 사용법
### 1. 훈련
train.py  

### 2. 테스트
test.py  



## 문의
dudans@kaist.ac.kr
