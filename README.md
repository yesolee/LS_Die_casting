# 🏆 다이캐스팅 공정 불량 예측 모델 개선

<img src="https://github.com/yesolee/LS_Die_casting/blob/main/img/die_casting.jpg" width="500">

## 🔥 다이캐스팅이란?
- 액체화된 금속을 주조(틀, Frame)에 넣고 원하는 모양의 금속 부품을 생산하는 방법

## 📆 분석 기간
- 2024.10.21 ~ 2024.10.28 (1주)

## 🤔분석 배경
- 공정에 새로운 시스템 도입 이후, 불량률이 감소했다.
- 공정 데이터의 분포가 변화했을텐데, 기존의 예측 모델을 계속 사용해도 될까?
- 데이터 분석을 통해 모델 성능을 점검하고 리스크를 최소화 하자.

## 🚩 분석 목표
- 데이터 분포가 유의미한지 PSI를 계산해 확인 한다.
- 데이터의 분포가 바뀌었다면, 모델을 개선하고 기존 모델과 성능을 비교해본다.
- 최종 모델의 불량 예측에 중요한 변수와 관리 방안을 제시한다.

## 📊 분석 결과
- PSI : 모델에서 데이터 분포의 변화를 측정하는 지표로, 훈련 데이터와 실제 데이터 간 차이를 평가


## 📄 데이터 정보
- 총 30개의 독립 변수와 1개의 종속 변수(passorfail) 존재
- 92,015 행, 31 열

## 🔎 분석 과정
- EDA
- 전처리 / 피처 엔지니어링
- 모델 학습
- PSI 비교
- 모델 생성
- 모델 해석
- 관리 방안 도출

## 🖍 평가지표
- 데이터 분포 변화
    - PSI : 데이터 분포의 변화를 측정하는 지표로, 훈련 데이터와 실제 데이터 간 차이를 평가
    
    <img src="https://github.com/yesolee/LS_Die_casting/blob/main/img/psi.jpg">

- 모델 성능 평가
    - Recall : 불량을 잘 예측한 비율
    - ROC AUC : 모델의 전체적인 분류 성능을 측정 (ROC 곡선 아래 면적)
    - G-mean : 클래스 불균형 상황에서 정확한 예측 균형을 평가하는 지표 (Recall과 특이도의 조화 평균)

## 데이터 분할 - train/test
- train1 (기존 모델) : 2019년 1월 2일 ~ 2019년 2월 14일
- train2 (시스템 도입 이후만) : 2019년 2월 15일 ~ 2019년 3월 24일
- train3 (기존 + 시스템 도입 이후) : 2019년 1월 2일 ~ 2019년 3월 24일
- test : 2019년 3월 25일 ~ 2019년 3월 31일

<img src="https://github.com/yesolee/LS_Die_casting/blob/main/img/models.jpg">

## 🔧분석 도구
- 불균형 데이터 처리
    - 랜덤 오버 샘플링 : 소수 클래스를 복제해 데이터 불균형을 완화
    - 랜덤 언더 샘플링 : 다수 클래스를 제거해 데이터 균형을 맞춤
    - SMOTE : 소수 클래스의 가상 데이터를 생성해 데이터 균형을 개선
    - 모델 자체 가중치 사용 : 클래스별 가중치를 조정해 모델 학습시 불균형을 보정

- 예측 모델
    - LightGBM : 속도와 대규모 데이터 처리에 강한 부스팅
    - Catboost : 범주형 데이터와 과적합 방지에 강한 부스팅

- 모델 해석
    - Feature Importance : 모델에 가장 중요한 피처를 정량적으로 보여주는 지표
    - SHAP : 개별 예측에 대한 각 피처의 기여도 설명
    - PDP : 특정 피처와 예측값의 관계를 전체적으로 파악


## 📚 기술 스택
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=SciPy&logoColor=white">
