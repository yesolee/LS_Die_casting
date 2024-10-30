## 전처리 부분 주석 처리를 다르게 하며 성능을 평가함
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.metrics import geometric_mean_score

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 데이터 불러오기
df = pd.read_csv("data_week4.csv", encoding='cp949')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 시간 변수 추가 [넣기 / 빼기]
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['day_of_year'] = df['registration_time'].dt.dayofyear
df['is_special_time'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 19, 20] else 0)

# 불필요한 컬럼 제거 [무조건 빼기]
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# 불필요한 컬럼 제거2  [무조건 빼기]
df = df.drop(['upper_mold_temp3', 'lower_mold_temp3', 'count', 'heating_furnace'], axis=1) 

# 결측치 확인
df.isna().sum()

# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거 [무조건 하기]
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000)]

# 범주형 변수를 one-hot encoding
df2_dummie = pd.get_dummies(df2, columns = ['working',   'mold_code'], drop_first=True)

# 데이터 1, 2, 3 테스트데이터 나누기
train1 = df2_dummie[df2_dummie['registration_time']<'2019-02-15']
train2 = df2_dummie[(df2_dummie['registration_time']<'2019-03-25')&(df2_dummie['registration_time']>='2019-02-15')]
train3 = df2_dummie[df2_dummie['registration_time']<'2019-03-25']

test = df2_dummie[df2_dummie['registration_time']>='2019-03-25']

# X, y 나누기
X_train1 = train1.drop(['registration_time', 'passorfail'], axis=1)
y_train1 = train1['passorfail']

X_train2 = train2.drop(['registration_time', 'passorfail'], axis=1)
y_train2 = train2['passorfail']

X_train3 = train3.drop(['registration_time', 'passorfail'], axis=1)
y_train3 = train3['passorfail']

X_test = test.drop(['registration_time', 'passorfail'], axis=1)
y_test = test['passorfail']

# 결측치 확인
X_train1.columns[X_train1.isna().sum() > 0]
X_train2.columns[X_train2.isna().sum() > 0]
X_train3.columns[X_train3.isna().sum() > 0]
X_test.columns[X_test.isna().sum() > 0]

# 데이터1에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    X_train1[col] = X_train1[col].fillna(X_train1[col].mean())

# 데이터2에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_volume']:
    X_train2[col] = X_train2[col].fillna(X_train2[col].mean())
    
# 데이터3에 대한 수치형 결측치 mean 값으로 채우기
for col in ['molten_temp','molten_volume']:
    X_train3[col] = X_train3[col].fillna(X_train3[col].mean())

# 테스트 데이터에 대한  수치형 결측치 mean 값으로 채우기
for col in ['molten_volume']:
    X_test[col] = X_test[col].fillna(X_test[col].mean()) 
    
# ------------------------------------------- **_g 파생변수
X_train1['molten_temp_g'] = np.where(X_train1['molten_temp']<600, 1,0)  # boxplot
X_train2['molten_temp_g'] = np.where(X_train2['molten_temp']<600, 1,0)  # boxplot
X_train3['molten_temp_g'] = np.where(X_train3['molten_temp']<600, 1,0)  # boxplot
X_test['molten_temp_g'] = np.where(X_test['molten_temp']<600, 1,0)  # boxplot

X_train1['cast_pressure_g'] = np.where(X_train1['cast_pressure'] <= 295, 1, 0) # scatter
X_train2['cast_pressure_g'] = np.where(X_train2['cast_pressure'] <= 295, 1, 0) # scatter
X_train3['cast_pressure_g'] = np.where(X_train3['cast_pressure'] <= 295, 1, 0) # scatter
X_test['cast_pressure_g'] = np.where(X_test['cast_pressure'] <= 295, 1, 0) # scatter

X_train1['biscuit_thickness_g'] = np.where((X_train1['biscuit_thickness']>60) |(X_train1['biscuit_thickness'] <= 20), 1, 0)   # scatter
X_train2['biscuit_thickness_g'] = np.where((X_train2['biscuit_thickness']>60) |(X_train2['biscuit_thickness'] <= 20), 1, 0)   # scatter
X_train3['biscuit_thickness_g'] = np.where((X_train3['biscuit_thickness']>60) |(X_train3['biscuit_thickness'] <= 20), 1, 0)   # scatter
X_test['biscuit_thickness_g'] = np.where((X_test['biscuit_thickness']>60) |(X_test['biscuit_thickness'] <= 20), 1, 0)   # scatter

X_train1['physical_strength_g'] = np.where(X_train1['physical_strength'] < 600, 1, 0)  # scatter
X_train2['physical_strength_g'] = np.where(X_train2['physical_strength'] < 600, 1, 0)  # scatter
X_train3['physical_strength_g'] = np.where(X_train3['physical_strength'] < 600, 1, 0)  # scatter
X_test['physical_strength_g'] = np.where(X_test['physical_strength'] < 600, 1, 0)  # scatter

X_train1['low_section_speed_g'] = np.where((X_train1['low_section_speed'] < 50)|(X_train1['low_section_speed'] > 140), 1, 0)  # scatter
X_train2['low_section_speed_g'] = np.where((X_train2['low_section_speed'] < 50)|(X_train2['low_section_speed'] > 140), 1, 0)  # scatter
X_train3['low_section_speed_g'] = np.where((X_train3['low_section_speed'] < 50)|(X_train3['low_section_speed'] > 140), 1, 0)  # scatter
X_test['low_section_speed_g'] = np.where((X_test['low_section_speed'] < 50)|(X_test['low_section_speed'] > 140), 1, 0)  # scatter

X_train1['high_section_speed_g'] = np.where((X_train1['high_section_speed'] < 90)|(X_train1['high_section_speed'] > 205), 1, 0)  # scatter
X_train2['high_section_speed_g'] = np.where((X_train2['high_section_speed'] < 90)|(X_train2['high_section_speed'] > 205), 1, 0)  # scatter 
X_train3['high_section_speed_g'] = np.where((X_train3['high_section_speed'] < 90)|(X_train3['high_section_speed'] > 205), 1, 0)  # scatter 
X_test['high_section_speed_g'] = np.where((X_test['high_section_speed'] < 90)|(X_test['high_section_speed'] > 205), 1, 0)  # scatter   
    
    
# -------------------------------------------- 이상치 파생변수

# 이상치 여부 컬럼 만들기
def IQR_outlier(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df

num_X_columns = ['molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
       'low_section_speed', 'high_section_speed', 'molten_volume',
       'cast_pressure', 'biscuit_thickness', 'upper_mold_temp1',
       'upper_mold_temp2', 'lower_mold_temp1', 'lower_mold_temp2',
       'sleeve_temperature', 'physical_strength', 'Coolant_temperature',
       'EMS_operation_time']

# train1에 대한 이상치 파생변수
num_X_train1 = X_train1[num_X_columns]
for col in num_X_columns:
   X_train1[f'{col}_outlier'] = np.where((X_train1[col]<IQR_outlier(num_X_train1).loc['하한',col])|(X_train1[col]>IQR_outlier(num_X_train1).loc['상한',col]),True,False)

# train2에 대한 이상치 파생변수
num_X_train2 = X_train2[num_X_columns]
for col in num_X_columns:
   X_train2[f'{col}_outlier'] = np.where((X_train2[col]<IQR_outlier(num_X_train2).loc['하한',col])|(X_train2[col]>IQR_outlier(num_X_train2).loc['상한',col]),True,False)

# train3에 대한 이상치 파생변수
num_X_train3 = X_train3[num_X_columns]
for col in num_X_columns:
   X_train3[f'{col}_outlier'] = np.where((X_train3[col]<IQR_outlier(num_X_train3).loc['하한',col])|(X_train3[col]>IQR_outlier(num_X_train3).loc['상한',col]),True,False)

# test에 대한 이상치 파생변수
num_X_test = X_test[num_X_columns]
for col in num_X_columns:
   X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_test).loc['하한',col])|(X_test[col]>IQR_outlier(num_X_test).loc['상한',col]),True,False)

# ------------------------------------------------------------  데이터1, k-fold , 모델 => catboost, lightgbm 각각 진행 [둘중 뭐가 좋은지 선택]
# 데이터 1에 대한 k-fold, 3개 데이터로 모델 돌리기, 성능 지표 저장 
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 성능 지표 저장

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

sampler_over = RandomOverSampler(random_state=42)
sampler_under = RandomUnderSampler(random_state=42)
sampler_smote = SMOTE(random_state=42)

def result_modeling_kf(X, y, sampler, model_type):

    results= {
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'g_mean': []
    }

    for train_index, valid_index in kf.split(X):
        
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

        # 샘플링
        if sampler == 'over':
            X_train_resampled, y_train_resampled = sampler_over.fit_resample(X_train_fold, y_train_fold)
        elif sampler == 'under':
            X_train_resampled, y_train_resampled = sampler_under.fit_resample(X_train_fold, y_train_fold)
        elif sampler == 'smote':
            X_train_resampled, y_train_resampled = sampler_smote.fit_resample(X_train_fold, y_train_fold)
        elif sampler == 'None':
            X_train_resampled, y_train_resampled = X_train_fold.copy(), y_train_fold.copy()   
            
        # 모델 학습
        if model_type == 'lgbm':
            model = LGBMClassifier(random_state=42)
        elif model_type == 'lgbmweight':
            class_weights = {0: 1, 1: 9}
            model = LGBMClassifier(class_weight=class_weights, random_state=42)          
        elif model_type == 'cat':
            model = CatBoostClassifier(random_state=42, verbose=0)
        elif model_type == 'catweight':
            class_weights = [1, 9]  # 다수 클래스: 1, 소수 클래스: 9
            model = CatBoostClassifier(class_weights=class_weights, random_state=42, verbose=0)
        model.fit(X_train_resampled, y_train_resampled)

        # 예측
        y_pred_proba_fold = model.predict_proba(X_valid_fold)[:, 1]
        y_pred_fold = (y_pred_proba_fold >= 0.5).astype(int)  # 확률을 0.5 임계값으로 이진화

        # 각 성능 지표 계산
        precision = precision_score(y_valid_fold, y_pred_fold)
        recall = recall_score( y_valid_fold, y_pred_fold)
        f1 = f1_score( y_valid_fold, y_pred_fold)
        roc_auc = roc_auc_score( y_valid_fold, y_pred_proba_fold)
        g_mean = geometric_mean_score( y_valid_fold, y_pred_fold)
        
        # 결과 저장
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['roc_auc'].append(roc_auc)
        results['g_mean'].append(g_mean)  
        
    # 평균 성능 지표 출력
    for metric, values in results.items():
        print(f'Average {metric}: {np.mean(values)}')

################################ k-fold로 전처리/모델 비교
# 특정 전처리 주석 처리/활성화 하며 성능 비교
# 비교1: 원본 데이터 (필요없는 'Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date', 'upper_mold_temp3', 'lower_mold_temp3', 'count', 'heating_furnace', 'registration_time' 제거 + 월일시분초 만들지 않음)
# 비교2: 시간(월일시분초) 변수 있는거
# 비교3: **_g 파생변수
# 비교4: 이상치(IQR) 파생변수
# 비교5: 시간(월일시분초) 변수, dayoryear, 특정 시간 파생변수
# 비교6: 시간(월일시분초) 변수, dayoryear, 특정 시간 파생변수, **_g파생변수
# 비교7: **_g 파생변수,이상치 파생변수
# 비교8: 시간(월일시분초) 변수, dayoryear, 특정 시간 파생변수, 이상치 파생변수
# 비교9: 시간(월일시분초) 변수, dayoryear, 특정 시간 ,**_g 파생변수, 이상치(IQR) 파생변수

# 아래 코드를 비교1~9 전처리 방법을 변경해보면서 출력 (6*9=54개 모델 비교!!)
result_modeling_kf(X_train1, y_train1, 'None', 'cat')
result_modeling_kf(X_train1, y_train1, 'None', 'lgbm')
result_modeling_kf(X_train2, y_train2, 'None', 'cat')
result_modeling_kf(X_train2, y_train2, 'None', 'lgbm')
result_modeling_kf(X_train3, y_train3, 'None', 'cat')
result_modeling_kf(X_train3, y_train3, 'None', 'lgbm')

################################ k-fold로 샘플링 성능 비교 (5*3=15개 모델 비교!!)   
# train1 샘플링(None, under, over, smote, None+모델가중치) 별 비교 
# 전처리: 시간(월일시분초) 변수 있는거 + lgbm
# model: lgbm
result_modeling_kf(X_train1, y_train1, 'None', 'lgbm')
result_modeling_kf(X_train1, y_train1, 'under', 'lgbm')
result_modeling_kf(X_train1, y_train1, 'over', 'lgbm')
result_modeling_kf(X_train1, y_train1, 'smote', 'lgbm')
result_modeling_kf(X_train1, y_train1, 'None', 'lgbmweight')

# train2 샘플링(None, under, over, smote, None+모델가중치) 별 비교 
# 전처리: 시간(월일시분초) 변수, dayoryear, 특정 시간 ,**_g 파생변수, 이상치(IQR) 파생변수
# model: catboost
result_modeling_kf(X_train2, y_train2, 'None', 'cat')
result_modeling_kf(X_train2, y_train2, 'under', 'cat')
result_modeling_kf(X_train2, y_train2, 'over', 'cat')
result_modeling_kf(X_train2, y_train2, 'smote', 'cat')
result_modeling_kf(X_train2, y_train2, 'None', 'catweight')

# train3 샘플링(None, under, over, smote, None+모델가중치) 별 비교 
# 전처리: 시간(월일시분초) 변수, dayoryear, 특정 시간 ,**_g 파생변수, 이상치(IQR) 파생변수
# model: catboost
result_modeling_kf(X_train3, y_train3, 'None', 'cat')
result_modeling_kf(X_train3, y_train3, 'under', 'cat')
result_modeling_kf(X_train3, y_train3, 'over', 'cat')
result_modeling_kf(X_train3, y_train3, 'smote', 'cat')
result_modeling_kf(X_train3, y_train3, 'None', 'catweight')

################################ train 최고 모델별 test 성능 및 결과 시각화
def result_test(X_train, y_train, X_test, y_test, sampler, model_type):

    if sampler == 'over':
        X_train_resampled, y_train_resampled = sampler_over.fit_resample(X_train, y_train)
    elif sampler == 'under':
        X_train_resampled, y_train_resampled = sampler_under.fit_resample(X_train, y_train)
    elif sampler == 'smote':
        X_train_resampled, y_train_resampled = sampler_smote.fit_resample(X_train, y_train)
    elif sampler == 'None':
        X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()   

    if model_type == 'lgbm':
        model = LGBMClassifier(random_state=42)
    elif model_type == 'cat':
        model = CatBoostClassifier(random_state=42)   
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # 예측
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_proba_test >= 0.5).astype(int)  # 확률을 0.5 임계값으로 이진화

    # 각 성능 지표 계산
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    g_mean = geometric_mean_score(y_test, y_pred_test)
    
    # 성능 지표 출력
    print(f'Test precision: {precision}')
    print(f'Test recall: {recall}')
    print(f'Test f1: {f1}')
    print(f'roc_auc: {roc_auc}')
    print(f'g_mean: {g_mean}')
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred_test)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # 주요 변수 추출
    if model_type == 'lgbm':
        feature_importances = model.feature_importances_
    elif model_type == 'cat':
        feature_importances = model.get_feature_importance()
    
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # 중요도 정렬
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

    from sklearn.inspection import PartialDependenceDisplay
    # PDP 시각화
    n_features = X_train.shape[1]
    n_rows = 12  # 12행
    n_cols = 5   # 5열
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 40))  # 12행 5열 서브플롯

    # 각 변수에 대해 PDP 생성
    for i in range(n_rows):
        for j in range(n_cols):
            feature_index = i * n_cols + j
            if feature_index < n_features:  # 변수 수 범위 체크
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_train_resampled,
                    features=[feature_index],
                    ax=ax[i, j]
                )
                ax[i, j].set_title(f'PDP for Feature {feature_index}')
            else:
                ax[i, j].axis('off')  # 변수가 없으면 빈 서브플롯 처리

    plt.tight_layout()
    plt.show()                

# train1 전처리: 시간(월일시분초) 변수 있는거 + lgbm
result_test(X_train1, y_train1, X_test, y_test,'under', 'lgbm')

# train2,3 전처리: 시간(월일시분초) 변수, dayoryear, 특정 시간 ,**_g 파생변수, 이상치(IQR) 파생변수
result_test(X_train2, y_train2, X_test, y_test, 'under', 'cat')
result_test(X_train3, y_train3, X_test, y_test, 'under', 'cat')
