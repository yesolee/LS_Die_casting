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
df2_dummie = pd.get_dummies(df2, columns = ['working', 'mold_code'], drop_first=True)

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
import optuna

# 성능 지표 저장

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

sampler_under = RandomUnderSampler(random_state=42)

# def objective(trial):
    
#     g_mean_scores = []
    
#     # LGBM
#     # n_estimators = trial.suggest_int('n_estimators', 50, 500)  # 트리의 개수를 더 넓게 탐색
#     # learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 0.1)  # 학습률 탐색 범위 좁힘
#     # max_depth = trial.suggest_int('max_depth', 3, 12)  # 깊이를 더 깊게 탐색
#     # lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 5.0)  # L1 규제 추가
#     # lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 5.0)  # L2 규제 탐색 범위 좁힘
#     # subsample = trial.suggest_float('subsample', 0.5, 1.0)  # 트리의 각 학습에 사용하는 샘플 비율
#     # colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)  # 각 트리에서 사용할 피처 비율

#     # model = LGBMClassifier(
#     #     n_estimators=n_estimators,
#     #     learning_rate=learning_rate,
#     #     max_depth=max_depth,
#     #     lambda_l1=lambda_l1,
#     #     lambda_l2=lambda_l2,
#     #     subsample=subsample,
#     #     colsample_bytree=colsample_bytree,
#     #     random_state=42
#     # )
    
#     # # catboost    
#     n_estimators = trial.suggest_int('n_estimators', 50, 300)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1)
#     depth = trial.suggest_int('depth', 1, 10)
#     l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 1, 10)

#     model = CatBoostClassifier(
#         iterations=n_estimators,
#         learning_rate=learning_rate,
#         depth=depth,
#         l2_leaf_reg=l2_leaf_reg,
#         silent=True,
#         random_state=42
#     )

#     # K-Fold 설정
#     kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#     # K-Fold 교차 검증
#     for train_index, valid_index in kf.split(X_train3, y_train3):
#         X_train_fold, X_valid_fold = X_train3.iloc[train_index], X_train3.iloc[valid_index]
#         y_train_fold, y_valid_fold = y_train3.iloc[train_index], y_train3.iloc[valid_index]
        
#         # 샘플링
#         X_train_fold_under, y_train_fold_under = sampler_under.fit_resample(X_train_fold, y_train_fold)

#         model.fit(X_train_fold_under, y_train_fold_under)

#         # 예측
#         y_pred_proba_fold = model.predict_proba(X_valid_fold)[:, 1]
#         y_pred_fold = (y_pred_proba_fold >= 0.5).astype(int)  # 확률을 0.5 임계값으로 이진화

#         # 성능 지표 계산
#         g_mean = geometric_mean_score( y_valid_fold, y_pred_fold)
#         g_mean_scores.append(g_mean)

#     return np.mean(g_mean_scores)

# # Study 생성 및 하이퍼파라미터 튜닝
# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
# study.optimize(objective, n_trials=100)

# # 1. 최적의 하이퍼파라미터 가져오기
# best_params = study.best_params
# print(f'Best parameters: {best_params}')
# print(f'Best geometric mean score: {study.best_value}')
# Best parameters: {'n_estimators': 222, 'learning_rate': 0.1620690533960441, 'depth': 6, 'l2_leaf_reg': 1}
# Best geometric mean score: 0.9796690438259769
best_params = {'n_estimators': 222, 'learning_rate': 0.1620690533960441, 'depth': 6, 'l2_leaf_reg': 1}
# 2. 최적의 하이퍼파라미터로 모델 생성 (LightGBM을 예시로 사용)
# LGBM
# best_model = LGBMClassifier(
#     n_estimators=best_params['n_estimators'],
#     learning_rate=best_params['learning_rate'],
#     max_depth=best_params['max_depth'],
#     lambda_l1=best_params.get('lambda_l1', 0.0),  # 'lambda_l1'이 없으면 기본값 0.0 사용
#     lambda_l2=best_params.get('lambda_l2', 0.0),  # 'lambda_l2'이 없으면 기본값 0.0 사용
#     subsample=best_params.get('subsample', 1.0),  # 'subsample'이 없으면 기본값 1.0 사용
#     colsample_bytree=best_params.get('colsample_bytree', 1.0),  # 'colsample_bytree'이 없으면 기본값 1.0 사용
#     random_state=42
# )

# CatBoostClassifier로 최적의 모델 생성
best_model = CatBoostClassifier(
    iterations=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params.get('l2_leaf_reg', 0.0),  # l2_regularization
    random_state=42,
    silent=True
)
# 3. 최적의 하이퍼파라미터로 train 데이터 학습
X_train3_under, y_train3_under = sampler_under.fit_resample(X_train3, y_train3)
best_model.fit(X_train3_under, y_train3_under)

# 4. test 데이터에 대해 예측 수행
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = best_model.predict(X_test)

precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_proba_test)
g_mean = geometric_mean_score(y_test, y_pred_test)

# 5. 결과 출력
print("Test predictions:", precision)
print("Test recall:", recall)
print("Test f1:", f1)
print("Test roc_auc:", roc_auc)
print("Test g_mean:", g_mean)

# Best parameters: {'n_estimators': 222, 'learning_rate': 0.1620690533960441, 'depth': 6, 'l2_leaf_reg': 1}
# Best geometric mean score: 0.9796690438259769
# Test predictions: 0.22535211267605634
# Test recall: 0.9411764705882353
# Test f1: 0.36363636363636365
# Test roc_auc: 0.9971639793512204
# Test g_mean: 0.9628892430321765

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
feature_importances = best_model.get_feature_importance()

feature_names = X_train3_under.columns  # 학습에 사용된 X_train1_under의 컬럼명 사용
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# 중요도 정렬
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))  # 크기 조정
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')

# y축의 글자를 회전시키고 크기 조정
plt.yticks(fontsize=10)  # 글자 크기 조정

plt.tight_layout()  # 레이아웃 최적화
plt.show()

# PDP 시각화
from sklearn.inspection import PartialDependenceDisplay
n_features = X_train3_under.shape[1]  # 학습에 사용된 데이터 사용
n_rows = 12  # 12행
n_cols = 5   # 5열
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 40))  # 12행 5열 서브플롯

# 각 변수에 대해 PDP 생성
for i in range(n_rows):
    for j in range(n_cols):
        feature_index = i * n_cols + j
        if feature_index < n_features:  # 변수 수 범위 체크
            PartialDependenceDisplay.from_estimator(
                best_model,
                X_train3_under,  # 학습에 사용된 데이터로 변경
                features=[feature_index],
                ax=ax[i, j]
            )
            ax[i, j].set_title(f'PDP for Feature {feature_index}')
        else:
            ax[i, j].axis('off')  # 변수가 없으면 빈 서브플롯 처리

plt.tight_layout()
plt.show()

###################### shap
import shap

# SHAP 값을 계산하기 위한 TreeExplainer 생성
explainer = shap.TreeExplainer(best_model)

# 학습 데이터에 대한 SHAP 값 계산
shap_values = explainer.shap_values(X_train3_under)

# 1. SHAP summary plot: 전체 변수 중요도와 분포를 시각화
shap.summary_plot(shap_values, X_train3_under)

# 2. SHAP dependence plot: 특정 변수의 SHAP 값을 시각화하고, 해당 변수와 예측 결과 간의 관계를 보여줌
shap.dependence_plot("lower_mold_temp2", shap_values, X_train3_under)

# 3. SHAP force plot: 개별 데이터 포인트에 대한 예측 기여도를 시각화
# 첫 번째 데이터 포인트에 대해 시각화 (인덱스 0)
# SHAP JavaScript 초기화
shap.initjs()

# 이후 force plot을 포함한 SHAP 시각화를 실행
shap.force_plot(explainer.expected_value, shap_values[0], X_train3_under.iloc[0])


################################

# PDP 결과를 데이터프레임으로 저장하는 함수
def save_pdp_to_dataframe(model, X, feature):
    # PDP 계산 (PartialDependenceDisplay 사용)
    pdp_result = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[X.columns.get_loc(feature)]
    )
    
    # x축 값 (feature 값)과 y축 값 (PDP 값) 추출
    # PDP 디스플레이 객체의 figure에서 데이터 추출
    for i, ax in enumerate(pdp_result.axes_.ravel()):  # PDP의 축을 순환하며 데이터 추출
        line = ax.lines[0]  # PDP 플롯에서 첫 번째 선을 가져옴
        x_values = line.get_xdata()  # X축 데이터
        y_values = line.get_ydata()  # Y축 데이터

    # x값과 y값을 데이터프레임으로 저장
    pdp_df = pd.DataFrame({
        f'{feature}_x': x_values,  # feature의 값
        f'{feature}_y': y_values   # PDP 결과 (PDP 값)
    })
    
    return pdp_df

# 예시: 'cast_pressure' 변수에 대한 PDP 계산
pdp_df = save_pdp_to_dataframe(best_model, X_train3_under, 'cast_pressure')

# PDP 데이터프레임 출력
pdp_df['cast_pressure_y'].unique()
pdp_df.groupby('cast_pressure_y',as_index=False).agg(n=('cast_pressure_x','count'))
pdp_df.query('cast_pressure_y > 0.908811')
pdp_df.loc[80:100]

sns.boxenplot(data=train3, x='passorfail', y='high_section_speed', hue='passorfail')
sns.violinplot(data=train3, x='passorfail', y='high_section_speed', hue='passorfail')