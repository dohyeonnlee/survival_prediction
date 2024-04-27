# 🍽️서울시 음식점 평균 생존 기간을 활용한 생존 예측 모형
### 👀Fourcaster(미래를 내다본 자들)👀
###### 2022데이터 청년 캠퍼스 - 파이썬 기반 빅데이터 분석을 통한 비즈니스 인사이트 역량제고 과정
👧 강민경

🤴 김대건

👩 손채영

👩‍🦰 이도현

👩 최규진

🧑 최성호

###  프로젝트 개요🍴

서울시 음식점 생존에 영향을 미치는 다양한 외부 요인 분석을 통한 음식점 생존 예측 모형 연구

### 주제 선정 배경🤔

코로나 장기화 및 경기 악화로 인한 서울시 음식점 운영이 힘들어지며 잠재적으로 폐업을 고민중이거나 폐업 위기에 놓인 점포가 다수 존재할 것으로 예상

따라서, 서울시 내에 폐업률이 가장 높은 업종인 음식점업에 대한 생존 예측의 필요성을 느낌

### 분석 목적 및 목표⭐
종속변수를 음식점 평균 영업기간인 5년을 기준으로 생존/폐업으로 분류하여 서울시 음식점의 개별 생존여부 예측을 통한 인사이트 도출

▶️ 생존 혹은 폐업에 영향을 주는 변수를 통해 요인 분석 후 생존 예측을 통한 업종별 지원 정책 제안




### 분석 과정📝
1. 데이터 수집
2. 데이터 전처리(Excel, Python)
3. EDA
4. 모델 구현
5. 모델 성능 평가
6. 결론 시각화 및 결론 도출


### 데이터 출처🛒
- 서울열린데이터광장
- 문화빅데이터플랫폼
- 스마트치안빅데이터플랫폼


### 분석 도구 소개⛏️
- Pandas : 파이썬의 대표적인 데이터 분석 도구. 데이터 연산 기능 제공
- Numpy : 파이썬
- matplotlib : 자료를 차트나 플롯으로 시각화하는 도구
- seaborn : matplotlib를 바탕으로 만든 시각화 툴
- statsmodels : python에서 다양한 통계 분석을 할 수 있도록 기능을 제공
- scikit-learn : python에서 머신러닝을 쉽게 적용할 수 있도록 여러 기능을 제공


### 실행도구⚙️
- jupyter notebook
- colab

(기본 jupyter 기준 코드, colab용 코드 주석 처리)


### 활용 모델📊
- Logistic regression
- KNN
- RandomForest
- XGBoost


### 설치🔧
```
#코랩용 나눔폰트 설치, 실행후 런타임 다시 시작해야함
!sudo apt-get install -y fonts-nanum

!sudo fc-cache -fv

!rm ~/.cache/matplotlib -rf

#sklearn 1.1.0버전으로 재설치
#오버샘플링 시 필요한 imblearn을 위한 sklearn 버전

!pip uninstall sklearn 

!pip install --upgrade sklearn

!pip install scikit-learn==1.1.0 --user
  
#오버샘플링

!pip install imblearn

#xgboost

!pip install xgboost
```
  
### 코드 설명⌨️
#### .ipynb 파일 내 주석으로 자세한 설명 첨부


### 주요 활용 코드

#### 1. 로그 변환
```
#log변환 필요한 연속형 변수는 왜도가 -1~1 밖의 값으로 지정

#로그 변환
Alog["임대료"] = np.log1p(Alog["임대료"])

f, ax = plt.subplots(figsize = (10, 6))
#시각화
sns.distplot(Alog["임대료"])
#왜도 첨도 확인
print("Skewness: {:.3f}".format(Alog["임대료"].skew()))
print("Kurtosis: {:.3f}".format(Alog["임대료"].kurt()))
```
<로그변환 전>
![image](https://user-images.githubusercontent.com/101872963/187333347-a088d38a-e706-4c56-8b3c-623dec998e58.png)

<로그변환 후>
![image](https://user-images.githubusercontent.com/101872963/187333379-e84e430d-72ea-42c7-ac9c-44eff72bc137.png)


#### 2. 오버샘플링

```
from imblearn.over_sampling import SMOTE
method=SMOTE()
X_resampled,y_resampled=method.fit_resample(x_train,y_train) #오버샘플링
X_resampled.shape  #독립변수 구조 확인
y_resampled.value_counts() #균형다시맞춤
x_train=X_resampled #결과 다시 x_train변수에 담기
y_train=y_resampled#결과 다시 y_train변수에 담기
x_train.shape
y_train.value_counts() #균형다시맞춤
```

#### 3. 로지스틱 회귀분석
```
#모델생성
model = sm.Logit(y_train,x_train)
results = model.fit(method = "newton")
#결과확인
results.summary() # '골목상권', '관광특구', '교육 수', '구분불가', '발달상권', '버퍼내폐업비율','아파트 세대수',  '전통시장'제거
#오즈비
np.exp(results.params)
#예측
y_pred = results.predict(x_test)
y_pred

#스레스홀드 함수만들기
def PRED(y, threshold):
    Y=y.copy()
    Y[Y>threshold] = 1
    Y[Y <= threshold] = 0
    return(Y.astype(int))
#임계값 0.7예측
Y_pred = PRED(y_pred,0.7)
Y_pred

#성능확인
cfmat = confusion_matrix(y_test, Y_pred)
print(cfmat)

#정확도 함수 만들기
def acc(cfmat) :
    acc=(cfmat[0,0]+cfmat[1,1])/np.sum(cfmat)
    return(acc)

#정확도확인
acc(cfmat) 

#f1 확인
f1 = f1_score(y_test,Y_pred,average='weighted')
print(f"f1:{f1:4f}") 
```

#### 4. KNN
```
from sklearn.neighbors import KNeighborsClassifier

#적절한 k찾기
test_acc=[]
for n in range(1,10): #k=1~10까지
    clf=KNeighborsClassifier(n_neighbors=n)
    clf.fit(x_train2,y_train)
    y_pred=clf.predict(x_test2)
    test_acc.append(accuracy_score(y_test,y_pred))
    
    print("k: {}, 정확도: {}".format(n,accuracy_score(y_test,y_pred)))
    
    
#knn 학습, k=2 지정
clf=KNeighborsClassifier(2)
clf.fit(x_train2,y_train)

#예측
y_pred=clf.predict(x_test2)

#f1-score
f1 = f1_score(y_test, y_pred, average='macro') 
#accuracy, f1-score 출력
print(f"f1:{f1:4f} accuracy:{acc:.4f}")

```

#### 5. RandomForest
```
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = 100,#결정트리의 개수
                               max_depth = 15,#트리의 최대 깊이
                                n_jobs = -1,
                                 verbose = 1,
                               random_state = 101)
#랜덤포레스트 학습
rf_clf.fit(x_train2, y_train)
pred = rf_clf.predict(x_test2)

#정확도, f1-score 확인
accuracy = accuracy_score(y_test, pred) 
f1 = f1_score(y_test,pred,average='macro') 

print(f"f1:{f1:4f} accuracy:{accuracy:.4f}")

#GridCV
from sklearn.model_selection import GridSearchCV

#파라미터 
params = {'n_estimators' : [100, 200],#결정트리의 개수
          'max_depth' : [6, 8, 10, 12],#트리의 최대 깊이
         'min_samples_leaf' : [8,12,18],#lead node가 되기 위해 필요한 최소한의 샘플 데이터수
         'min_samples_split' : [8, 16, 20]}#노드를 분할하기 위한 최소한의 샘플 

rf_clf4 = RandomForestClassifier(random_state = 103,
                                 n_jobs = -1,
                                 verbose = 1)
grid_cv1 = GridSearchCV(rf_clf4,
                       param_grid = params,
                       n_jobs = -1,
                       verbose = 1,
                       cv=3)

grid_cv1.fit(x_train2, y_train)

grid_cv2 = GridSearchCV(rf_clf4,
                       param_grid = params,
                       n_jobs = -1,
                       verbose = 1,
                       cv=3,
                       scoring='f1')

grid_cv2.fit(x_train2, y_train)
#f1스코어 모델 기준 예측값
pred=grid_cv2.predict(x_test2)

print('최적 하이퍼 마라미터: ', grid_cv1.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv1.best_score_))
print('최적 하이퍼 마라미터: ', grid_cv2.best_params_)
print('최고 f1-score: {:.4f}'.format(grid_cv2.best_score_))
```

#### 6. XGBoost
```
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score

# 검증 데이터 넣어주어서 교차검증 해보도록하기
evals = [(x_test2, y_test)]
#파라미터지정 
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1,
                           max_depth=3)
# eval_metric넣어주면서 검증 데이터로 loss 측정할 때 사용할 metric 지정
xgb_wrapper.fit(x_train2, y_train, early_stopping_rounds=200,
               eval_set=evals, eval_metric='logloss')
#예측
preds = xgb_wrapper.predict(x_test2)
preds_proba = xgb_wrapper.predict_proba(x_test2)[:, 1]
print(preds_proba[:10])
#정확도, f1-score 확인
accuracy = accuracy_score(y_test,preds)
f1 = f1_score(y_test,preds,average='macro')

print(f"f1:{f1:4f} accuracy:{accuracy:.4f}") 



xgb_wrapper.predict(x_test2)

print("##########################################")

## GridSearchCV 이용해서 교차검증&최적의 파라미터 찾기
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth' : [3,5,8,10,15] ,#트리의 최대 깊이
    'n_estimators':[100,200],#생성할 weak learner의 수
    'learning_rate':[0.01,0.05, 0.1],#learning rate
    'gamma': [0.5,1,2,3], #leaf node의 추가분할을 결정할 최소손실 감소값
    'colsample_bytree' : [0.8,0.9],#각 tree별 사용된 feature의 퍼센티지
    'random_state':[99]
}


grid_cv = GridSearchCV(xgb_wrapper, param_grid=params,
                      n_jobs=-1, cv=3, verbose=1)
grid_cv.fit(x_train2, y_train)
grid_cv2 = GridSearchCV(xgb_wrapper, param_grid=params,
                      n_jobs=-1, cv=3, verbose=1, scoring='f1')
grid_cv2.fit(x_train2, y_train)

pred = grid_cv2.predict(x_test2) #f1스코어기준 모델의 예측값

print("최적의 파라미터:", grid_cv.best_params_)
print("최고의 정확도 :", grid_cv.best_score_)
print("최적의 파라미터:", grid_cv2.best_params_)
print("최고의 f1socre :", grid_cv2.best_score_)
```

