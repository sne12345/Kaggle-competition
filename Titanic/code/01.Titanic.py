#!/usr/bin/env python
# coding: utf-8

# # 0. 데이터 불러오기

# In[171]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[172]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[524]:


# titanic_df = pd.read_csv('/content/drive/MyDrive/Kaggle_Study/titanic/train.csv')
titanic_df = pd.read_csv('./train.csv')

titanic_df.head()


# In[525]:


titanic_df[40:60]


# In[526]:


print('\n ### 학습 데이터 정보 ### \n')
print(titanic_df.info())


# # 1. 결측치 개수 확인 & 처리

# In[527]:


print('데이터 세트 Null 값의 개수 \n', titanic_df.isnull().sum())


# In[528]:


def get_name_category(name):
  cat = ''
  if 'Mr.' in name: cat = 'Mr'
  elif 'Mrs.' in name: cat = 'Mrs'
  elif 'Miss.' in name: cat = 'Miss'
  elif 'Master.' in name: cat = 'Master'
  else : cat = 'else'

  return cat

plt.figure(figsize=(10,6))

group_names = ['Mr', 'Mrs', 'Miss', 'Master', 'else']

titanic_df['name_cat'] = titanic_df['Name'].apply(lambda x : get_name_category(x))
sns.barplot(x='name_cat', y='Age', data=titanic_df, order=group_names)
titanic_df


# In[529]:


titanic_df.groupby("name_cat")["Age"].mean()


# In[530]:


titanic_df["Age"].fillna(
    titanic_df.groupby("name_cat")["Age"].transform("mean"), inplace=True
)
titanic_df


# In[531]:


print('데이터 세트 Null 값의 개수 \n', titanic_df.isnull().sum())


# In[532]:


print('\n ### 학습 데이터 정보 ### \n')
print(titanic_df.info())


# In[533]:


titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
print('데이터 세트 Null 값의 개수 ', titanic_df.isnull().sum().sum())


# In[535]:


titanic_df['Cabin', 'Sex', 'Embarked','name_cat'].head()


# In[536]:


print('Sex 값 분포 :\n', titanic_df['Sex'].value_counts())
print('\n\n Cabin 값 분포 :\n', titanic_df['Cabin'].value_counts())
print('\n\n Embarked 값 분포 :\n', titanic_df['Embarked'].value_counts())


# # 2. 데이터 전처리
# ### 2.1 선실 등급에 따라 부자와 가난한 사람에 대한 판별을 하기 위해 선실 등급(Cabin 맨 앞자리)만 추출한다. 

# In[537]:


titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
titanic_df['Cabin'].head()


# ### 2.2 성별에 따른 생존여부 분석

# In[538]:


titanic_df.groupby(['Sex', 'Survived'])['Survived'].count()


# 여성 생존자수가 남성 생존자수보다 현저히 많은 것을 알 수 있다.

# In[539]:


sns.barplot(x='Sex', y='Survived', data=titanic_df)


# In[540]:


sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)


# In[541]:


def get_category(age):
  cat = ''
  if age <= -1: cat = 'Unknown'
  elif age <= 5: cat = 'Baby'
  elif age <= 12: cat = 'Child'
  elif age <= 18: cat = 'Teenager'
  elif age <= 25: cat = 'Student'
  elif age <= 35: cat = 'Young Adult'
  elif age <= 60: cat = 'Adult'
  else : cat = 'Elderly'

  return cat

plt.figure(figsize=(10,6))

group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
# titanic_df.drop('Age_cat', axis=1, inplace=True)


# In[543]:


titanic_df[['Cabin', 'Sex', 'Embarked','name_cat']].head()


# In[499]:


sns.barplot(x='Age_cat', y='SibSp', data=titanic_df)


# In[500]:


sns.barplot(x='Age_cat', y='Parch', data=titanic_df)


# In[510]:


from sklearn import preprocessing

def encode_features(dataDF):
  features = ['Cabin', 'Sex', 'Embarked','name_cat']
  for feature in features:
    le = preprocessing.LabelEncoder()
    le = le.fit(dataDF[feature])
    dataDF[feature] = le.transform(dataDF[feature])

  return dataDF

titanic_df = encode_features(titanic_df)
titanic_df[['Cabin', 'Sex', 'Embarked','name_cat']].head()


# ### 2.3 지금까지 전처리한 코드를 transform_features() 라는 하나의 함수에서 실행할 수 있도록 만들었습니다.

# In[459]:


def get_name_category(name):
  cat = ''
  if 'Mr.' in name: cat = 'Mr'
  elif 'Mrs.' in name: cat = 'Mrs'
  elif 'Miss.' in name: cat = 'Miss'
  elif 'Master.' in name: cat = 'Master'
  else : cat = 'else'

  return cat

def get_name(df):
    plt.figure(figsize=(10,6))

    group_names = ['Mr', 'Mrs', 'Miss', 'Master', 'else']
    df['name_cat'] = df['Name'].apply(lambda x : get_name_category(x))
    return df


def fillna(df):
    
  df["Age"].fillna(
      df.groupby("name_cat")["Age"].transform("mean"), inplace=True
  )
  
  df['Cabin'].fillna('N', inplace=True)
  df['Embarked'].fillna('N', inplace=True)
  return df

# 불필요한 속성 제거
def drop_features(df):
  df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
  return df

def format_features(df):
  df['Cabin'] = df['Cabin'].str[:1]
  features = ['Cabin', 'Sex', 'Embarked', 'name_cat']
  for feature in features:
    le = preprocessing.LabelEncoder()
    le = le.fit(df[feature])
    df[feature] = le.transform(df[feature])
  return df

def transform_features(df):
  df = get_name(df)
  df = fillna(df)
  df = drop_features(df)
  df = format_features(df)
  return df


# # 3. 머신러닝 모델 학습해보기

# In[460]:


titanic_df = pd.read_csv('./train.csv')
titanic_df = transform_features(titanic_df)

y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)

print(X_titanic_df.head())
print('전체 데이터의 개수 :',X_titanic_df.size)


# In[461]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2)


# In[462]:


print('train_test_split 한 후의 train data 개수 :',X_train.size)
print('train_test_split 한 후의 test data 개수 :',X_test.size)


# In[464]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ### 3.1 세 가지 모델 학습 / 예측 / 평가

# In[465]:


dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()

# DecisionTreeClassifier 학습 / 예측 / 평가
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('DecisionTreeClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier 학습 / 예측 / 평가
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도: {0: .4f}'.format(accuracy_score(y_test, rf_pred)))

# LogisticRegression 학습 / 예측 / 평가
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도: {0: .4f}'.format(accuracy_score(y_test, lr_pred)))


# ### 3.2 KFold 교차 검증

# In[466]:


from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
  kfold = KFold(n_splits=folds)
  scores = []

  for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
    X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
    y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    scores.append(accuracy)
    print('교차 검증 {0} 정확도: {1: .4f}'.format(iter_count, accuracy))
  
  mean_score = np.mean(scores)
  print('평균 정확도:{0: .4f}'.format(mean_score))

print('DecisionTreeClassifier\n')
print(exec_kfold(dt_clf, folds=5))

print('\nRandomForestClassifier\n')
print(exec_kfold(rf_clf, folds=5))

print('\nLogisticRegression\n')
print(exec_kfold(lr_clf, folds=5))


# ### 3.3 cross validation 교차 검증

# In[467]:


from sklearn.model_selection import cross_val_score

print('DecisionTreeClassifier')
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
  print('교차 검증 {0} 정확도: {1: .4f}'.format(iter_count, accuracy))
print('평균 정확도: {0:.4f}'.format(np.mean(scores)))


print('\nRandomForestClassifier')
scores = cross_val_score(rf_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
  print('교차 검증 {0} 정확도: {1: .4f}'.format(iter_count, accuracy))
print('평균 정확도: {0:.4f}'.format(np.mean(scores)))

print('\nLogisticRegression')
scores = cross_val_score(lr_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
  print('교차 검증 {0} 정확도: {1: .4f}'.format(iter_count, accuracy))
print('평균 정확도: {0:.4f}'.format(np.mean(scores)))


# ### 3.4 GridSearchCV
# 
# 

# In[468]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10], 'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=20)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터 : ', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 : {0: .4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0: .4f}'.format(accuracy))


# In[469]:


grid_dclf = GridSearchCV(rf_clf, param_grid=parameters, scoring='accuracy', cv=20)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터 : ', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 : {0: .4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 RandomForestClassifier 정확도 : {0: .4f}'.format(accuracy))


# # 4. test 데이터 넣어서 test 하기

# In[470]:


test_df = pd.read_csv('./test.csv')
test_df.head()


# In[471]:


transform_features(test_df)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
test_df.isnull().sum()


# In[475]:


grid_dclf = GridSearchCV(rf_clf, param_grid=parameters, scoring='accuracy', cv=20)
grid_dclf.fit(X_titanic_df, y_titanic_df)

print('GridSearchCV 최적 하이퍼 파라미터 : ', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 : {0: .4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpredictions = best_dclf.predict(test_df)


# In[476]:


dpredictions.size


# In[477]:


df = pd.DataFrame(dpredictions)
df.rename(columns={0:'Survived'}, inplace=True)
df


# In[478]:


test_new_df = pd.read_csv('./test.csv')
test_new_df.head()


# In[479]:


result = pd.concat([test_new_df['PassengerId'], df], axis=1)
result


# In[480]:


result.to_csv('answer_rf_total_transform_cvhigher.csv', index=False)


# In[80]:




