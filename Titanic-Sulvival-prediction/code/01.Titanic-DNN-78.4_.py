#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 필수 라이브러리
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns   
get_ipython().run_line_magic('matplotlib', 'inline   #쥬피터노트북에서 이미지 표시가능하게 하는 쥬피터노트북 매직함수')


# 램덤 시드 고정
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# 데이터 확인
drive_path = '/content/drive/MyDrive/Kaggle_Study/titanic/'

train = pd.read_csv(drive_path + 'train.csv')
test =  pd.read_csv(drive_path + 'test.csv')

test.head(10)


# In[ ]:


tmp = train['Cabin'].str[:1]
tmp.value_counts()


# In[ ]:


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

def format_features_drop_cabin(df):
  df = df.drop(["Cabin","SibSp","Parch"], axis=1)
  features = ['Sex', 'Embarked', 'name_cat']
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


# In[ ]:


train = transform_features(train)
test = transform_features(test)


# In[ ]:


train.head()


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(data = train.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')


# In[ ]:


# train 데이터 X,y 나누기

y_train = train['Survived']
X_train_scaled = train.drop('Survived', axis=1)
X_train_scaled.head()


# In[ ]:


X_train_scaled.shape


# In[ ]:


# 심층 신경망 모델
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

adam = tf.keras.optimizers.Adam(
    learning_rate=0.001
)
def build_model(train_data, train_target):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=train_data.shape[1]))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = build_model(X_train_scaled, y_train)
model.summary()


# In[ ]:


# 콜백 함수 - Early Stopping 기법
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train
                                            , test_size=0.4
                                            , shuffle=True
                                            , random_state=SEED)
EPOCHS = 500
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_tr, y_tr
                    , batch_size=32
                    , epochs=EPOCHS
                    # , callbacks=[early_stopping]
                    , validation_data=(X_val, y_val),verbose=2)
                    
model.evaluate(X_val, y_val)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


X_test = test
X_test.head()


# In[ ]:


# test 데이터에 대한 예측값 정리
y_pred_proba = model.predict(X_test)
y_pred_proba

y_pred_label = np.around(y_pred_proba)

df = pd.DataFrame(y_pred_label.astype(int),columns=["Survived"])
df.head(30)


# In[ ]:


test_new_df = pd.read_csv(drive_path + '/test.csv')
test_new_df.head()


# In[ ]:


result = pd.concat([test_new_df['PassengerId'], df], axis=1)
result


# In[ ]:


result.to_csv('NN-basic-batchNormalization_earlystop_new.csv', index=False)


# In[ ]:




