#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import h5py
 
import PIL
import matplotlib.image as img
import matplotlib.pyplot as plt
import io, os

from sklearn.model_selection import train_test_split


# In[2]:


import pathlib
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input,  MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# ## 3. train data 불러오기

# In[4]:


save_path = '/content/drive/MyDrive/Kaggle_Study/나은/new-26.h5'

h5f = h5py.File(save_path, 'r')
data1 = h5f.get('images')[()]
print(data1[:10])
h5f.close()


# In[56]:


np.shape(data1)


# In[5]:


plt.imshow(data1[0])
plt.show()


# In[ ]:





# ## 3-2. train label 만들기

# In[6]:


name_labels = pd.read_csv("/content/drive/MyDrive/Kaggle_Study/나은/names_labels.csv", header=None,index_col=0)
name_labels = name_labels.transpose()


# In[7]:


name_labels


# In[8]:


name_labels = name_labels.values.tolist()[0]


# In[9]:


name_labels = name_labels[1:]
len(name_labels)


# In[10]:


names_list = name_labels


# In[60]:


save_path = '/content/drive/MyDrive/Kaggle_Study/나은/new-26.h5'

with h5py.File(save_path, 'r') as hdf:
    base_items = list(hdf.items())

names_train = []

for item in base_items:
    for i in range(48):
        names_train.append(item[0])
    
names_list = list(map(int, names_train))
names_list[:500]


# In[11]:


np_names_train = np.array(names_list)

np_train_labels = to_categorical(np_names_train, 1000)
np_train_labels


# In[12]:


np.shape(np_train_labels)


# ## 4. train_valid_split

# In[13]:


# Train : 60% Valid: 40% 나누기
train_images, valid_images, train_labels, valid_labels = train_test_split(data1, np_train_labels, test_size=0.2)


# In[14]:


np.shape(valid_images)


# In[15]:


np.shape(valid_labels)


# In[18]:


plt.imshow(data1[11])
plt.show()


# ## 5. data generator

# In[19]:


datagen_kwargs = dict(rescale=1./255)
dataflow_kwargs = dict()

# Train
train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        rotation_range = 20,
        # brightness_range=[0.6,1.0],
    **datagen_kwargs
)

train_generator = train_datagen.flow(
    train_images, 
    y=train_labels,
    **dataflow_kwargs
)

# Validation
valid_datagen = ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow(
    valid_images,
    y=valid_labels,
    shuffle=False, 
    **dataflow_kwargs
)

# # Test
# test_datagen = ImageDataGenerator(**datagen_kwargs)
# test_generator = test_datagen.flow(
#     test_images, 
#     y=names_test, 
#     shuffle=False, 
#     **dataflow_kwargs
# )


# ## 6. MLP 모델

# In[20]:


image_height = 64
image_width = 64
image_channel = 3


# In[21]:


def MLP(img_size):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(img_size, img_size, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512,activation='relu'))
    model.add(tf.keras.layers.Dense(units=256,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1000,activation='softmax'))
    
    return model

model = MLP(64)

model.summary()


# In[22]:


model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# In[23]:


# 처음 만든 모델이라면 EPOCHS를 1~5개로 하여 잘 돌아가는지 
# 성능을 확인해보고 값을 증가 시켜 봅시다. 
EPOCHS = 100

# EPOCHS에 따른 성능을 보기 위하여 history 사용
history = model.fit(
    train_images,
    train_labels, 
    validation_data = (valid_images, valid_labels),
    epochs=EPOCHS
)


# In[ ]:




