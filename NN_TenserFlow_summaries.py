
# coding: utf-8

# In[1]:


#https://www.tensorflow.org/alpha/tutorials/keras/feature_columns
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split

from tensorflow import layers


# In[2]:


## Import files "written"
written_train   = np.load("written_train.npy", allow_pickle=True)

# # Import files "spoken"
spoken_array = np.load('spoken_train.npy', allow_pickle=True)
spoken_train = []

for array in spoken_array:
    x = array.flatten()
    spoken_train.append(x)

# # Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)


# In[3]:


target = []
                       
for i, j in enumerate(match_train):
    if j == True:
        target.append(1)
    elif j == False:
        target.append(0)


# In[4]:


## MEANS
written_means = []
for i in written_train:
    written_means.append(np.mean(i))

spoken_means = []
for i in spoken_train:
    spoken_means.append(np.mean(i))
    
## LENGTH
spoken_length = []

for i in spoken_train:
    spoken_length.append(len(i))

## MINIMUM
written_min = []
for i in written_train:
    written_min.append(np.amin(i))

spoken_min = []
for i in spoken_train:
    spoken_min.append(np.amin(i))
    
## MAXIMUM
written_max = []
for i in written_train:
    written_max.append(np.amax(i))

spoken_max = []
for i in spoken_train:
    spoken_max.append(np.amax(i))
    
## GLOBAL MEAN
written_global = np.mean(written_train)
spoken_global = np.mean(spoken_means)

## STANDARD DEVIATION (PER MEAN OF ARRAY)
# written_sd = []

# for i in written_means:
#     x = np.sqrt(written_global* ((i-(written_global*i))**2))
#     written_sd.append(x)

# spoken_sd = []

# for i in spoken_means:
#     x = np.sqrt(spoken_global*((i-(spoken_global*i))**2))
#     spoken_sd.append(x)
    
# print(written_sd)
# print(spoken_sd)


# In[5]:


import pandas as pd
index = range(0,len(written_train))
columns = ['Mean written', 'Mean spoken', 'Length spoken', 'Minimum written', 'Minimum spoken', 'Maximum written', 'Maximum spoken', 'Target']

df_a = pd.DataFrame(index=index, columns=columns)
df_a['Mean written'] = written_means
df_a['Mean spoken'] = spoken_means
df_a['Length spoken'] = spoken_length
df_a['Minimum written'] = written_min
df_a['Minimum spoken'] = spoken_min
df_a['Maximum written'] = written_max
df_a['Maximum spoken'] =  spoken_max
df_a['Target'] = target


# In[6]:


train, val = train_test_split(df_a, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')


# In[7]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe
    #labels = dataframe.drop("Target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds


# In[8]:


batch_size = 500 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)


# In[11]:


feature_columns = []

# numeric cols
for header in columns:
    feature_columns.append(feature_column.numeric_column(header))


# In[14]:


feature_layer = tf.layers.Dense(feature_columns)


# In[17]:


#feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# In[18]:


batch_size = 5000
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)


# In[ ]:


# model = tf.Sequential([
#   feature_layer,
#   layers.Dense(128, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=5)


# In[20]:


model = tf.keras.models.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)


# In[ ]:


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

