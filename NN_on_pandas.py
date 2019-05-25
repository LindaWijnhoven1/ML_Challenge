
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


## Import files "written"
written_train = np.load("written_train.npy", allow_pickle=True)

# # Import files "spoken"
spoken_array = np.load('spoken_train.npy', allow_pickle=True)
spoken_train = []

for array in spoken_array:
    x = array.flatten()
    spoken_train.append(x)

# # Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)


# In[3]:


written_corner = np.load('image_hoek.npy', allow_pickle=True)
plasma_sd = np.load('plasma_sd_Z.npy', allow_pickle=True)
plasma_m = np.load('plasma_mean_Z.npy', allow_pickle=True)


# In[ ]:


target = []
indexes_true = []
indexes_false = []
                       
for i, j in enumerate(match_train):
    if j == True:
        target.append(1)
        indexes_true.append(i)
    elif j == False:
        target.append(0)
        indexes_false.append(i)
        
false_included = indexes_false[:6000]
included_rows = indexes_true + false_included


# In[6]:


## MEANS
written_means = []
for i in written_train:
    written_means.append(np.mean(i))

spoken_means = []
for i in spoken_train:
    spoken_means.append(np.mean(i))
    
## COUNT OF DARK PIXELS
dark_pixels = []
for i in written_train:
    dark_pixels_count = 0
    for j in i:
        if j < 50:
            dark_pixels_count += 1
    dark_pixels.append(dark_pixels_count)
    
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


# In[ ]:


import pandas as pd
index = range(0,len(written_train))
#columns = ['Mean written', 'Mean spoken', 'Length spoken', 'Minimum written', 'Minimum spoken', 'Maximum written', 'Maximum spoken', 'Target']
#columns = ['Mean written', 'Mean spoken', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']
#columns = ['Mean written', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']
columns = ['Mean written', 'Corners written', 'Plasma sd', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']


df_a = pd.DataFrame(index=index, columns=columns)
df_a['Mean written'] = written_means
#df_a['Dark pixels written'] = dark_pixels
df_a['Corners written'] = written_corner
#df_a['Plasma mean'] = plasma_m
df_a['Plasma sd'] = plasma_sd
#df_a['Mean spoken'] = spoken_means
df_a['Length spoken'] = spoken_length
#df_a['Minimum written'] = written_min
df_a['Minimum spoken'] = spoken_min
#df_a['Maximum written'] = written_max
df_a['Maximum spoken'] =  spoken_max
df_a['Target'] = target


# In[26]:


df_a


# In[28]:


## Test on applying clustering for the written digits

# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# writtenK = KMeans(n_clusters=10)
# w = writtenK.fit(df_a)
# writtenL = writtenK.labels_
# print(writtenL[1001])


# In[ ]:


# Split data into X's (summaries) and y (which is the target value)

X = df_a.iloc[included_rows, 0:6].values
y = df_a.iloc[included_rows, 6].values


# In[ ]:


# Split dataset in training and validation

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)


# In[ ]:


# Standardize values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initalize model

classifier = Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
#classifier.add(Flatten())
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 6))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)


# In[ ]:


loss, accuracy = classifier.evaluate(X_val, y_val)
print("Accuracy", accuracy)


# In[ ]:


# Find count of true matches and the indexes that belong to this on validation data

count_true_val = 0
count_true_index = []

for i in range(0, len(y_val)):
    if y_val[i] == 1:
        count_true_val += 1
        count_true_index.append(i)
print(count_true_val)

count_false_val = 0
count_false_index = []

for i in range(0, len(y_val)):
    if y_val[i] == 0:
        count_false_val += 1
        count_false_index.append(i)
print(count_false_val)

count_true_val + count_false_val

print(count_true_index)


# In[ ]:


# Find count of true matches and the indexes that belong to this on predictions on validation data

y_pred = classifier.predict(X_val)

count_true = 0
count_true_i = []

for i in range(0, len(y_val)):
    if y_pred[i] > 0.5:
        count_true += 1
        count_true_i.append(i)
print(count_true)

count_false = 0
count_false_i = []
for i in range(0,len(y_pred)):
    if y_pred[i] < 0.5:
        count_false += 1
        count_false_i.append(i)
print(count_false)

print(count_true_i)

# match = []

# count_true + count_false
# for i in count_true_i:
#     for j in count_true_index:
#         if i == j:
#             match.append("X")
#         else:
#             match.append(" ")
# print(match)


# In[ ]:


## TN, FP
## FN, TP

y_predict = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_predict)
print(cm)


# In[ ]:


y_pred[0:30]


# In[ ]:


y_val[0:30]

