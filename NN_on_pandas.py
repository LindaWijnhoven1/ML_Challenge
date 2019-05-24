
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


X = df_a.iloc[:, 0:7].values
y = df_a.iloc[:, 7].values


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)


# In[9]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)


# In[10]:


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


# In[11]:


# Adding the input layer and the first hidden layer
#classifier.add(Flatten())
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 7))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[12]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[13]:


classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 100)


# In[18]:


loss, accuracy = classifier.evaluate(X_val, y_val)
print("Accuracy", accuracy)


# In[20]:


y_pred = classifier.predict(X_val)
y_pred = (y_pred > 0.5)
print(y_pred)


# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
print(cm)


# In[24]:


8107/893

