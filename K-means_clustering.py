
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


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


written_train_sample = written_train[:6000]
spoken_train_sample = spoken_train[:6000]

## Add zeros to the end of the array if it is not the longest length
x = len(max(spoken_train_sample, key=len))

spoken_train_zeros = []

for i in spoken_train_sample:
    if len(i) < x:
        j = (x - len(i)) * [0]
        z = np.hstack((i, j))
        #z = i + j
        spoken_train_zeros.append(z)
    else:
        spoken_train_zeros.append(i)


# In[4]:


## Cluster datasets into 10 clusters (digit 0 to 9 = 10 different clusters)
writtenK = KMeans(n_clusters=10)
spokenK = KMeans(n_clusters=10)
w = writtenK.fit(written_train_sample)
s = spokenK.fit(spoken_train_zeros)


# In[14]:


writtenL = writtenK.labels_
print("Clusters written")
print(writtenL[12])
print(writtenL[51])
print(writtenL[57])
print(writtenL[64])
spokenL = spokenK.labels_
print("Clusters spoken")
print(spokenL[12])
print(spokenL[51])
print(spokenL[57])
print(spokenL[64])


# In[ ]:


## Function to predict new classes

Kmean.predict()


# In[16]:


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
        
# false_included = indexes_false[:6000]
# included_rows = indexes_true + false_included
print(indexes_true)

