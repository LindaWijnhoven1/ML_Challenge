#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from fastdtw import fastdtw
import scipy.spatial.distance
from scipy.spatial import distance
from scipy.spatial.distance import euclidean


# In[2]:


## Import files "written"
written_train   = np.load("written_train.npy", allow_pickle=True)
#written_test = np.load(p + 'written_test.npy', allow_pickle=True)

# # Import files "spoken"
spoken_train = np.load('spoken_train.npy', allow_pickle=True)
#spoken_test = np.load('spoken_test.npy', allow_pickle=True)

# # Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)


# In[ ]:


sample_w = written_train[:200]
sample_s = spoken_train[:200]
sample_m = match_train[:200]

val_w = written_train[200:400]
val_s = spoken_train[200:400]
val_m = match_train[200:400]


# In[ ]:


print(match_train[200])


# In[ ]:


# Put all indexes for which true or false into array
indexes_true = []
indexes_false = []
                       
for i, j in enumerate(sample_m):
    if j == True:
        indexes_true.append(i)
    elif j == False:
        indexes_false.append(i)

# Put all values on the indexes for which true or false in an array
true_written = sample_w[indexes_true]
false_written = sample_w[indexes_false]

true_spoken = sample_s[indexes_true]
false_spoken = sample_s[indexes_false]


# In[ ]:


for ind, val in enumerate(val_m):
    if val == True:
        print(ind)


# In[ ]:


#euclideans_spoken = []

for ind, i in enumerate(val_w):
    match_score_w = 0
    index_max = 0
    match_score_s = 0.0
    for k, j in enumerate(true_written):
        x = 1 - scipy.spatial.distance.cosine(i, j)
        if x > match_score_w:
            match_score_w = x
            index_max = k    
    
    match_score_s, path = fastdtw(val_s[ind], true_spoken[index_max], dist=euclidean)
    #euclideans_spoken.append(match_score_s)
    print("Values for written from sample set at index", ind)
    print(match_score_w)
    print(index_max)
    print("Values for spoken at index,", ind)
    print(match_score_s)
    
    yes_no = "No match"
    
    if match_score_s < 50:
        yes_no = "Match"
    
    print(yes_no)
    print(" ")
        


# In[46]:


import matplotlib.pyplot as plt
img = written_train[1004]
#for index, value in enumerate(img):
#    if value < 80:
#        img[index] = 1
#    else:
#        img[index] = 0
#
img = img.reshape(28,28)

plt.imshow(img, cmap='Set3')


# In[ ]:


import matplotlib.pyplot as plt
img = sample_w[0]
#for index, value in enumerate(img):
#    if value < 80:
#        img[index] = 1
#    else:
#        img[index] = 0
#
img = img.reshape(28,28)

plt.imshow(img, cmap='Set3')


# In[ ]:




