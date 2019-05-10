
# coding: utf-8

# In[2]:


# TODO: create git repository
# TODO: create env with packages
import numpy as np


# In[12]:


# Import files "written" --> error in CWD

written_train = np.load('written_train.npy', allow_pickle=True)
written_test = np.load('written_test.npy', allow_pickle=True)

# Import files "spoken"
spoken_train = np.load('spoken_train.npy', allow_pickle=True)
spoken_test = np.load('spoken_test.npy', allow_pickle=True)

# Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)

print(written_train[0])

