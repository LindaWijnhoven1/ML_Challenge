#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


# In[47]:


## Import files "written"
written_train = np.load("written_train.npy", allow_pickle=True)

# # Import files "spoken"
spoken_train = np.load('spoken_train.npy', allow_pickle=True)

# # Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)

written_corner = np.load('image_hoek.npy', allow_pickle=True)
plasma_sd = np.load('plasma_sd_Z.npy', allow_pickle=True)
plasma_m = np.load('plasma_mean_Z.npy', allow_pickle=True)


# In[48]:


target = []
indexes_true = []
indexes_false = []
                       
for i, j in enumerate(match_train):
    if j == True:
        target.append(2)
        indexes_true.append(i)
    elif j == False:
        target.append(1)
        indexes_false.append(i)
        
false_included = indexes_false[:6000]
included_rows = indexes_true + false_included


# In[49]:


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
    
## LENGTH 2 standardize
spoken_length2 = []

for i in spoken_array:
    spoken_length2.append(len(i))
    
spoken_length2 = stats.zscore(spoken_length2)
    
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


# In[78]:


import pandas as pd
index = range(0,len(written_train))
#columns = ['Mean written', 'Mean spoken', 'Length spoken', 'Minimum written', 'Minimum spoken', 'Maximum written', 'Maximum spoken', 'Target']
#columns = ['Mean written', 'Mean spoken', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']
#columns = ['Mean written', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']
columns = ['Mean written', 'Corners written','Plasma sd', 'Length spoken', 'Minimum spoken', 'Maximum spoken', 'Target']


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


df_a


# In[79]:


# Split data into X's (summaries) and y (which is the target value)

X = df_a.iloc[included_rows, 0:6].values
y = df_a.iloc[included_rows, 6].values


# In[80]:



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)


# In[81]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(X_train, y_train);


# In[82]:


rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'entropy', max_depth = None, 
                               min_samples_split = 2, min_samples_leaf = 1)


# In[83]:


# Use the forest's predict method on the test data
predictions = rf.predict(X_val)

# Calculate the absolute errors
errors = abs(predictions - y_val)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[84]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_val)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[86]:


count_true = 0
count_false = 0
for i in y_val:
    if i == 1:
        count_false += 1
    else:
        count_true += 1


# In[ ]:





# In[87]:


print(count_false)


# In[89]:


count_true/(count_true+count_false)


# In[90]:


index = range(0,len(y_val))
columns = ['predictions', 'y_val']


df_b = pd.DataFrame(index=index, columns=columns)
df_b['predictions'] = predictions
df_b['y_val'] = y_val


df_b


# In[ ]:




