
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


match_train   = np.load("match_train.npy", allow_pickle=True)
spoken_train  = np.load("spoken_train.npy", allow_pickle=True)
written_train = np.load("written_train.npy", allow_pickle=True)


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(written_train[nieuw.argsort()[100:200][30]:nieuw.argsort()[100:200][30]+1,:].reshape(28,28),cmap = plt.cm.binary)
plt.show()


# In[3]:


nieuw = np.empty(45000)
for index, value in enumerate(spoken_train):
    nieuw[index] = value.shape[0]

print("kortste sound:",nieuw[np.argmin(nieuw)])
print("langste sound:",nieuw[np.argmax(nieuw)])

np.argmin(nieuw)
print("tien kleinste sounds:", np.sort(nieuw)[:10])
print(match_train[np.argmin(nieuw)])

print(spoken_train[np.argmin(nieuw)])

print(nieuw[nieuw.argsort()[100:200]])
print(match_train[nieuw.argsort()[100:200]])

print(nieuw.argsort()[100:200][0])


# In[4]:


import tensorflow as tf


# In[5]:


probeersel = written_train[:1000,:]
#probeersel = probeersel.reshape(1000,28,28)
probeersel.shape


# In[6]:


probeersel = tf.keras.utils.normalize(probeersel)


# In[10]:


nieuw_2 = tf.keras.utils.normalize(nieuw, axis = 0).T
nieuw_3 = nieuw_2[:1000]
probeersel[0].shape
#np.hstack([probeersel[i],nieuw[i]])


# In[11]:


nieuw_4 = np.empty(1000*785).reshape(1000,785)
for i in range(0,probeersel.shape[0]):
    buffer = np.hstack([probeersel[i],nieuw_3[i]])
    nieuw_4[i] = buffer



# In[12]:


from sklearn.model_selection import train_test_split

#training en validatie set maken
X_train, X_val, y_train, y_val = train_test_split(nieuw_4, 
                                                  match_train[:1000], 
                                                  test_size=1/3, 
                                                  random_state=555)


# In[13]:


X_train.shape


# In[14]:



model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten(input_shape=(785,)))  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)



