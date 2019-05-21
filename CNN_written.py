
# coding: utf-8

# In[1]:


#https://datascienceplus.com/handwritten-digit-recognition-with-cnn/
#https://medium.com/@himanshubeniwal/handwritten-digit-recognition-using-machine-learning-ad30562a9b64

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


## Import files "written"
written_train   = np.load("written_train.npy", allow_pickle=True)
#written_test = np.load(p + 'written_test.npy', allow_pickle=True)

# # Import files "spoken"
spoken_train = np.load('spoken_train.npy', allow_pickle=True)
#spoken_test = np.load('spoken_test.npy', allow_pickle=True)

# # Import file "match"
match_train = np.load('match_train.npy', allow_pickle=True)


# In[3]:


print(len(written_train))


# In[4]:


written_train_batch = written_train[:100]
match_train_batch = match_train[:100]


# In[5]:


X_train, X_val, y_train, y_val = train_test_split(written_train_batch, 
                                                  match_train_batch, 
                                                  test_size=1/3, 
                                                  random_state=555)


# In[6]:


print(len(X_train))
print(len(X_val))


# In[7]:


batch_size = 128
num_classes = 10
epochs = 12

img_rows, img_cols = 28, 28

X_train = X_train.reshape(66,28,28,1)
X_val = X_val.reshape(34,28,28,1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# In[8]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[9]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

