
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


match_train   = np.load("match_train.npy", allow_pickle=True)
spoken_test   = np.load("spoken_test.npy", allow_pickle=True)
spoken_train  = np.load("spoken_train.npy", allow_pickle=True)
written_test  = np.load("written_test.npy", allow_pickle=True)
written_train = np.load("written_train.npy", allow_pickle=True)


# In[5]:


print(np.shape(match_train))
print(np.shape(spoken_test))
print(np.shape(written_train))
print(np.shape(spoken_train))


# In[6]:


test = spoken_train[:1]
test1 = test[0]


print(np.std(test1[:20], axis = 1))


# In[7]:


test2 = spoken_train[1:2]


# In[8]:


print(np.mean(test[0][0]))
print(np.mean(test[0][1]))
print(np.mean(test[0][3]))
print(np.mean(test[0][4]))


# In[9]:


np.shape(written_train[0])


# In[10]:


print(np.shape(match_train))

waar = 0
nietwaar = 0
for i in match_train:
    if i == False:
        nietwaar +=1
    else:
        waar += 1
        
print("waar:",waar)
print("niet waar:", nietwaar)
print(np.sum(match_train))


# In[11]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
test1.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()


# In[12]:


28*28


# In[55]:


x = written_train[1:2]
y = x[0][:10]
y = y.astype("<U8")
print(y)
for i, value in enumerate(y):
    if value < "130":
        y[i] = "x"
    else:
        y[i] = ""
print(y)
    

#x = x.astype("int")
#np.savetxt("x.txt", x,fmt='%.1e')


# In[59]:



x = x.astype("<U8")

for i, value in enumerate(x[0]):
    if value < "130":
        x[0][i] = "x"
    else:
        x[0][i] = " "

#print(x.reshape(28,28))
 
#np.savetxt("x.txt", x)


# In[57]:



x = x.reshape(28,28)
np.savetxt("x.txt", x, fmt = "%s")


# In[60]:


print(np.mean(spoken_train))


# In[ ]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X_train = features_mean(train_signal)
X_valid = features_mean(valid_signal)
print("Passes\t Acc")
for passes in [5, 10, 20, 40]:
    model = Perceptron(random_state=666, n_iter=passes)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_valid, model.predict(X_valid))
    print("{}\t {}".format(passes, acc))


# In[ ]:


from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_val, y_train, y_val = train_test_split(iris.data[:,:3], 
                                                  iris.data[:,3], 
                                                  test_size=1/3, 
                                                  random_state=666)

print(np.shape(match_train))
print(np.shape(spoken_test))
print(np.shape(written_train))
print(np.shape(spoken_train))

