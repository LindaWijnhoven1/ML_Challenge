
# coding: utf-8

# In[6]:


import numpy as np


# In[7]:


match_train   = np.load("match_train.npy", allow_pickle=True)
spoken_test   = np.load("spoken_test.npy", allow_pickle=True)
spoken_train  = np.load("spoken_train.npy", allow_pickle=True)
written_test  = np.load("written_test.npy", allow_pickle=True)
written_train = np.load("written_train.npy", allow_pickle=True)


# In[8]:



print("written_train",np.shape(written_train))
print("spoken_train",np.shape(spoken_train))
print("match_train",np.shape(match_train))
print("spoken_test",np.shape(spoken_test))


# In[80]:


## sample van trainingdata om klein te beginnen, staat nu op de volledige trainingsset
## maar kan voor data exploratie hier aangepast worden. 
sample_w = written_train[:,:]
sample_s = spoken_train[:]


# In[81]:


## eerste 100 records heeft 9 TRUE's (matches)
print(sum(match_train[:100]))


# In[96]:


## mean berekenen van Written
mean_w = np.mean(sample_w, axis = 1)

## mean van spoken wordt berekent in een for loop hieronder, omdat elke sample varieert in N * 13 features
# eerst een lege array aanmaken
mean_s = np.empty(len(sample_s))

#for loop om gemiddelde uit te rekenen
for i in range(0, len(sample_s)):
    mean_s[i] = np.mean(sample_s[i])

    
# means in 1 array zetten en vervolgens transponeren om in het model te passen
test = np.array([mean_s,mean_w])
test = test.T


# In[109]:


from sklearn.model_selection import train_test_split

#training en validatie set maken
X_train, X_val, y_train, y_val = train_test_split(test, 
                                                  match_train, 
                                                  test_size=1/3, 
                                                  random_state=555)


# In[112]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

#Perceptron model

X_train = X_train
X_valid = X_val
print("baseline", (1- (np.sum(match_train)/match_train.size)))
print()
print("perceptron model:\n")
print("Passes\t Acc")
for passes in [5, 10, 20, 40]:
    model = Perceptron(random_state=666, n_iter=passes)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_valid))
    print("{}\t {}".format(passes, acc))

