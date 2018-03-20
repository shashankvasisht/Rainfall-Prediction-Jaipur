
# coding: utf-8

# In[2]:

import numpy as np
import csv
import sklearn
import sklearn.linear_model
import math


# In[3]:

import pandas as pd


# In[4]:

data = pd.read_csv("abhay.csv")


# In[5]:

df = pd.DataFrame(data ,index=range(0, len(data) ,1))
df


# In[6]:

msk = np.random.rand(len(df)) < 0.8
#train = df[msk]
limit = int(math.ceil(0.8 * len(data)))
test = df[limit:]


train = df[1 : limit]


# In[7]:

train


# In[8]:

len(train)


# In[9]:

len(df)


# In[10]:

len(test)


# In[11]:

test.head()


# In[12]:

test_y = test[["Precipitation"]]
test_x = test[['Mean Temp', 'max Temp', 'Min temp' , 'Dew Point' , 'Average Humidity' , 'Max Humidity' , 'Minimum Humidity' 
                    , 'Sea Level Pressure' ,  'Average Wind Speed' , 'Maximum Wind Speed']]
print test_x
#print test_y


# In[13]:

train_data_y = train["Precipitation"]
train_data_x = train[['Mean Temp', 'max Temp', 'Min temp' , 'Dew Point' , 'Average Humidity' , 'Max Humidity' , 'Minimum Humidity' 
                    , 'Sea Level Pressure' ,  'Average Wind Speed' , 'Maximum Wind Speed']]
#print train_data_y


# In[14]:

#print train_data_x


# In[15]:

example_model = sklearn.linear_model.LinearRegression()


# In[16]:

print len(train_data_x)
print len(train_data_y)
print len(test_x)
print len(test_y)
print type(len(test_x))


# In[17]:

example_model.fit(train_data_x , train_data_y )



# In[32]:

a = 0
for i in range(7):
    x_new = test_x[i : i + 1]
    pred =  example_model.predict(x_new)
    if (pred > 0):
        print pred 
    else:
        print '0.000000'
    a = a + pred
print 'weekly rainfall:' , a
        


# In[ ]:

# x_new = test_x[56:57] #GDP PER CAP
# print (example_model.predict(x_new))
# print test_y[56:57]

# print test_y.shape
# print test_x.shape
# xa = test_x[0]
# print xa
# for i in range(10):
#     a[i] = i
#     print i
# # i = 0
print test_y.shape
print x_new.shape
print test_x.shape
print '-----'
a = test_x.loc[[limit]].shape
print  a
print '---'
for i in range(limit , len(data)) :
    x_new[limit - i ] = test_x.loc[i]
    #a[i] = (example_model.predict(x_new))
    #print x_new
#print a
print test_y.shape
print x_new.shape
print test_x.shape
#print len(a)


# In[167]:



for i in range(949):
    x_new = test_x[i : i+1]
    a =  example_model.predict(x_new)
    b = np.reshape(a , (1 , 949))
    b[i : i + 1] = example_model.predict(x_new)
#     print a
    print b
                   
print x_new.shape
print test_x.shape
print a.shape
print b.shape


# In[166]:

print b.shape
print test_y .shape
#sklearn.metrics.mean_absolute_error(test_y, b, sample_weight=None, multioutput= 'uniform_average')


# In[234]:

#sklearn.metrics.accuracy_score(test_y , a)


# In[248]:

conc = np.c_[test_y, a]
print conc


# In[235]:

a = 100
b = np.random.rand(2,2)
print a


# In[19]:

from firebase import firebase
from firebase.firebase import FirebaseApplication
import json


# In[237]:

firebase = firebase.FirebaseApplication('https://abhayfirebase.firebaseio.com/')
# result = firebase.get('/user' , None) #none = one
# print result 


# In[238]:

my_json_string = json.dumps({'key1': a})


# In[239]:

print my_json_string


# In[244]:

list1 = [1, 2, 3]
data = {'variable' : a}
sent = json.dumps(data)
result = firebase.post("/user/" + str(a), {'date_of_birth': list1, 'full_name':['1' , '2']}) #sent
print result


# In[218]:

result1 = firebase.post('/users', {'hi' : 'hi'})
print result1


# In[224]:

list1 = [1, 2, 3]
str1 = ''.join(str(e) for e in list1)


# In[225]:

print str1


# In[247]:

result1 = firebase.post('/users', {'hi' : list1})
print result1


# In[33]:

import pickle


# In[34]:

filename = 'rainfall_final.pkl'


# In[35]:

pickle.dump(example_model, open(filename, 'wb'))


# In[36]:

ls


# In[ ]:



