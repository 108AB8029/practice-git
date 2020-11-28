#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
sns.set()
from keras.models import Sequential
from keras.layers import Dense, Dropout


# # 透過pd.read_csv導入資料

# In[2]:


df_train = pd.read_csv(r'C:\Users\Lab433\Desktop\ChuJun\6Course\機器學習\Regression\HW\ntut-ml-2020-regression\ntut-ml-regression-2020\train-v3.csv')
df_test = pd.read_csv(r'C:\Users\Lab433\Desktop\ChuJun\6Course\機器學習\Regression\HW\ntut-ml-2020-regression\ntut-ml-regression-2020\test-v3.csv')
df_valid =  pd.read_csv(r'C:\Users\Lab433\Desktop\ChuJun\6Course\機器學習\Regression\HW\ntut-ml-2020-regression\ntut-ml-regression-2020\valid-v3.csv')

df_train


# # 將train與validation的資料合併，進行做模型訓練

# In[3]:


df_trainCombinevalid=pd.DataFrame()
df_trainCombinevalid=df_train.append(df_valid)
df_trainCombinevalid.reset_index(drop=True,inplace=True)


# In[4]:


df_trainCombinevalid


# # 資料視覺化
# 長條圖 Bar Chart
# plt.bar(position_data.POSITION.unique(),
#         position_data.POSITION.value_counts(), 
#         width=0.5, 
#         bottom=None, 
#         align='center', 
#         color=['lightsteelblue', 
#                'cornflowerblue', 
#                'royalblue', 
#                'midnightblue', 
#                'navy', 
#                'darkblue', 
#                'mediumblue'])
# plt.xticks(rotation='vertical')

# In[5]:


x = df_trainCombinevalid["sqft_living"]
y = df_trainCombinevalid["price"]
plt.xlabel("sqft_living")
plt.ylabel("price")
plt.scatter(x,y)
plt.show


# In[6]:


x = df_trainCombinevalid["sqft_lot"]
y = df_trainCombinevalid["price"]
plt.xlabel("sqft_lot")
plt.ylabel("price")
plt.scatter(x,y)
plt.show


# In[7]:


x = df_trainCombinevalid["sqft_living15"]
y = df_trainCombinevalid["price"]
plt.xlabel("sqft_living15")
plt.ylabel("price")
plt.scatter(x,y)
plt.show


# In[8]:


x = df_trainCombinevalid["sqft_lot15"]
y = df_trainCombinevalid["price"]
plt.xlabel("sqft_lot15")
plt.ylabel("price")
plt.scatter(x,y)
plt.show


# In[9]:


x = df_trainCombinevalid["sqft_above"]
y = df_trainCombinevalid["price"]
plt.xlabel("sqft_above")
plt.ylabel("price")
plt.scatter(x,y)
plt.show


# # 資料刪除

# In[10]:


#刪除bedrooms的離群值
mask = df_trainCombinevalid[df_trainCombinevalid["bedrooms"]>10]
mask.head(5)
df_trainCombinevalid=df_trainCombinevalid.drop(index=[6799,12012])
df_trainCombinevalid


# In[11]:


#刪除sqft_lot的離群值
mask = df_trainCombinevalid[df_trainCombinevalid["sqft_lot"]>0.8*10**6]
mask.head(5)
df_trainCombinevalid=df_trainCombinevalid.drop(index=(7688))
df_trainCombinevalid


# In[12]:


# 刪除sqft_living15的離群值

mask1 = df_trainCombinevalid["price"]>6*10**6
mask = df_trainCombinevalid[(mask1)]

mask.head(10)

df_trainCombinevalid.drop(index=(df_trainCombinevalid.loc[(df_trainCombinevalid["price"]>6*10**6)].index),inplace=True)
df_trainCombinevalid


# In[13]:


#刪除sqft_lot15的離群值

# mask1 = df_trainCombinevalid["sqft_lot15"]>800000
df_trainCombinevalid.drop(index=(df_trainCombinevalid.loc[df_trainCombinevalid["sqft_lot15"]>800000]).index,inplace=True)
df_trainCombinevalid


# In[14]:


#刪除sqft_above的離群值

df_trainCombinevalid.drop(index=(df_trainCombinevalid.loc[df_trainCombinevalid["sqft_above"]>8000]).index,inplace=True)
df_trainCombinevalid


# # 看其他特徵與price的相關係數

# In[15]:


data_names = ['price',
              'sale_yr',
              'sale_month',
              'sale_day',
              'bedrooms',
              'bathrooms',
              'sqft_living',
              'sqft_lot',
              'floors',
              'waterfront',
              'view',
              'condition',
              'grade',
              'sqft_above',
              'sqft_basement',
              'yr_built',
              'yr_renovated',
              'zipcode',
              'lat',
              'long',
              'sqft_living15',
              'sqft_lot15']
corrMat = df_trainCombinevalid[data_names].corr()
mask = np.array(corrMat)
mask[np.tril_indices_from(mask)] = False
plt.subplots(figsize=(20,10))
plt.xticks(rotation=60)#设置刻度标签角度
sns.heatmap(corrMat, mask=mask,vmax=.8, square=True,annot=True)


# In[16]:


print(corrMat["price"].sort_values(ascending=False))


# In[17]:


features = [  'sale_yr',
            'bedrooms',
              'bathrooms',
              'sqft_living',
              'sqft_lot',
              'floors',
              'waterfront',
              'view',
              'condition',
              'grade',
              'sqft_above',
              'sqft_basement',
              'yr_built',
              'yr_renovated',
              'lat',
              'long',
              'sqft_living15',
              'sqft_lot15']
X_train = df_trainCombinevalid[features]
Y_train = df_trainCombinevalid["price"]
X_test = df_test[features]
X_valid = df_valid[features]
Y_valid = df_valid["price"]

# 'sale_yr',
#             'bedrooms',
#               'bathrooms',
#               'sqft_living',
#               'sqft_lot',
#               'floors',
#               'waterfront',
#               'view',
#               'condition',
#               'grade',
#               'sqft_above',
#               'sqft_basement',
#               'yr_built',
#               'yr_renovated',
#               'lat',
#               'long',
#               'sqft_living15',
#               'sqft_lot15'


# In[18]:


#導入數據標準化模組
from sklearn import preprocessing
ss_X= preprocessing.StandardScaler()
ss_X.fit(X_train)
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.fit_transform(X_test)
X_valid = ss_X.fit_transform(X_valid)


# In[19]:


model = Sequential()
model.add(Dense(80,input_dim=len(list(features)), kernel_initializer='normal', activation='relu'))

model.add(Dense(100, activation='relu', kernel_initializer='normal'))
model.add(Dense(400, activation='relu', kernel_initializer='normal'))

model.add(Dense(100, activation='relu', kernel_initializer='normal'))

model.add(Dense(units=1, activation='relu', kernel_initializer='normal'))
model.compile(loss='MAE', optimizer='adam')
model.summary()


# In[20]:


model.fit(X_train, Y_train,epochs=200, batch_size=32, verbose=1,validation_data=(X_valid, Y_valid)) 


#,validation_data=(X_valid, Y_valid)


# 

# In[21]:


Y_test = model.predict(X_test)
Y_test = Y_test.tolist()


# In[22]:


for i in range(len(Y_test)):
    Y_test[i].insert(0,i+1)


# In[23]:


Y_test


# In[24]:


np.savetxt('predict.csv', Y_test, delimiter=',')


# In[25]:


final_df=pd.DataFrame(Y_test,columns=['id','price'])


# In[26]:


final_df.to_csv('final_dftest3.csv',index=False)


# In[ ]:




