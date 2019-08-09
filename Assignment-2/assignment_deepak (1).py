#!/usr/bin/env python
# coding: utf-8

# In[112]:


#importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[93]:


#reading wine datasets

data_red=pd.read_csv("winequality-red.csv",sep=';')
data_white=pd.read_csv("winequality-white.csv",sep=";")


# In[94]:


#joining two dataframes 

f_df=pd.DataFrame.append(data_red,data_white,)
f_df.insert(0, 'unit_matriix', 1)
#shuffling of dataframe s

f_df = f_df.sample(frac=1).reset_index(drop=True)
f_df.head(3)


# In[95]:


# confirming the join of two dataframe 
print(data_red.shape,data_white.shape,f_df.shape)

#checking the null values

f_df.isnull().sum()


# In[96]:


#dropping column y and spliting the data into train and train


f_df_new=f_df.drop('quality',axis=1)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(f_df_new, f_df['quality'], test_size=0.20, random_state=42)


# In[104]:


def linear_regression(xtrain,ytrain,alpha,itera):
    n=len(xtrain.columns) #no.of features
    m=len(xtrain.index) #no. of observations
    beta=np.zeros(n)  #coefficient matrix of zeros
    beta=beta.reshape(n,1) #reshaping 
    ypred=np.dot(xtrain,beta) #yprediction initialisation by matrix multiplication
    print(beta.shape, "  \n",ypred.shape) 
    ytrain= np.array(ytrain)
    ytrain=ytrain.reshape(m,1)
    
    #basic updationof ypred in a loop
    for i in range(0,itera):
        hx=np.dot(xtrain,beta) # hx is function of beta and beta is matrix multiplied with xtrain
        
        error=hx-ytrain # residual error of hx func and ytrain
        delta=(alpha/m)*(np.dot(xtrain.T,error)) # delta is partial derivative of cost function
        delta=np.array(delta)
        beta=beta-delta.reshape(12,1)
        cost = np.sum((error ** 2)) / (2 * m) #cost function
        ypred=np.dot(xtrain,beta) #final ypred after every iteration
        
      
    return ypred,cost,beta


# In[105]:


ypred,cost,beta=linear_regression(xtrain,ytrain,0.000080,10000)


# In[106]:


cost


# In[107]:


beta


# In[113]:


print(mean_squared_error(ytrain, ypred),r2_score(ytrain, ypred))


# In[114]:


ytest_pred=np.dot(xtest,beta)


# In[116]:


print(mean_squared_error(ytest, ytest_pred),r2_score(ytest, ytest_pred))


# In[ ]:




