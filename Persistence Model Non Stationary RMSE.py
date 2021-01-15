
# coding: utf-8

# In[1]:


#import required packages
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.vector_ar.var_model import VAR
from numpy import array
import math
from sklearn.metrics import mean_squared_error
import statistics
from numpy import hstack
from matplotlib import pyplot
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# In[2]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)
datat = data.T


# In[3]:


scaler = MinMaxScaler()
data_l = datat.values
data_sc = scaler.fit(data_l)
dataset = scaler.transform(data_l)
dataset1 = dataset[0:60,]
data = dataset1


# In[4]:


print(data.shape)


# In[5]:


# naive forecast 
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
# split a univariate dataset into train/test sets
def split_dataset(data):
# split into standard month
    train, test = data[0:-12,], data[-12:,]
    #print(train.shape)
    #print(test.shape)
# restructure into windows 
   # train = array(split(train, len(train)/1))
    #test = array(split(test, len(test)/1))
   # print(train.shape)
    return train, test
# evaluate forecasts 
def evaluate_forecasts(actual, predicted):
    #print(len(actual))
    scores = list()
    # calculate an RMSE score for each day
    for i in range(12):
        #print("act", actual[:, i])
    # calculate mse
        mse = mean_absolute_error(actual[i], predicted[i])
      #  print("mse", mse)
        # calculate rmse
        rmse = sqrt(mse)
        # store
   #     print("rmse", rmse)
        scores.append(rmse)
   # print(len(scores))
    # calculate overall RMSE
   # s = 0
    #for row in range(actual.shape[0]):
     #   for col in range(12):
      #      s += (actual[row, col] - predicted[row, col])**2
    #score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return scores
# summarize scores
def summarize_scores(name, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
  #  print('%s: [%.3f] %s' % (name, s_scores))
# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = []
    history = [x for x in train]
    #print(len(history))
   # print(len(test))
    # walk-forward validation over each week
    predictions = list()
   # for i in range(len(test)):
    # predict
    # get real observation and add to history for predicting 
    history.append(test[-12:])
  #  print("history", len(history))
    yhat_sequence = model_func(history)
    # store the predictions
  #  print("yhat", len(yhat_sequence))
   # predictions.append(yhat_sequence)
    #print(predictions)
    predictions = yhat_sequence
 #  print("pred", len(predictions))
  #  print(predictions)
   # print(history)
    
    # evaluate predictions 
    scores = evaluate_forecasts(history, predictions)
   # print(mean(scores))
    l = []
    l.append(scores)
   # print(len(l))
    return scores
# daily persistence model
def m_persistence(history):
    last_week = history[-11]
   # print("last_w", len(last_week))
    value = last_week
   # print(value)
# prepare forecast
    forecast = [value for _ in range(12)]
   # print("for", len(forecast))
  #  print(forecast)
    return forecast

# split into train and test
ss = []
for i in range(0,606,1):
    data1 = dataset1[0:60,i]
    data = data1.reshape(60,1)
    train, test = split_dataset(data)
#train, test = split_dataset(data)
# define the names and functions for the models we wish to evaluate
    models = dict()
    models['monthly'] = m_persistence
    # evaluate each model
    days = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    for name, func in models.items():
    # evaluate and get scores
        #print(name, func)
        scores = evaluate_model(func, train, test)
    # summarize scores
        ss.append(mean(scores))
      #  print("ss", len(ss))
       # print(ss)
print(round(mean(ss),4))

