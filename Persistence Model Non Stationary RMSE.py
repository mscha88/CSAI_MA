
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


# In[9]:


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
    train = array(split(train, len(train)/3))
    test = array(split(test, len(test)/3))
    return train, test
# evaluate forecasts 
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        #print("act", actual[:, i])
    # calculate mse
        mse = mean_absolute_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = (mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    #print('%s: [%.3f] %s' % (name, score, s_scores))
# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
    # predict
        yhat_sequence = model_func(history)
    # store the predictions
        predictions.append(yhat_sequence)
    # get real observation and add to history for predicting 
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions 
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores
# daily persistence model
def m_persistence(history):
    last_week = history[-1]
    value = last_week[-1, 0]
# prepare forecast
    forecast = [value for _ in range(12)]
    return forecast

# split into train and test
ss = []
for i in range(606):
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
        score, scores = evaluate_model(func, train, test)
    # summarize scores
        ss.append(score)
        summarize_scores(name, score, scores)
    # plot scores
        #pyplot.plot(days, scores, marker='o', label=name)
    # show plot
    #pyplot.legend()
    #pyplot.show()
scores_m, score_std = mean(ss), std(ss)
print('%s: %.4f RMSE (+/- %.4f)' % (name, scores_m, score_std))
# box and whisker plot
pyplot.boxplot(ss)
pyplot.show()

