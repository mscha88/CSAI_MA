
# coding: utf-8

# In[1]:


#import required packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler 
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns


# In[2]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)
datat = data.T
data_cols = datat.columns
data = datat.values


# In[3]:


data_sc1 = pd.DataFrame(data = data)


# In[4]:


# ADF test

adf = []
for i in data_sc1:
    series = data_sc1[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[5]:


# 1st difference
data_sc1_diff = data_sc1.diff().dropna()


# In[6]:


# ADF test

adf = []
for i in data_sc1_diff:
    series = data_sc1_diff[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[7]:


# 2st difference
data_sc1_diff2 = data_sc1_diff.diff().dropna()


# In[8]:


# ADF test

adf = []
for i in data_sc1_diff2:
    series = data_sc1_diff2[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[9]:


# 3st difference
data_sc1_diff3 = data_sc1_diff2.diff().dropna()


# In[10]:


# ADF test

adf = []
for i in data_sc1_diff3:
    series = data_sc1_diff3[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[11]:


# 4st difference
data_sc1_diff4 = data_sc1_diff3.diff().dropna()


# In[12]:


# ADF test

adf = []
for i in data_sc1_diff4:
    series = data_sc1_diff4[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[13]:


# 5st difference
data_sc1_diff5 = data_sc1_diff4.diff().dropna()


# In[14]:


# ADF test

adf = []
for i in data_sc1_diff5:
    series = data_sc1_diff5[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[15]:


# 6st difference
data_sc1_diff6 = data_sc1_diff5.diff().dropna()


# In[16]:


# ADF test

adf = []
for i in data_sc1_diff6:
    series = data_sc1_diff6[i]
    result = adfuller(series, autolag='AIC')
   # print(f'ADF Statistic: {result[0]}')
    #print(f'p-value: {result[1]}')
    adf.append(result[1])
dataframe = pd.DataFrame(adf)
# print non stationary sales time series
stationary = dataframe.loc[dataframe[0] <= 0.05]
non_stationary = dataframe.loc[dataframe[0] > 0.05]
print("stationary;", stationary.count())
print("non_stationary:", non_stationary.count())


# In[17]:


print(array(data_sc1_diff6).shape)


# In[18]:


data = array(data_sc1_diff6)


# In[19]:


scaler = MinMaxScaler()
data_l = data
data_sc = scaler.fit(data_l)
dataset = scaler.transform(data_l)
dataset1 = dataset[0:60,]
data = dataset1


# In[20]:


print(data.shape)


# In[22]:


# data split
n_test = 12

# split the dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=1):
	# convert config to a key
	n_input = config
	key = n_input
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return (key, scores)

# walk-forward validation
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	train1 = train
	# fit model
	model_fitted, n_features, y = model_fit(train, cfg)
	n_input = cfg
	out = durbin_watson(model_fitted.resid)
	#print(out)
	low = 0
	high = 0
	total = 0   
	value = []
	for col, val in zip(data_cols, out):
		if val > 1.5 and val < 2.5:
			low += 1
			total += 1 
		if val < 1.5 or val > 2.5:
			high += 1 
			total += 1
		value.append(val)
	pp = PdfPages('VAR DWS'+str(n_input)+'.pdf')
	axes = plt.subplots(0,2)
	pyplot.title("VAR - Durbin Watson’s Statistic: " + str(n_input))
	pyplot.ylabel("Durbin Watson’s Statistic")
	axes = sns.violinplot(x=out)
	pp.savefig()
	pp.close()
	#seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model_fitted, history, cfg, n_features, y)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions, train1, test, cfg)
	#print(error)
#	print(' > %.3f' % error)
	return error


# fit a model
def model_fit(train, config):
	# unpack config
	n_input = config
	# prepare data
	X, y = split_sequences(train, n_input)
	X = np.asarray(X)
	y = np.asarray(y)
# flatten output
	n_input1 = X.shape[1] * X.shape[2]
	train_x = X.reshape((X.shape[0], n_input1))
# flatten output
	n_output = y.shape[1] * y.shape[2]
	y = y.reshape((y.shape[0], n_output))
# the dataset knows the number of features, e.g. 2
	n_features = X.shape[0]    
# define model
	model = VAR(endog=train_x)
	model_fit = model.fit()
	return model_fit, n_features, y

# split a multivariate sequence into samples
def split_sequences(sequences, n_input):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_input[0]
		out_end_ix = end_ix + n_input[0]
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# forecast with a pre-fit model
def model_predict(model, history, config, n_features, y):
	# unpack config
	n_input = config
	# prepare data
	x_input = array(history[-12:])
	yhat = model.forecast(y, steps=12)
	return yhat[0]

# root mean squared error
def measure_rmse(actual, predicted, train1, test, cfg):
	# unpack config
	n_input = cfg
	#print(len(predicted))
	#print(actual.shape)
	#print(train1.shape)
	predicted1 = array(predicted).reshape(n_input[0]*12, test.shape[1])
	predicted1 = predicted1[0:12,:]
	predicted1 = predicted1.reshape(test.shape[1], 12)
	actual = actual.reshape(test.shape[1], 12)
	train1 = train1.reshape(test.shape[1], 48)
	start = np.empty([test.shape[1], 48])
	d = hstack((train1, actual))
	p = hstack((train1, predicted1))
	e = []
	for i in range(actual.shape[1]):
		for j in range(actual.shape[0]):
			error = sqrt(((actual[j,i] - predicted1[j,i])**2)/2)
			e.append(error)
	return e

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
    # sort configs by error, asc
	scores.sort(key=lambda tup: tup[0])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [1, 3, 6, 12]
	# create configs
	configs = list()
	for i in n_input:
		cfg = [i]
		configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs


# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scm = []
	for i in range(len(scores[0])):
		for l in range(len(scores)):
			sc = (scores[l])
			sc = (sc[i])
			scm.append(sc)
		scores_m, score_std = mean(scm), std(scm)

# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:12]:
	print(cfg, round(mean(error), 4))

