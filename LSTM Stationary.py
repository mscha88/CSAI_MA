
# coding: utf-8

# In[1]:


from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from numpy import hstack
import numpy as np
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras import backend
from matplotlib.backends.backend_pdf import PdfPages


# In[2]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)
datat = data.T


# In[3]:


# 6st difference
data_sc1_diff6 = datat.diff(6).dropna()


# In[4]:


print(data_sc1_diff6.shape)


# In[5]:


scaler = MinMaxScaler()
data_l = data_sc1_diff6
data_sc = scaler.fit(data_l)
dataset = scaler.transform(data_l)
dataset1 = dataset[0:60,]
data = dataset1


# In[6]:


print(data.shape)


# In[7]:


# data split
n_test = 12
# choose a number of time steps
#n_steps_in, n_steps_out = 6, 12


# split the dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# split dataset
train, test = train_test_split(data, n_test)

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=5):
	# convert config to a key
	n_input, n_nodes, n_epochs, n_batch = config
	key, node, epochs = n_input, n_nodes, n_epochs
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	return (key, node, epochs, scores)

# walk-forward validation 
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	train1 = train
	# fit model
	model, n_features = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg, n_features)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions, train1, cfg)
	return error


# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch = config
	# prepare data
	X, y = split_sequences(train, n_input, n_steps_out=12)
	X = np.asarray(X)
	y = np.asarray(y)
	n_features = X.shape[2]
# flatten output
	n_output = y.shape[1] * y.shape[2]
	y = y.reshape((y.shape[0], n_output))
# the dataset knows the number of features
# define model  
	def rmse(y_true, y_pred):
		return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
	model = Sequential()
	model.add(LSTM(n_nodes, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), return_sequences=False, stateful=True))
	model.add((Dense(n_output)))
	model.compile(optimizer='adam', loss=rmse, metrics=[rmse])
# fit model
	mse = []
	for i in range(n_epochs):
		history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		values = history.history['rmse']
		mse.append(sqrt(values[0]))
		model.reset_states()
	return model, n_features

# split a multivariate sequence into samples
def split_sequences(sequences, n_input, n_steps_out=12):
	X, y = list(), list()
	for i in range(len(sequences)):
# find the end of this pattern
		end_ix = i + n_input
		out_end_ix = end_ix + n_steps_out
# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# forecast with a pre-fit model
def model_predict(model, history, config, n_features):
	# unpack config
	n_input, _, _, _, = config
	# prepare data
	x_input = array(history[-n_input:])
	x_input = x_input.reshape(1, n_input, n_features) 
	# forecast
	yhat = model.predict(x_input, verbose=0)
	return yhat[0]

# root mean squared error
def measure_rmse(actual, predicted, train1, config):
	n_input, n_nodes, n_epochs, n_batch = config
	predicted1 = array(predicted).reshape(144, test.shape[1])
	predicted1 = predicted1[0:12,:]
	predicted1 = predicted1.reshape(test.shape[1], 12)
	actual = actual.reshape(test.shape[1], 12)
	train1 = train.reshape(test.shape[1], 48)
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
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [1, 3, 6, 12]
	n_nodes = [3]
	n_epochs = [10]
	n_batch = [1]
	# create configs
	configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					cfg = [i, j, k, l]
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
		print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))

# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')

for cfg, node, epochs, error in scores[:35]:
    print(cfg, node, epochs, "RMSE:", round(mean(error), 4))

