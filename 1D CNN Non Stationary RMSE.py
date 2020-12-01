
# coding: utf-8

# In[2]:


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
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# In[3]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)
datat = data.T
data_l = datat.values


# In[4]:


scaler = MinMaxScaler()
data_sc = scaler.fit(data_l)
dataset = scaler.transform(data_l)
dataset1 = dataset[0:60,]
data = dataset1


# In[5]:


data = dataset[0:60,]


# In[6]:


print(data.shape)


# In[7]:


# data split
n_test = 12
# choose a number of time steps
n_steps_out = 12


# split the dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# split dataset
train, test = train_test_split(data, n_test)

# repeat evaluation of a config
def repeat_evaluate(data, config, n_test, n_repeats=5):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	#print(len(scores))
	#print(len(scores[0]))
	#print(scores, _)
	return (key, scores)

# walk-forward validation
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	train1 = train
	#print("trst", test.shape)
	# fit model
	model, n_features = model_fit(train, cfg)
	#print(model)
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
	#print("predictions", len(predictions))
	error = measure_rmse(test, predictions, train1)
	#print(error)
#	print(' > %.3f' % error)
	return error


# fit a model 
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch, n_act, n_kern, n_filt, n_pool  = config
	# prepare data
	X, y = split_sequences(train, n_input, n_steps_out)
	#train_x, train_y = data[:, :-1], data[:, -1]
	X = np.asarray(X)
	#print(X.shape)
	y = np.asarray(y)
# flatten output
	n_output = y.shape[1] * y.shape[2]
	y = y.reshape((y.shape[0], n_output))
# the dataset knows the number of features
	n_features = X.shape[2]   
    # define model
# define model  
	model = Sequential()
	model.add(Conv1D(filters=n_filt, kernel_size=n_kern, activation=n_act, input_shape=(n_input, n_features)))
	model.add(MaxPooling1D(pool_size=n_pool))
	model.add(Flatten())
	model.add(Dense(n_nodes, activation=n_act))
	model.add(Dense(n_output))
	model.compile(optimizer='adam', loss='mse')
# fit model
	model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
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
	n_input, _, _, _, _, _, _, _ = config
	# prepare data
	x_input = array(history[-n_input:])
	#print(x_input.shape)
	x_input = x_input.reshape(1, n_input, n_features)  
	# forecast
	#print(x_input.shape)
	yhat = model.predict(x_input, verbose=0)
	#print("yhat", yhat.shape)
	return yhat[0]

# root mean squared error
def measure_rmse(actual, predicted, train1):
	#print(len(predicted))
	#print(actual.shape)
	predicted1 = array(predicted).reshape(144, test.shape[1])
	predicted1 = predicted1[0:12,:]
	predicted1 = predicted1.reshape(test.shape[1], 12)
	actual = actual.reshape(test.shape[1], 12)
	train1 = train.reshape(test.shape[1], 48)
	#print(predicted1.shape)
	start = np.empty([test.shape[1], 48])
	d = hstack((train1, actual))
	p = hstack((train1, predicted1))
	#for i in range(test.shape[1]):
		#pyplot.plot(d[i,:])
		#pyplot.plot(p[i,:])
		#pyplot.show()
	e = []
	for i in range(actual.shape[1]):
		for j in range(actual.shape[0]):
			error = sqrt(((actual[j,i] - predicted1[j,i])**2)/2)
			e.append(error)
	#print(len(e))
	return e

# grid search
#scores = repeat_evaluate(data, config, n_test)

# grid search configs
def grid_search(data, cfg_list, n_test):
	# evaluate configs
	scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	#print(len(scores))
    # sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a list of configs to try
def model_configs():
	# define scope of configs
	n_input = [3, 6, 12]
	n_nodes = [100]
	n_epochs = [500]
	n_batch = [8]
	n_act = ["tanh"]
	n_kern = [2]
	n_filt = [32]
	n_pool = [2]
	# create configs
	configs = list()
	for i in n_input:
		for j in n_nodes:
			for k in n_epochs:
				for l in n_batch:
					for m in n_act:
						for n in n_kern:
							for o in n_filt:
								for p in n_pool:
									cfg = [i, j, k, l, m, n, o, p]
									configs.append(cfg)
	print('Total configs: %d' % len(configs))
	return configs


# summarize model performance
def summarize_scores(name, scores):
	# print a summary
	scm = []
	for i in range(len(scores[0])):
		for l in range(len(scores)):
#			print((scores[l]))
			sc = (scores[l])
			#print((len(sc)))
			#sc = array(sc).reshape(test.shape[0], test.shape[1])
			sc = (sc[i])
			scm.append(sc)
		#print(scm)
		scores_m, score_std = mean(scm), std(scm)
		print('%s: %.3f RMSE (+/- %.3f)' % (name, scores_m, score_std))
		# box and whisker plot
		pyplot.boxplot(scm)
		pyplot.show()
		scm = []
              
#summarize_scores('mlp', scores)

# model configs
cfg_list = model_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:10]:
	print(cfg, "RMSE:", round(mean(error), 4))

