
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from sklearn.metrics import mean_squared_error

import constants

from typing import List
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')



# split into train and test sets
def split_to_train_test(dataset: PandasDataFrame, porcentagem, debug=False):
  dataset = dataset.values
  dataset = dataset.astype('float32')

  train_size = int(len(dataset) * porcentagem)
  test_size = len(dataset) - train_size

  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

  if debug:
    print('Parameter \'dataset\' length: ', len(dataset))
    print('Dataset \'train\' length: ', len(train))
    print('Dataset \'test\' length: ', len(test))

  return (train, test)


def evaluate_mlp_models(train: List, test: List):
  # get model configs
  cfg_list = constants.get_mlp_model_configs()

  # grid search
  #scores = grid_search(train, test, cfg_list)


# transform list into supervised learning format
def series_to_supervised(data: List, n_in=1, n_out=1) -> List:
	df = pd.DataFrame(data)
	cols = list()

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))

	# put it all together
	agg = pd.concat(cols, axis=1)

	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values


# root mean squared error or RMSE
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))


# mean squared error or MSE
def measure_mse(actual, predicted):
	return mean_squared_error(actual, predicted)


# fit a model
def model_fit(train, config):
	# unpack config
	n_input, n_nodes, n_epochs, n_batch, n_diff = config
	
  # transform series into supervised format
	data = series_to_supervised(train, n_in=n_input)
	
  # separate inputs and outputs
	train_x, train_y = data[:, :-1], data[:, -1]
	
  # define model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(n_nodes, activation='relu', input_dim=n_input))
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss='mse', optimizer='adam')
	
  # fit model
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
	return model


# walk-forward validation for univariate data
def walk_forward_validation(train: List, test: List, cfg):
  predictions = list()

  # fit model
  model = model_fit(train, cfg)

  # seed history with training dataset
  history = [x for x in train]

  # step over each time-step in the test set
  for i in range(len(test)):
    # fit model and make forecast for history
    yhat = model_predict(model, history, cfg)
    # store forecast in list of predictions
    predictions.append(yhat)
    # add actual observation to history for the next loop
    history.append(test[i])

  # estimate prediction error
  #error = measure_rmse(test, predictions)
  error = measure_mse(test, predictions)

  print('MSE: %.9f' % error)
  return error


# score a model, return None on failure
def repeat_evaluate(train: List, test: List, config, n_repeats=10):
  # convert config to a key
  key = str(config)

  # fit and evaluate the model n times
  scores = [walk_forward_validation(train, test, config) for _ in range(n_repeats)]

  # summarize score
  result = mean(scores)
  print('> Model[%s] %.3f' % (key, result))
  return (key, result)


# grid search configs
def grid_search(train: List, test: List, cfg_list):
  # evaluate configs
  scores = [repeat_evaluate(train, test, cfg) for cfg in cfg_list]
  
  # sort configs by error, asc
  scores.sort(key=lambda tup: tup[1])
  return scores

