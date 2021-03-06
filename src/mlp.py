
import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from sklearn.metrics import mean_squared_error

import constants

from typing import List
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')



path: str
desempenho_df: PandasDataFrame
desempenho = []


def get_abs_file_path(rel_path):
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  abs_file_path = os.path.join(path, rel_path)
  return abs_file_path


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


def evaluate_mlp_models(train: List, test: List, output: str):
  # get model configs
  cfg_list = constants.get_mlp_model_configs()

  # tenho que carregar o desempenho salvo num arquivo csv
  global path
  path = get_abs_file_path(output)

  global desempenho_df
  desempenho_df = pd.read_csv(path, names=['params', 'MSE'], header=0)
  
  global desempenho
  desempenho = desempenho_df.to_dict('records')
  
  # grid search
  scores = grid_search(train, test, cfg_list)

  df_desempenho = pd.read_csv(path, names=['params', 'MSE'], header=0)
  best_mlp = df_desempenho.loc[df_desempenho['MSE'].idxmin()]
  print('Melhores parametros: MLP=%s MSE=%.9f' % (best_mlp['params'], best_mlp['MSE']))


# grid search configs
def grid_search(train: List, test: List, cfg_list):
  # evaluate configs
  scores = []

  for config in cfg_list:
    config_dict = tuple_config_to_dict(config)

    global desempenho_df
    # verificar se a configuracao ja foi avaliada
    if len(desempenho_df.loc[desempenho_df['params'] == str(config_dict)]) > 0:
      print('[IGNORADO] MLP: ', config_dict, 'ja foi avaliado')
      continue

    repeat_evaluate(train, test, config)


  # sort configs by error, asc
  scores.sort(key=lambda tup: tup[1])
  return scores


# score a model, return None on failure
def repeat_evaluate(train: List, test: List, config, n_repeats=10):
  # convert config to a key
  key = str(config)

  # fit and evaluate the model n times
  scores = []

  #for r in range(n_repeats): # TODO n_repeats nao esta sendo usado
  walk_forward_validation(train, test, config)

  # TODO apartir daqui nao esta sendo usado
  # summarize score
  result = np.mean(scores)
  #print('> Model[%s] %.3f' % (key, result))
  return (key, result)


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


  best_score = 0.0
  if error < best_score:
    pass
  
  config_str = str(tuple_config_to_dict(cfg))

  global desempenho
  desempenho.append({'params': config_str, 'MSE': error})

  print('[SUCESSO] MLP=%s MSE=%.9f' % (config_str, error))

  global df_desempenho
  # salvar o desempenho do ARIMA de maneira incremental
  df_desempenho = pd.DataFrame(desempenho, columns = ['params', 'MSE'])
  global path
  df_desempenho.to_csv(path, index=False)

  return error


# forecast with the fit model
def model_predict(model, history, config):
  # unpack config
  n_input, _, _, _, _ = config

  # prepare data
  correction = 0.0
  
  # shape input for model
  x_input = np.array(history[-n_input:]).reshape((1, n_input))
  # make forecast
  yhat = model.predict(x_input, verbose=0)
  # correct forecast if it was differenced
  return correction + yhat[0]


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
  n_input, n_nodes, n_epochs, n_batch, n_layers = config

  config_dict = tuple_config_to_dict(config)
  #print(config_dict)

  # transform series into supervised format
  data = series_to_supervised(train, n_in=n_input)

  # separate inputs and outputs
  train_x, train_y = data[:, :-1], data[:, -1]

  # define model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(n_nodes, activation='relu', input_dim=n_input))

  for _ in range(n_layers):
    model.add(tf.keras.layers.Dense(n_nodes, activation='relu'))

  model.add(tf.keras.layers.Dense(1))
  model.compile(loss='mse', optimizer='adam')

  #model.summary()

  #monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
  #model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[monitor])

  # fit model
  model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
  return model


def tuple_config_to_dict(config):
  n_input, n_nodes, n_epochs, n_batch, n_layers = config

  config_dict = {
    'n_input': n_input, 
    'n_nodes': n_nodes, 
    'n_epochs': n_epochs,
    'n_batch': n_batch, 
    'n_layers': n_layers
  }

  return config_dict

