
import sys
import time
import os

import pandas as pd
import numpy as np

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt



# retorna uma combinacao dos parametros para serem avaliados
def get_arima_params(p_values, d_values, q_values):
  arima_params = []

  for p in p_values:
    for d in d_values:
      for q in q_values:
        arima_params.append((p, d, q))

  return arima_params


def split_dataset(dataset, porcentagem, debug=False):
  train_size = int(len(dataset) * porcentagem)
  
  train, test = dataset[0:train_size], dataset[train_size:]

  if debug:
    print('Parameter \'dataset\' length: ', len(dataset))
    print('Dataset \'train\' length: ', len(train))
    print('Dataset \'test\' length: ', len(test))
  
  return (train, test)


# evaluate an ARIMA model for a given order (p, d, q)
def evaluate_arima_model(train, test, arima_order):
  history = [x for x in train]
  
  # make predictions
  predictions = list()

  for t in range(len(test)):
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=False)    
    output = model_fit.forecast()
    
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))

  # calculate out of sample error
  error = mean_squared_error(test, predictions)
  return error


# avalia combinacoes de (p, d, q) para o modelo ARIMA
def evaluate_models(train, test, arima_params, output):
  best_score, best_cfg = float("inf"), None
  
  # tenho que carregar o desempenho salvo num arquivo csv
  path = get_abs_file_path(output)

  df = pd.read_csv(path, names=['params', 'MSE', 'status'], header=0)
  desempenho = df.to_dict('records')

  for params in arima_params:
    try:
      # verificar se o desempenho ja foi avaliado
      if len(df.loc[df['params'] == str(params)]) > 0:
        print('[IGNORADO] ARIMA', params, ' ja foi avaliado')
        continue

      # TODO implementar um benchmarking no momento do treinamento do modelo
      # fazer o mesmo com redes neurais para comparacao
      #t0= time.clock()

      mse = evaluate_arima_model(train, test, params)

      #t1 = time.clock() - t0      
      #execution_time = t1 - t0 # CPU seconds elapsed (floating point)

      if mse < best_score:
        best_score, best_cfg = mse, params
      
      desempenho.append({'params': params, 'MSE': mse, 'status': 'SUCESSO'})

      print('[SUCESSO] ARIMA%s MSE=%.9f' % (params, mse))

      # salvar o desempenho do ARIMA de maneira incremental
      df_desempenho = pd.DataFrame(desempenho, columns = ['params', 'MSE', 'status'])
      df_desempenho.to_csv(path, index=False)

    except:
      # estou ignorando parametros instaveis que produzem erros
      print('[FALHA] A configuracao ARIMA', params, ' falhou')

      desempenho.append({'params': params, 'MSE': None, 'status': 'FALHA'})

      # salvar o desempenho do ARIMA de maneira incremental
      df_desempenho = pd.DataFrame(desempenho, columns = ['params', 'MSE', 'status'])
      df_desempenho.to_csv(path, index=False)
      continue

  df_desempenho = pd.read_csv(path, names=['params', 'MSE', 'status'], header=0)
  best_arima = df_desempenho.loc[df_desempenho['MSE'].idxmin()]
  print('Melhores parametros: ARIMA%s MSE=%.9f' % (best_arima['params'], best_arima['MSE']))
  #print('Best ARIMA%s MSE=%.9f' % (best_cfg, best_score))


def get_abs_file_path(rel_path):
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  abs_file_path = os.path.join(path, rel_path)
  return abs_file_path


def print_dataframe_info(df):
  #print(str(df.info()) + '\n')
  print('SHAPE: ' + str(df.shape) + '\n')
  print('DTYPES: ' + str(df.dtypes) + '\n')
  
  print('#' * 90)
  print('HEAD \n')
  print(str(df.head(5)) + '\n\n')
  
  print('#' * 90)
  print('TAIL \n')
  print(str(df.tail(5)))


def forecast(train, test, arima_params):
  predictions = list()
  historical = list()

  model = ARIMA(train, order=arima_params)
  model_fit = model.fit(disp=-1)
  forecast = model_fit.forecast(steps=len(test))[0]

  t = 0
  for yhat in forecast:
    observed = np.exp(test['rate'][t])
    historical.append(observed)

    predicted = np.exp(float(yhat))
    predictions.append(predicted)

    #print('Predicted = %.9f, Expected = %.9f' % (predicted, observed))
    t += 1
  
  return (historical, predictions)


def plot_historical_and_predictions(historical, predictions, test):
    error = mean_squared_error(historical, predictions)
    print('Test MSE: %.9f' % error)

    historical = pd.Series(historical, index=test.index)
    predictions = pd.Series(predictions, index=test.index)

    plt.plot(historical)
    plt.plot(predictions, color='red')
    plt.show()

