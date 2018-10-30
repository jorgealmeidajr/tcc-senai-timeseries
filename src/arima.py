
import sys

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# retorna uma combinacao dos parametros para serem avaliados
def get_arima_params(p_values, d_values, q_values):
  arima_params = []

  for p in p_values:
    for d in d_values:
      for q in q_values:
        arima_params.append((p, d, q))

  return arima_params


def split_dataset(dataset, porcentagem, debug=False):
  if debug:
    print('Parameter \'dataset\' length: ', len(dataset))

  train_size = int(len(dataset) * porcentagem)
  
  if debug:
    print('Variable \'train_size\' value: ', train_size)

  train, test = dataset[0:train_size], dataset[train_size:]

  if debug:
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
    model_fit = model.fit(disp=0)
    
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))

  # calculate out of sample error
  error = 0
  try:
    error = mean_squared_error(test, predictions)
  except:
    print('Erro ao calcular MSE, predictions: ', predictions)
    raise

  return error


# avalia combinacoes de (p, d, q) para o modelo ARIMA
def evaluate_models(dataset, arima_params):
  dataset = dataset.astype('float32')
  best_score, best_cfg = float("inf"), None
  
  train, test = split_dataset(dataset, porcentagem=0.66, debug=True)

  for params in arima_params:
    try:
      mse = evaluate_arima_model(train, test, params)
      
      if mse < best_score:
        best_score, best_cfg = mse, params
        
      print('ARIMA%s MSE=%.9f' % (params, mse))

    except:
      print("Configuracao instavel: ", str(params), ", Error: ", sys.exc_info()[0])
      continue

  print('Best ARIMA%s MSE=%.9f' % (best_cfg, best_score))


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

