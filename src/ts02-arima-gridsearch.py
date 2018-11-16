
import warnings
import pandas as pd
import numpy as np

import arima
import data
import constants



OUTPUT_M2: str = 'output\\ts01-m2-arima.csv'
OUTPUT_M: str = 'output\\ts01-m-arima.csv'


def main():
  arima_params = constants.ARIMA_PARAMS
  print(' > Numero de parametros para o ARIMA: %s' % len(arima_params))
  
  # [ATENCAO] descomente para executar o grid search na serie mensal com dois valores
  grid_search_ts_monthly2(arima_params)

  # [ATENCAO] descomente para executar o grid search na serie mensal
  #grid_search_ts_monthly(arima_params)


def grid_search_ts_monthly2(arima_params):
  print(' > GRID SEARCH na serie MENSAL com duas amostras')
  
  # carrego a serie temporal mensal com duas amostras
  df_monthly2 = data.load_timeseries02_monthly2()

  # primeira transformacao
  log_df_monthly2 = np.log(df_monthly2)

  # segunda transformacao
  log_df_monthly2_diff = log_df_monthly2 - log_df_monthly2.shift()
  log_df_monthly2_diff.dropna(inplace=True)

  dataset = log_df_monthly2.values
  dataset = dataset.astype('float32')

  train, test = arima.split_dataset(dataset, porcentagem=constants.PORCENTAGEM, debug=False)

  arima.evaluate_models(train, test, arima_params, OUTPUT_M2)
  print()


def grid_search_ts_monthly(arima_params):
  print(' > GRID SEARCH na serie MENSAL')

  # carrego a serie temporal mensal
  df_monthly = data.load_timeseries02_monthly()

  # primeira transformacao
  log_df_monthly = np.log(df_monthly)

  # segunda transformacao
  log_df_monthly_diff = log_df_monthly - log_df_monthly.shift()
  log_df_monthly_diff.dropna(inplace=True)

  dataset = log_df_monthly.values
  dataset = dataset.astype('float32')

  train, test = arima.split_dataset(dataset, porcentagem=constants.PORCENTAGEM, debug=False)

  arima.evaluate_models(train, test, arima_params, OUTPUT_M)
  print()



if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()
