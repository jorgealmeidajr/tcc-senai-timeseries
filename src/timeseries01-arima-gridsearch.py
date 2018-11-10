
import warnings
import pandas as pd
import numpy as np

import arima
import data

warnings.filterwarnings("ignore")



def main():
  arima_params = arima.get_arima_params(
    p_values=range(0, 6), 
    d_values=range(0, 5), 
    q_values=range(0, 5)
  )

  print(' > Numero de parametros para o ARIMA: %s' % len(arima_params))
  #grid_search_ts_daily(arima_params)
  grid_search_ts_monthly(arima_params)


def grid_search_ts_daily(arima_params):
  print(' > GRID SEARCH na serie DIARIA')

  print()


def grid_search_ts_monthly(arima_params):
  print(' > GRID SEARCH na serie MENSAL')

  # carrego a serie temporal mensal
  df_monthly = data.load_timeseries01_monthly()

  # primeira transformacao
  log_df_monthly = np.log(df_monthly)

  # segunda transformacao
  log_df_monthly_diff = log_df_monthly - log_df_monthly.shift()
  log_df_monthly_diff.dropna(inplace=True)

  dataset = log_df_monthly.values
  dataset = dataset.astype('float32')

  train, test = arima.split_dataset(dataset, porcentagem=0.66, debug=False)

  arima.evaluate_models(train, test, arima_params, 'output\\ts01-m-arima.csv')
  print()



if __name__ == '__main__':
  main()
