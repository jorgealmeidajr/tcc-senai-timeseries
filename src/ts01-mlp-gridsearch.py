
import warnings
import pandas as pd
import numpy as np

import data
import constants
import mlp



#OUTPUT_M: str = 'output\\ts01-m-mlp.csv'

# carrego a serie temporal mensal com duas amostras
df_monthly2 = data.load_timeseries01_monthly2()

# carrego a serie temporal mensal
df_monthly = data.load_timeseries01_monthly()


def main():
  print('\nGRID SEARCH na serie TS01 - busca pela melhor REDE NEURAL')

  #arima_params = constants.ARIMA_PARAMS
  #print(' > Numero de parametros para o ARIMA: %s' % len(arima_params))
  
  # [ATENCAO] descomente para executar o grid search na serie mensal com dois valores
  #grid_search_ts_monthly2(arima_params)

  # [ATENCAO] descomente para executar o grid search na serie mensal
  grid_search_ts_monthly()


def grid_search_ts_monthly2():
  print(' > GRID SEARCH na serie MENSAL com duas amostras')
  
  dataset = df_monthly2.values
  dataset = dataset.astype('float32')

  #train, test = arima.split_dataset(dataset, porcentagem=constants.PORCENTAGEM, debug=True)

  #arima.evaluate_models(train, test, arima_params, OUTPUT_M2)
  print()


def grid_search_ts_monthly():
  print('> GRID SEARCH na serie MENSAL')

  train, test = mlp.split_to_train_test(df_monthly, porcentagem=constants.PORCENTAGEM, debug=True)
  mlp.evaluate_mlp_models(train, test)
  
  print()



if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()

