
import warnings
import pandas as pd
import numpy as np

import data
import constants
import mlp



OUTPUT_M: str = 'output\\ts01\\ts01-m-mlp.csv'
OUTPUT_M2: str = 'output\\ts01\\ts01-m2-mlp.csv'

# carrego a serie temporal mensal com duas amostras
df_monthly2 = data.load_timeseries01_monthly2()

# carrego a serie temporal mensal
df_monthly = data.load_timeseries01_monthly()


def main():
  print('\nMLP: Serie Temporal TS01 - busca pela melhor REDE NEURAL')

  #arima_params = constants.ARIMA_PARAMS
  #print(' > Numero de parametros para o ARIMA: %s' % len(arima_params))
  
  # [ATENCAO] descomente para executar o grid search na serie mensal com dois valores
  #grid_search_ts_monthly2(arima_params)

  # [ATENCAO] descomente para executar o grid search na serie mensal
  grid_search_ts_monthly()


def grid_search_ts_monthly2():
  print(' > MLP: Grid Search na serie MENSAL com duas amostras')
  
  dataset = df_monthly2.values
  dataset = dataset.astype('float32')

  #train, test = arima.split_dataset(dataset, porcentagem=constants.PORCENTAGEM, debug=True)

  #arima.evaluate_models(train, test, arima_params, OUTPUT_M2)
  print()


def grid_search_ts_monthly():
  print('> MLP: Grid Search na serie MENSAL')
  train, test = mlp.split_to_train_test(df_monthly, porcentagem=constants.PORCENTAGEM, debug=True)

  mlp.evaluate_mlp_models(train, test, OUTPUT_M)
  print()



if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()
