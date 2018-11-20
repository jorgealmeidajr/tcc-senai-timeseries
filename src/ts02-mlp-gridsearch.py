
import warnings
import pandas as pd
import numpy as np

import data
import constants
import mlp



OUTPUT_M: str = 'output\\ts02\\ts02-m-mlp.csv'
OUTPUT_M2: str = 'output\\ts02\\ts02-m2-mlp.csv'

# carrego a serie temporal mensal com duas amostras
df_monthly2 = data.load_timeseries02_monthly2()

# carrego a serie temporal mensal
df_monthly = data.load_timeseries02_monthly()


def main():
  print('\nMLP: Busca pela melhor REDE NEURAL - Serie Temporal TS02')
  #grid_search_ts_monthly2()
  #grid_search_ts_monthly()


def grid_search_ts_monthly2():
  print(' > MLP: Grid Search na serie MENSAL com duas amostras')
  train, test = mlp.split_to_train_test(df_monthly2, porcentagem=constants.PORCENTAGEM, debug=True)

  mlp.evaluate_mlp_models(train, test, OUTPUT_M2)
  print()


def grid_search_ts_monthly():
  print('> MLP: Grid Search na serie MENSAL')
  train, test = mlp.split_to_train_test(df_monthly, porcentagem=constants.PORCENTAGEM, debug=True)

  mlp.evaluate_mlp_models(train, test, OUTPUT_M)
  print()



if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()
