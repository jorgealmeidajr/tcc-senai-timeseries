
import warnings
import pandas as pd
import numpy as np

import arima
import data

warnings.filterwarnings("ignore")




def main():
  # carrego a serie temporal mensal
  df_monthly = data.load_timeseries01_monthly()

  # primeira transformacao
  log_df_monthly = np.log(df_monthly)

  # segunda transformacao
  log_df_monthly_diff = log_df_monthly - log_df_monthly.shift()
  log_df_monthly_diff.dropna(inplace=True)

  arima_params = arima.get_arima_params(
    p_values=range(0, 1), d_values=range(0, 1), q_values=range(0, 2)
  )

  dataset = log_df_monthly.values
  dataset = dataset.astype('float32')

  train, test = arima.split_dataset(dataset, porcentagem=0.66, debug=True)

  arima.evaluate_models(train, test, arima_params, 'output\\ts01-m-arima.csv')



if __name__ == '__main__':
  main()
