
import os
import pandas as pd
import numpy as np




# Serie Temporal 01
# cotacao do dolar em relacao ao real - 18 anos
def load_timeseries01_original():
  script_path = os.path.abspath(__file__)
  path = script_path.rpartition('\\src\\')[0]

  rel_path = "output\\timeseries01.csv"
  abs_file_path = os.path.join(path, rel_path)

  return pd.read_csv(abs_file_path, header=0, names=['date', 'rate'])
  #return pd.read_csv('../output/timeseries01.csv', header=0, names=['date', 'rate'])


def load_timeseries01_daily():
  df_original = load_timeseries01_original()

  df_daily = df_original.copy()
  df_daily['date'] = pd.to_datetime(df_daily['date'])
  df_daily = df_daily.set_index('date')

  #idx = pd.date_range('01-02-1995', '09-21-2018')
  #df = df.reindex(idx, fill_value=0)
  df_daily = df_daily.resample('D').ffill()

  df_daily = df_daily.replace(0, np.nan)
  df_daily = df_daily.replace('.', np.nan)

  df_daily = df_daily.fillna(method='ffill')

  df_daily['rate'] = pd.to_numeric(df_daily['rate'])

  # removo o periodo inicial da serie temporal
  # tem um padrao que nao se repete
  # estou trabalhando com dados de 18 anos
  df_daily = df_daily['2000-11-03':]

  return df_daily


def load_timeseries01_weekly():
  df_daily = load_timeseries01_daily()

  df_weekly = df_daily.resample('W-FRI')
  df_weekly = df_weekly.mean()

  return df_weekly


def load_timeseries01_monthly():
  df_daily = load_timeseries01_daily()

  df_monthly = df_daily.resample('M')
  df_monthly = df_monthly.mean()

  return df_monthly





# Serie Temporal 02
# Cotacao do Fundo de Investimento de Renda Fixa - 7 anos
def load_timeseries02_original():
  return pd.read_csv('../data/fundo01-cotas-rendafixa.csv', header=0, encoding='iso-8859-1')


def load_timeseries02_daily():
  df_original = load_timeseries02_original()

  # remove as colunas desnecessarias
  # axis=1 diz que estou removendo colunas
  df_daily = df_original.drop('Código', axis=1)
  df_daily = df_daily.drop('Fundo', axis=1)
  df_daily = df_daily.drop('Variação', axis=1)
  df_daily = df_daily.drop('Captação', axis=1)
  df_daily = df_daily.drop('Resgate', axis=1)
  df_daily = df_daily.drop('PL', axis=1)
  df_daily = df_daily.drop('Cotistas', axis=1)

  df_daily['Data'] = pd.to_datetime(df_daily['Data'], format='%d/%m/%Y')
  df_daily.index = df_daily['Data']
  del df_daily['Data']

  df_daily['Cota'] = df_daily['Cota'].apply(lambda x: float(x.replace(',', '.')))
  df_daily['Cota'] = df_daily['Cota'].astype(float)

  df_daily = df_daily.resample('D').ffill()

  if df_daily.isnull().values.any():
    raise Exception('o dataframe diario nao pode ter valores NaN')

  # estou trabalhando com dados de 7 anos
  df_daily = df_daily['2011-09-27':]

  return df_daily


def load_timeseries02_monthly():
  df_daily = load_timeseries02_daily()

  df_monthly = df_daily.resample('M')
  df_monthly = df_monthly.mean()

  return df_monthly





# Serie Temporal 03
# Cotacao do Fundo de Investimento de Acoes - 3 anos
def load_timeseries03_original():
  return pd.read_csv('../data/fundo02-cotas-acoes.csv', header=0, encoding='iso-8859-1')


def load_timeseries03_daily():
  df_original = load_timeseries03_original()

  # axis=1 diz que estou removendo colunas
  df_daily = df_original.drop('Código', axis=1)
  df_daily = df_daily.drop('Fundo', axis=1)
  df_daily = df_daily.drop('Variação', axis=1)
  df_daily = df_daily.drop('Captação', axis=1)
  df_daily = df_daily.drop('Resgate', axis=1)
  df_daily = df_daily.drop('PL', axis=1)
  df_daily = df_daily.drop('Cotistas', axis=1)

  df_daily['Data'] = pd.to_datetime(df_daily['Data'], format='%d/%m/%Y')
  df_daily.index = df_daily['Data']
  del df_daily['Data']

  df_daily['Cota'] = df_daily['Cota'].apply(lambda x: float(x.replace(',', '.')))
  df_daily['Cota'] = df_daily['Cota'].astype(float)

  df_daily = df_daily.resample('D').ffill()

  if df_daily.isnull().values.any():
    raise Exception('o dataframe diario nao pode ter valores NaN')

  # estou trabalhando com dados de 3 anos
  df_daily = df_daily['2015-09-27':]

  return df_daily


def load_timeseries03_monthly():
  df_daily = load_timeseries03_daily()

  df_monthly = df_daily.resample('M')
  df_monthly = df_monthly.mean()

  return df_monthly

