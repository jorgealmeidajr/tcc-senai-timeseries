
import pandas as pd
import numpy as np


# Serie Temporal 01
# cotacao do dolar em relacao ao real - 18 anos
def load_timeseries01_original():
  return pd.read_csv('../output/timeseries01.csv', header=0, names=['date', 'rate'])


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


def load_timeseries01_monthly():
  df_daily = load_timeseries01_daily()

  df_monthly = df_daily.resample('M')
  df_monthly = df_monthly.mean()

  return df_monthly



# Serie Temporal 02
# Cotacao do Fundo de Investimento de Renda Fixa - ? anos
def load_timeseries02_original():
  return pd.read_csv('../data/fundo01-cotas-rendafixa.csv', header=0, encoding='iso-8859-1')


def load_timeseries02():
  df_original = load_timeseries02_original()

  # remove as colunas desnecessarias
  df2 = df_original.drop('Código', axis=1)
  df2 = df2.drop('Fundo', axis=1)
  df2 = df2.drop('Variação', axis=1)
  df2 = df2.drop('Captação', axis=1)
  df2 = df2.drop('Resgate', axis=1)
  df2 = df2.drop('PL', axis=1)
  df2 = df2.drop('Cotistas', axis=1)

  df2['Data'] = pd.to_datetime(df2['Data'], format='%d/%m/%Y')
  df2.index = df2['Data']
  del df2['Data']

  df2['Cota'] = df2['Cota'].apply(lambda x: float(x.replace(',', '.')))
  df2['Cota'] = df2['Cota'].astype(float)

  return df2


def load_timeseries02_daily():
  df = load_timeseries02()

  # esse comando jah executa o sort da serie temporal
  df_daily = df.resample('D').ffill()

  return df_daily


def load_timeseries02_monthly():
  df = load_timeseries02()

  # esse comando jah executa o sort da serie temporal
  df_daily = df.resample('D').ffill()

  df_monthly = df_daily.resample('M')
  df_monthly = df_monthly.mean()

  return df_monthly

