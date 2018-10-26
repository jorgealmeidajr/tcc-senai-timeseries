
import pandas as pd


def load_timeseries02():
  df_original = pd.read_csv('../data/fundo01-cotas-rendafixa.csv', header=0, encoding='iso-8859-1')

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

