
import re
import pandas as pd
import numpy as np


def load_timeseries01_original():
  return pd.read_csv('../data/us-dollar-vs-brazilian-real-rate.csv', header=0, names=['date', 'rate'])


def load_timeseries01_missing_data():
  file = open('../data/us-dollar-to-real-rate-20181104.html', 'r') 

  result = []
  valor: str = ''
  data: str = ''

  for line in file.readlines():    
    line = line.strip()
    
    m = re.search(r'1\s+USD\s+=\s+(\d+.*)\s+BRL', line)
    if m:
      #print(line)
      #print('Match found: ', m.group(1))
      valor = m.group(1)
      continue
        
    m = re.search(r'USD\s+BRL\s+rate\s+for\s+(\d+/\d+/\d+)', line)
    if m:
      data = m.group(1)
      
      if data == '21/09/2018':
        break

      result.append({'date': data, 'rate': valor})
      valor = ''
      data = ''
      
      continue

  return result


def load_timeseries01_daily():
  df_original = load_timeseries01_original()
  df = df_original.copy()

  df['date'] = pd.to_datetime(df['date'])
  df = df.set_index('date')

  idx = pd.date_range('01-02-1995', '09-21-2018')
  df = df.reindex(idx, fill_value=0)
  df = df.replace(0, np.nan)
  df = df.replace('.', np.nan)
  df = df.fillna(method='ffill')

  df.rate = pd.to_numeric(df.rate)

  df = df['1999-09-21':]
  return df


def load_timeseries01_monthly():
  df = load_timeseries01_daily()
  df = df.resample('M')
  df = df.mean()
  return df


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

