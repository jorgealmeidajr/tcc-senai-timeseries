
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt



def check_stationarity(timeseries, window):
    # Determing rolling statistics
    rolling_mean = timeseries.rolling(window=window, center=False).mean() 
    rolling_std = timeseries.rolling(window=window, center=False).std()

    # Plot rolling statistics
    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
    mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label = 'Rolling Std')
    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    dickey_fuller_test(timeseries)
    

# function to perform Dickey-Fuller test
def dickey_fuller_test(timeseries):
    print('Results of Dickey-Fuller Test:')

    dickey_fuller_test = adfuller(timeseries.iloc[:,0].values, autolag='AIC')

    dfresults = pd.Series(
        dickey_fuller_test[0:4], 
        index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    for key, value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)' % key] = value
    
    #print(dfresults)
    #print()
    print_test_results(dfresults)


def print_test_results(dfresults):
    test_statistic = None
    p_value = None
    critical_value_1 = None
    critical_value_5 = None
    critical_value_10 = None

    for key, value in dfresults.iteritems():
        if key == 'p-value':
            p_value = value
            continue
        
        if key == 'Test Statistic':
            test_statistic = value
            continue
        
        if key == 'Critical Value (1%)':
            critical_value_1 = value
            continue

        if key == 'Critical Value (5%)':
            critical_value_5 = value
            continue

        if key == 'Critical Value (10%)':
            critical_value_10 = value
            continue

    if test_statistic > critical_value_1:
        print('[FALHA] Test Statistic (%f) > Critical Value 1 (%f)' % (test_statistic, critical_value_1))
    else:
        print('[SUCESSO] Test Statistic (%f) < Critical Value 1 (%f)' % (test_statistic, critical_value_1))

    if test_statistic > critical_value_5:
        print('[FALHA] Test Statistic (%f) > Critical Value 5 (%f)' % (test_statistic, critical_value_5))
    else:
        print('[SUCESSO] Test Statistic (%f) < Critical Value 5 (%f)' % (test_statistic, critical_value_5))

    if test_statistic > critical_value_10:
        print('[FALHA] Test Statistic (%f) > Critical Value 10 (%f)' % (test_statistic, critical_value_10))
    else:
        print('[SUCESSO] Test Statistic (%f) < Critical Value 10 (%f)' % (test_statistic, critical_value_10))

    if p_value > 0.05:
        print('[FALHA] p-value (%f) > 0.05' % (p_value))
    else:
        print('[SUCESSO] p-value (%f) < 0.05' % (p_value))


from typing import TypeVar

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

def plot_timeserie(df: PandasDataFrame):
    df.plot(figsize=(15, 12), linewidth=2, fontsize=14)
    plt.grid(True)
    plt.xlabel('Ano', fontsize=14)


def df_has_any_null(df: PandasDataFrame):
    if df.isnull().values.any():
        print('[FALHA] Existe valores nao definidos no dataframe')
    else:
        print('[SUCESSO] Todos os valores estao definidos no dataframe')


# algoritmo para transformacao de serie temporal para estacionaria
def ts_transform1(df: PandasDataFrame):
    transform = {'original': df}

    # Apply a nonlinear log transformation
    ts_log = np.log(df)
    transform['ts_log'] = ts_log
    
    # Remove trend and seasonality with differencing
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
    transform['ts_log_diff'] = ts_log_diff
    
    return transform


def plot_train_and_test(train: PandasDataFrame, test: PandasDataFrame):
    plt.plot(train, label='train', color='blue')
    plt.plot(test, label='test', color='red')
    plt.legend(loc='upper left')


def plot_train_and_arima(ts_train: PandasDataFrame, results_ARIMA):
    if len(ts_train) != len(results_ARIMA.fittedvalues):
        raise Exception('Os dois parametros precisam ter o mesmo tamanho')
    
    print('MSE: %.9f' % mean_squared_error(ts_train.values, results_ARIMA.fittedvalues.values))

    plt.plot(ts_train.index.to_pydatetime(), ts_train.values)
    plt.plot(ts_train.index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')
    #plt.title('MSE: %.9f' % mean_squared_error(ts_train.values, results_ARIMA.fittedvalues.values))


def plot_train_and_predictions(ts_train: PandasDataFrame, predictions_ARIMA):
    if len(ts_train) != len(predictions_ARIMA.values):
        raise Exception('Os dois parametros precisam ter o mesmo tamanho')

    print('MSE: %.9f' % mean_squared_error(ts_train.values, predictions_ARIMA.values))

    plt.plot(ts_train.index.to_pydatetime(), ts_train.values)
    plt.plot(ts_train.index.to_pydatetime(), predictions_ARIMA.values, color='red')
    #plt.title('MSE: %.9f' % mean_squared_error(ts_train.values, predictions_ARIMA.values))

