
import pandas as pd

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt


def check_stationarity(timeseries, window):
    # Determing rolling statistics
    rolling_mean = timeseries.rolling(window=window, center=False).mean() 
    rolling_std = timeseries.rolling(window=window, center=False).std()

    # Plot rolling statistics
    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')
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
    
    print(dfresults)
    print()
    print_test_results(dfresults)


def print_test_results(dfresults):
    is_stationary = True
    test_statistic = None
    p_value = None
    critical_value_5 = None

    for key, value in dfresults.iteritems():
        if key == 'p-value':
            p_value = value
            continue
        
        if key == 'Test Statistic':
            test_statistic = value
            continue
        
        if key == 'Critical Value (5%)':
            critical_value_5 = value
            continue
    
    if test_statistic > critical_value_5:
        print('test_statistic (%f) > critical_value_5 (%f)' % (test_statistic, critical_value_5))
        is_stationary = False

    if p_value > 0.05:
        print('p_value (%f) > 0.05' % (p_value))
        is_stationary = False

    if is_stationary:
        print('the time serie is stationary')
    else:
        print('the time serie is NOT stationary')

