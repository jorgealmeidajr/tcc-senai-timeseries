
import pandas as pd

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt


def check_stationarity(timeseries, window):
    #Determing rolling statistics
    print(window)
    rolling_mean = timeseries.rolling(window=window, center=False).mean() 
    rolling_std = timeseries.rolling(window=window, center=False).std()

    #Plot rolling statistics:
    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')
    mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')

    dickey_fuller_test = adfuller(timeseries.iloc[:,0].values, autolag='AIC')
    dfresults = pd.Series(dickey_fuller_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)'%key] = value
    
    print (dfresults)

