import plotly.graph_objects as go
import pandas as pd
import numpy as np
from mpl_finance import candlestick2_ohlc
# import matplotlib.pyplot as plt
import pylab as plt
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from datetime import date
from nsepy import get_history
Stock_History = get_history(symbol="SBIN", start=date(2015,1,1), end=date(2015,1,31))
df = pd.DataFrame(Stock_History)

def macd(df):
    # df0 = df.set_index(pd.DatetimeIndex(df['Date'].values))
    shortEMA = df.Close.ewm(span=12,adjust=False).mean()
    longEMA = df.Close.ewm(span=26,adjust=False).mean()
    MACD = shortEMA-longEMA
    signal = MACD.ewm(span=9,adjust=False).mean()
    df['MACD'] = MACD
    df['Signal'] = signal
    df
    # df.insert(MACD,signal)

def buy_sell(signal):
    macd(signal)
    Buy = []
    Sell = []
    flag = -1
    for i in range(0,len(signal)):
        if signal['MACD'][i] > signal['Signal Line'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal['Close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal['MACD'][i] < signal['Signal Line'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal['Close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy,Sell)
if __name__ == '__main__':
    a = buy_sell(df)
    df['Buy_SIgnal'] = a[0]
    df['sell_signal'] = a[1]

    plt.scatter(df.index,df['Buy_SIgnal'],color = 'green',label = 'Buy',marker = '^',alpha=1)
    plt.scatter(df.index,df['sell_signal'],color = 'red',label = 'Sell',marker = 'v',alpha=1)

    plt.plot(df['Close'], alpha = 0.35)
    plt.show()