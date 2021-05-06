''' Author : Manish Jha
    Start Date : 2021,04,01
    '''
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

# def best_fit(X, Y):
#
#     xbar = sum(X)/len(X)
#     ybar = sum(Y)/len(Y)
#     n = len(X) # or len(Y)
#
#     numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
#     denum = sum([xi**2 for xi in X]) - n * xbar**2
#
#     b = numer / denum
#     a = ybar - b * xbar
#
#     print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
#
#     return a, b

def macd(df):
    # df0 = df.set_index(pd.DatetimeIndex(df['Date'].values))
    shortEMA = df.Close.ewm(span=12,adjust=False).mean()
    longEMA = df.Close.ewm(span=26,adjust=False).mean()
    MACD = shortEMA-longEMA
    signal = MACD.ewm(span=9,adjust=False).mean()
    # df['MACD'] = MACD
    # df['Signal'] = signal
    # BUy = []
    # Sell = []
    # flag = -1
    # for i in range(0,len(signal)):
    #     if df['MACD'][i] > df['signal'][i]:
    #         Sell.append(np.nan)
    #         if flag != 1:
    #             BUy.append(df['Close'][i])
    #             flag = 1
    #         else:
    #             BUy.append(np.nan)
    #
    #     elif df['MACD'][i] < df['signal'][i]:
    #         BUy.append(np.nan)
    #         if flag != 0:
    #             Sell.append(df['Close'][i])
    #             flag = 0
    #         else:
    #             Sell.append(np.nan)
    #     else:
    #         BUy.append(np.nan)
    #         Sell.append(np.nan)
    # return (BUy,Sell)
    # plt.scatter(df.index)
    plt.subplot(2, 1, 1)
    plt.plot(df.index,df['Close'],label = 'Close', color = 'green')
    plt.xticks(rotation=45)
    # fig = go.Figure(data=[go.Candlestick(x=df.index,
    #                                      open=df['Open'],
    #                                      high=df['High'],
    #                                      low=df['Low'],
    #                                      close=df['Close'])])
    # fig.show()
    plt.subplot(2,1,2)
    plt.plot(df.index, MACD, label='MACD', color='red')
    plt.plot(df.index,signal,label = 'Signal Line',color = 'blue')
    plt.xticks(rotation = 45)
    plt.legend(loc = 'upper left')
    # plt.xticks(df['Close'], df.index)
    #
    # plt.gca().set_ylim(ymin=0)
    # plt.xticks(df['Close'], df.index)
    plt.show()



def SuppRes(df0):
    df_open = df0.Open.copy()
    df_high = df0.High.copy()
    df_low = df0.Low.copy()
    df_close = df0.Close.copy()

    df_support = argrelmin(df_low.values, order=5)
    support_prices = df_low[df_support[0]]
    support_prices_lower = df_open[df_support[0]]

    resistance = argrelmax(df_high.values, order=5)
    resistance_prices = df_high[resistance[0]]
    resistance_prices_higher = df_open[resistance[0]]

    print('Support prices', support_prices)
    print('Support:', df_support)
    print('Resistance:', resistance)
    resistance_prices
    fig, ax = plt.subplots(figsize=[15, 9])
    candlestick2_ohlc(ax, df0['Open'], df0['High'], df0['Low'], df0['Close'], colorup='green', colordown='red',
                      width=0.5)

    plt.scatter(df_support, support_prices)
    plt.scatter(resistance, resistance_prices)
    # plt.plot(np.unique(df_support), np.poly1d(np.polyfit(df_support, support_prices, 1))(np.unique(df_support)))
    plt.show()

    # for i in range(len(df_support)):
    #     plt.plot((0, len(support_prices)), (df_support[i], resistance[i]), 'r--')
macd(df)