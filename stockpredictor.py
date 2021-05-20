import pandas as pd
import PySimpleGUI as sg
import plotly.express as px
import pylab as plt
import plotly.graph_objects as go
import smtplib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import *
import math
from keras.models import *
from keras.layers import *
from datetime import date
from nsepy import get_history
from mpl_finance import candlestick2_ohlc
# import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from nsetools import nse
'''This Function is used to predict the stock prices using sequential model from LSTM.
The data is fetched from a data pipeline built using nsepy.
Start : 20210226'''

def Predictor():
    df = pd.DataFrame(Stock_History)

    data = df.filter(['Close']) #Filtering data for just Close price

    dataset = data.values #change to numpy array

    '''Spercify the length of training Dataset from complete Dataset'''
    training_data_len = math.ceil(len(dataset) * .65)
    print(training_data_len)

    '''Scaling the dataset'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60: i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)
    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    sg.Print(rmse)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    print(valid)
    sg.Print(predictions[-1])

    plt.figure(figsize=(16, 8))
    plt.title(print("Stock price of" + h))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.show()

def DeliveryPerc():
    df = pd.DataFrame(Stock_History)
    data = df['%Deliverble'].apply(lambda x: x*100)
    print(data)


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
    sg.Print(resistance_prices)
    fig, ax = plt.subplots(figsize=[15, 9])
    candlestick2_ohlc(ax, df0['Open'], df0['High'], df0['Low'], df0['Close'], colorup='green', colordown='red',
                      width=1)

    plt.scatter(df_support, support_prices)
    plt.scatter(resistance, resistance_prices)
    plt.show()

'''This Function is used to get all the listed staocks'''
def alltick():
    ewatch = []
    all_stock_codes = nse.get_stock_codes()
    for i,j in all_stock_codes.items():
        watch = ewatch.append(i)
    return ewatch

def MovingAvgScanner(watchlist):
    for name in watchlist:
        try:
            df = get_history(symbol= name , start=date(2020,1,1), end=date(2020,11,6))
            df['9ema'] = df['Close'].ewm(span=13, adjust=False).mean()
            df['21ema'] = df['Close'].ewm(span=49, adjust=False).mean()
            df["9vol"] = df['Volume'].ewm(span=13, adjust=False).mean()
            df['21vol'] = df['Volume'].ewm(span=49, adjust=False).mean()
            df['del'] = df['%Deliverble'].apply(lambda x: x*100)
            # df['ma'] = round(talib.MA(df['Close'], timeperiod=30, matype=0) ,1)
            # df['rsi'] = round(talib.RSI(df['Close'], timeperiod=14) ,1)
            ema21 = df.iloc[-1]['21ema']
            ema9 = df.iloc[-1]['9ema']

            vol21 =  df.iloc[-1]['21vol']
            vol9 = df.iloc[-1]['9vol']

            delivery = df.iloc[-1]['del']
            if ema9 > ema21 and vol9 > vol21:
                print(f"buy {name}")
            elif ema9 < ema21 and vol9 < vol21<50:
                print(f"sell {name}")
            else:
                print(f"Ignore {name}")
        except:
            pass



# sg.theme('DarkAmber')

layout = [[sg.Text('Enter symbol of stock'), sg.InputText()], [sg.Text('Date in format(YYYY,MM,DD)')],
          [sg.Text('From'), sg.InputText()], [sg.Text('TO'), sg.InputText()],
          [sg.Button("Predict"),sg.Button("Delivery"),
           sg.Button("TrendLines"),sg.CloseButton("Cancel")]]

window = sg.Window('Window Title', layout)

while True:
    event, values = window.read()
    if event == 'Predict':
        h = values[0]
        t = (tuple(map(int,(values[1]).split(','))))
        f = (tuple(map(int,(values[2]).split(','))))
        a = t[0]
        b = t[1]
        c = t[2]
        d = f[0]
        e = f[1]
        g = f[2]
        Stock_History = get_history(symbol=h,
                                    start=date(a, b, c),
                                    end=date(d, e, g))
        Predictor()

    elif event == 'Delivery':
        h = values[0]
        t = (tuple(map(int, (values[1]).split(','))))
        f = (tuple(map(int, (values[2]).split(','))))
        a = t[0]
        b = t[1]
        c = t[2]
        d = f[0]
        e = f[1]
        g = f[2]
        Stock_History = get_history(symbol=h,
                                    start=date(a, b, c),
                                    end=date(d, e, g))
        DeliveryPerc()
    elif event == 'TrendLines':
        h = values[0]
        t = (tuple(map(int, (values[1]).split(','))))
        f = (tuple(map(int, (values[2]).split(','))))
        a = t[0]
        b = t[1]
        c = t[2]
        d = f[0]
        e = f[1]
        g = f[2]
        Stock_History = get_history(symbol=h,
                                    start=date(a, b, c),
                                    end=date(d, e, g))
        SuppRes(pd.DataFrame(Stock_History))

    elif event in (None, 'Cancel') or sg.WIN_CLOSED:
        break
window.close()

# Stock_History = get_history(symbol= h,
#                             start=date(a,b,c),
#                             end=date(d,e,g))






