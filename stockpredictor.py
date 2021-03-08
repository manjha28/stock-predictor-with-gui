import pandas as pd
import  PySimpleGUI as sg
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

def Predictor():
    df = pd.DataFrame(Stock_History)

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .65)
    print(training_data_len)

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
    rmse

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

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
sg.theme('DarkAmber')

layout = [[sg.Text('Enter symbol of stock'), sg.InputText()], [sg.Text('Date in format(YYYY,MM,DD)')],
          [sg.Text('From'), sg.InputText()], [sg.Text('TO'), sg.InputText()],
          [sg.Button("Predict"),sg.Button("Delivery"), sg.CloseButton("Cancel")]]

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

    elif event in (None, 'Cancel') or sg.WIN_CLOSED:
        break
window.close()

# Stock_History = get_history(symbol= h,
#                             start=date(a,b,c),
#                             end=date(d,e,g))






