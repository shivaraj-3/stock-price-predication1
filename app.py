from flask import Flask,request, url_for, redirect, render_template
import pickle
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import  numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from numpy import array
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
def tableString(arr):
    reS="<table> <thead> <tr><th>Days</th><th>Cost</th></tr> </thead> <tbody>"
    i=1;
    for val in arr:
        reS+="<tr><td>"+(str(i))+"</td><td>"+(str(val[0]))+"</td></tr>";
        i+=1;
    reS+="</tbody></table>";
    return reS;
def table1String(arr):
    res="";
    i=1;
    for val in arr:
        res+="Day-"+str(i)+"      price: "+(str(val[0]))+"\n";

        i+=1
    return res;

@app.route('/')
def hello_world():
    # return render_template("forest.html")
    return render_template("stock.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    df = pd.read_csv('AAPL.csv')
    # df1 contain only close
    df1 = df.reset_index()['close']

    # ploting the close price
    # plt.plot(df1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    # reshape means single column

    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # data preprocessing
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=10, batch_size=64, verbose=1)

    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    ##Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    ### Calculate RMSE performance metrics
    math.sqrt(mean_squared_error(y_train, train_predict))

    ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest, test_predict))

    ### Plotting
    # shift train predictions for plotting
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict
    # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(df1))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()

    x_input = test_data[341:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # demonstrate prediction for next 30 days

    lst_output = []
    n_steps = 100
    i = 0
    while (i < 30):
        if (len(temp_input) > 100):
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    final_result=scaler.inverse_transform(lst_output)

    return render_template('stock.html',
                           pred=tableString(final_result),
                           bhai="kuch karna hain iska ab?")



if __name__ == '__main__':
    app.run(debug=True)
