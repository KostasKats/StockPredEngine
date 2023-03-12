import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from utils.PlotUtils import plotFutureSteps, plotCurrentStatus


def loadData(ticker,period):
    return yf.download(ticker, period=period)
def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y

def predict(id):
    ticker = id
    df = loadData(ticker,'2y')

    df.shape
    df = df['Open'].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0] * 0.8)])
    dataset_test = np.array(df[int(df.shape[0] * 0.8):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=20, batch_size=32)
    # model.save('stock_prediction.h5')
    #
    # model = load_model('stock_prediction.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # make predictions for future time steps
    future_predictions = []
    last_sequence = x_test[-1]
    look_back = 50
    for i in range(5):
        # predict next value
        next_prediction = model.predict(last_sequence.reshape(1, look_back, 1))[0][0]
        # add to predictions array
        future_predictions.append(next_prediction)
        # update input sequence
        last_sequence = np.concatenate((last_sequence[1:], [[next_prediction]]), axis=0)

    # invert scaling of predictions
    test_predictions = scaler.inverse_transform(predictions)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    plotFutureSteps(y_test_scaled,predictions, future_predictions,ticker,'Open')


if __name__ == "__main__":
    # MYTIL.AT
    # AAPL
    # BELA.AT
    predict('MYTIL.AT')
    # id= "MYTIL.AT"
    # df = loadData(id,'1y')
    # last_30_days = df.tail(30)
    # plotCurrentStatus(last_30_days,id)





