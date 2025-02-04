import keras
import yfinance as yf
import numpy as np
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime

from utils.PlotUtils import savePlotFuture


def loadData(ticker, period):
    return yf.download(ticker, period=period)

def create_dataset(df, steps):
    x = []
    y = []
    for i in range(steps, df.shape[0]):
        x.append(df[i-steps:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y

def predict(id, type, days, steps):
    print(f"********************* Starting prediction for: {id}, days: {days}, steps: {steps} *********************")
    ticker = id
    df = loadData(ticker, '4y')

    df = df[type].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0] * 0.7)])  # 70% training
    dataset_val = np.array(df[int(df.shape[0] * 0.7):int(df.shape[0] * 0.9)])  # 20% validation
    dataset_test = np.array(df[int(df.shape[0] * 0.9):])  # 10% testing

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_val = scaler.transform(dataset_val)
    dataset_test = scaler.transform(dataset_test)

    x_train, y_train = create_dataset(dataset_train, steps)
    x_val, y_val = create_dataset(dataset_val, steps)
    x_test, y_test = create_dataset(dataset_test, steps)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = keras.Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(x_train.shape[1], 1)))
    model.add(Bidirectional(LSTM(100, return_sequences=False, kernel_regularizer=l2(0.01))))
    model.add(Dense(25))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True
    )


    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    future_predictions = []
    last_sequence = x_test[-1]
    for i in range(days):
        next_prediction = model.predict(last_sequence.reshape(1, steps, 1))[0][0]
        future_predictions.append(next_prediction)
        last_sequence = np.concatenate((last_sequence[1:], [[next_prediction]]), axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    savePlotFuture(y_test_scaled, predictions, future_predictions, ticker, type)

if __name__ == "__main__":
    # plotCurrentStatus(loadData('MYTIL.AT','1mo').tail(30),'MYTIL')
    # MYTIL.AT
    # AAPL
    # BELA.AT
    # GEKTERNA.AT
    # AEGN.AT
    # LAMDA.AT
    # DEI
    predict('EUROB.AT','Close',60, 50)
    # df = loadData(id,'1mo')
    # last_30_days = df.tail(30)
    # plotCurrentStatus(df,id)







