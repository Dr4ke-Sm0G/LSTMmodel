# First we will import the necessary Library
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import io
import tensorflow as tf

# For Evaluation we will use these library
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras import Sequential
from keras import Adam
from keras import Dense, LSTM, LeakyReLU, Dropout

# Load our dataset
maindf = pd.read_csv(io.BytesIO(uploaded['BTC-USD.csv']))
print('Total number of days present in the dataset: ', maindf.shape[0])
print('Total number of fields present in the dataset: ', maindf.shape[1])

# Check null values
print('Null Values:', maindf.isnull().values.sum())
print('NA values:', maindf.isnull().values.any())

# Printing the start date and End date of the dataset
sd = maindf.iloc[0][0]
ed = maindf.iloc[-1][0]
print('Starting Date', sd)
print('Ending Date', ed)

# Convert Date to datetime for easier manipulation
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

# First Step is Preparing Data for Training and Testing
# Here we are just considering 1 year data for training data
# Lets First Take all the Close Price
closedf = maindf[['Date', 'Close']]
print("Shape of close dataframe:", closedf.shape)

closedf = closedf[closedf['Date'] > '2022-05-21']
close_stock = closedf.copy()
print("Total data for prediction: ", closedf.shape[0])

# Normalizing Data
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
print(closedf.shape)

# we keep the training set as 60% and 40% testing set
training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size,
                                :], closedf[training_size:len(closedf), :1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# Function to create dataset with time step


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

# Building the LSTM model
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Training the model (you might need to uncomment and adjust this part)
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
print("Train data RMSE: ", math.sqrt(
    mean_squared_error(original_ytrain, train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(
    mean_squared_error(original_ytest, test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))

# Additional metrics
print("Train data explained variance regression score:",
      explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:",
      explained_variance_score(original_ytest, test_predict))
print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))
print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))

# Future prediction setup (you might need to adjust or uncomment this part)
# x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
# temp_input = list(x_input)[0]
# lst_output = []
# n_steps = time_step
# pred_days = 30
# for i in range(pred_days):
#     x_input = np.array(temp_input[1:])
#     x_input = x_input.reshape(1, n_steps, 1)
#     yhat = model.predict(x_input, verbose=0)
#     temp_input.extend(yhat[0].tolist())
#     temp_input = temp_input[1:]
#     lst_output.extend(yhat.tolist())

# print("Output of predicted next days: ", len(lst_output))
