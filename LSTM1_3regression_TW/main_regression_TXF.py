# https://ithelp.ithome.com.tw/articles/10206312

csv_file_name = '20220215_TXFB2_I024.csv'
num_of_epochs = 50
num_of_batch_size = 32
timesteps = 60
days_forward = 20 # predicting how many days forward, 0 being the immediate next
features = ['time_diff','close', 'volume']         # close -> first_match_price
target = ['close']                                          # close -> first_match_price
num_of_features = len(features)
test_size_portion = 0.02
dropout_rate = 0.2




validation_split_portion = 0.1


initial_learning_rate = 0.001
decay_steps= 1 # num_of_epochs * 0.1
decay_rate = 0.6
momentum = 0.8



# ----------------------------------------------------------------------------------------
# Import the libraries
from turtle import shape
import numpy as np
import pandas as pd
import talib
from talib import abstract

# ----------------------------------------------------------------------------------------
# process data
# Import the dataset
dataset_in = pd.read_csv(csv_file_name)

# extract features
X_raw = dataset_in[features]


X_raw['MA5'] = talib.MA(dataset_in['close'], timeperiod = 5)
num_of_features += 1
X_raw['MA10'] = talib.MA(dataset_in['close'], timeperiod = 10)
num_of_features += 1
X_raw['MA20'] = talib.MA(dataset_in['close'], timeperiod = 20)
num_of_features += 1

MACDs = eval('abstract.'+'MACD'+'(dataset_in)')
X_raw['macd'] = MACDs['macd']
X_raw['macdsignal'] = MACDs['macdsignal']
X_raw['macdhist'] = MACDs['macdhist']
num_of_features += 3

# RSI = eval('abstract.'+'RSI'+'(dataset_in)')
# X_raw['rsi'] = RSI
# num_of_features += 1

OBV = eval('abstract.'+'OBV'+'(dataset_in)')
X_raw['obv'] = OBV
num_of_features += 1

# ADX = eval('abstract.'+'ADX'+'(dataset_in)')
# X_raw['adx'] = ADX
# num_of_features += 1




X_raw = X_raw[33:]
# X_raw = X_raw.dropna()




# extract targets
y_raw = X_raw[target]

# calculate total num of data
total_num_data = len(X_raw)


# split training / testing
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size = test_size_portion, shuffle = False)


# scaling
from sklearn.preprocessing import MinMaxScaler
scale_X_train = MinMaxScaler(feature_range = (0, 1))
X_train_scale = scale_X_train.fit_transform(X_train_raw)

scale_y_train = MinMaxScaler(feature_range = (0, 1))
y_train_scale = scale_y_train.fit_transform(y_train_raw)

scale_X_test = MinMaxScaler(feature_range = (0, 1))
X_test_scale = scale_X_test.fit_transform(X_test_raw)

scale_y_test = MinMaxScaler(feature_range = (0, 1))
y_test_scale = scale_y_test.fit_transform(y_test_raw)


# generate epochs
X_train = []   #預測點的前 timesteps 天的資料
y_train = []   #預測點
for i in range(timesteps, len(X_train_scale) - days_forward):
    X_train.append( X_train_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_train.append( y_train_scale [ (i + days_forward) , 0] ) # data of the target value

X_test = []
y_test = []
for i in range(timesteps, len(X_test_scale) - days_forward):
    X_test.append( X_test_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_test.append( y_test_scale [ (i + days_forward) , 0] ) # data of the target value


# convert to numpy array
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)   # 轉成numpy array的格式，以利輸入 RNN


# reshape data X to 3-dimension for model
assert num_of_features == X_train.shape[2]
X_train = np.reshape( X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2] ) )

assert num_of_features == X_test.shape[2]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))




# ----------------------------------------------------------------------------------------
# setup model
# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization


# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
assert num_of_features == X_train.shape[2]
model.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(dropout_rate))

# Adding a second LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(dropout_rate))

# Adding a third LSTM layer and some Dropout regularisation
# model.add(LSTM(units = 50, return_sequences = True))
# model.add(Dropout(dropout_rate))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 128))
model.add(Dropout(dropout_rate))

# Adding the output layer
model.add(Dense(units = 1))



import keras.backend as K
import tensorflow as tf
from keras import metrics
import sys

def custo_loss(y_true, y_pred):
    # https://stackoverflow.com/questions/46876213/custom-mean-directional-accuracy-loss-function-in-keras
    
    value_diff = tf.abs( tf.subtract(y_true, y_pred) )
    
    y_true = tf.concat([[[0]], y_true], 0)
    y_pred = tf.concat([[[0]], y_pred], 0)
    
    sign_diff = tf.cast( tf.not_equal(tf.sign(y_true[1:] - y_true[:-1]), tf.sign(y_pred[1:] - y_pred[:-1])) , tf.float32 )
    sign_diff = tf.add(sign_diff, 1)
    
    return tf.multiply(sign_diff, value_diff)
    
    
    # didn't work
    # # https://towardsdatascience.com/customize-loss-function-to-make-lstm-model-more-applicable-in-stock-price-prediction-b1c50e50b16c


def custo_acc(y_true, y_pred):
    within = 0
    for i in range(0, len(y_true)):
        if ( abs(y_pred[i] - y_true[i])/y_true[i] < acc_threshold ):
            within += 1
    return within/len(y_true)


from keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers

# learning rate
# lr_schedule = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps,
#     decay_rate,
#     staircase=True)

# sgd = optimizers.SGD(lr = initial_learning_rate, momentum = momentum, decay = decay_rate, nesterov = False)


# loss_method = 'mean_squared_error'
# loss_method_str = 'mean_squared_error'

loss_method = custo_loss
loss_method_str = 'custo_loss'
val_loss_method_str = 'val_' + loss_method_str



# acc_method = 'acc'
# acc_method_str = 'acc'

acc_method = custo_acc
acc_method_str = 'custo_acc'
acc_threshold = 0.01
val_acc_method_str = 'val_' + acc_method_str


# Compiling

model.compile(optimizer = 'adam', loss = loss_method, metrics=[acc_method])

# model.compile(optimizer = 'sgd', loss = loss_method, metrics=[acc_method])

# model.compile(optimizer = optimizers.Adam(learning_rate = lr_schedule), loss = loss_method, metrics=[acc_method])


# model.compile(optimizer = sgd, loss = loss_method, metrics=[acc_method])




# ----------------------------------------------------------------------------------------
# checkpoint / learning rate / callbacks set up
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor = val_acc_method_str, 
    mode = 'max', # {'auto', 'min', 'max'}
    save_best_only = True, # False: also save the model itself (structure)
    verbose = 1 # 0: silent, 1: displays messages when the callback takes an action
    )
callbacks_list = [checkpoint]


# ----------------------------------------------------------------------------------------
# train model
history = model.fit(X_train, y_train, validation_split = validation_split_portion, shuffle = True, epochs = num_of_epochs, batch_size = num_of_batch_size, callbacks = callbacks_list)


# ----------------------------------------------------------------------------------------
# load best model
model.load_weights("weights.best.hdf5")
results = model.evaluate(X_test, y_test)

# save the whole model
model.save("model_whole")

# predict
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scale_y_test.inverse_transform(predicted_stock_price)  # to get the original scale


# ----------------------------------------------------------------------------------------
# shift right timesteps

predicted_stock_price_shifted = []
for i in range(0, timesteps):
    predicted_stock_price_shifted.append(None)
for i in predicted_stock_price:
    predicted_stock_price_shifted.append(i[0])


# ----------------------------------------------------------------------------------------
# visualize 

import matplotlib.pyplot as plt  # for ploting results


plot_loss = plt.subplot2grid((2, 2), (0, 0))                #, colspan=2)
plot_accu = plt.subplot2grid((2, 2), (0, 1))                #, rowspan=3, colspan=2)
plot_test = plt.subplot2grid((2, 2), (1, 0), colspan=2)     #, rowspan=2)

# Visualising the loss
plot_loss.plot(history.history['loss'])
plot_loss.plot(history.history['val_loss'])
plot_loss.set_title('model loss')
plot_loss.set_ylabel('loss')
plot_loss.set_xlabel('epoch')
plot_loss.legend(['Train', 'Validation'], loc='upper left')


# # Visualising the accuracy
# plot_accu.plot(history.history['acc'])
# plot_accu.plot(history.history['val_acc'])
# plot_accu.set_title('model acc (==, built-in)')
# plot_accu.set_ylabel('accuracy')
# plot_accu.set_xlabel('epoch')
# plot_accu.legend(['Train', 'Validation'], loc='upper left')


# Visualising the accuracy
plot_accu.plot(history.history[acc_method_str])
plot_accu.plot(history.history[val_acc_method_str])
plot_accu.set_title('model acc (threshold: ' + str(acc_threshold) + ')')
plot_accu.set_ylabel('accuracy')
plot_accu.set_xlabel('epoch')
plot_accu.legend(['Train', 'Validation'], loc='upper left')


# Visualising the test results
real_stock_price = scale_y_test.inverse_transform(y_test_scale)
plot_test.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plot_test.plot(predicted_stock_price_shifted, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plot_test.set_title('Google Stock Price Prediction')
plot_test.set_xlabel('Time', loc='left')
plot_test.set_ylabel('Google Stock Price')
plot_test.legend()

# turn features to 1 string
feature_all = ''
for i in features:
    feature_all = feature_all + i + ", "

# Packing all the plots and displaying them
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps) \
    + " , Days forward: " + str(days_forward) + " , Test size: " + str(test_size_portion) + " , Dropout rate: " + str(dropout_rate) \
    + "\nFeatures: " + feature_all + "Target: " + target[0]
plt.figtext(0.9, 0.01, configuration, horizontalalignment = 'right', verticalalignment = 'bottom', wrap = True, fontsize = 12)
plt.tight_layout()
plt.show()