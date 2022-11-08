# https://ithelp.ithome.com.tw/articles/10206312

'''
agenda

save best model => the highest val acc / the lowest val loss

optimizer
learning rate decay / scheduling

done
--------------------------------
not yet

optimizer
adam w -> momentum (may not need schedular)

Customize acc
https://keras.io/guides/training_with_built_in_methods/#custom-metrics

early stoping
it is based on acc

Feature (加 delta)



combine models (Functional API )
Stationary (arima)
https://zh.wikipedia.org/zh-tw/時間序列


'''


csv_file_name = 'GOOG.csv'
num_of_epochs = 5
num_of_batch_size = 64
timesteps = 60
days_forward = 0 # predicting how many days forward, 0 being the immediate next
features = ['open','high', 'low', 'close', 'volume']
target = ['close']
num_of_features = len(features)
test_size_portion = 0.1
validation_split_portion = 0.1
dropout_rate = 0.1
num_of_label = 3 # 0:down, 1:flat, 2:up
margin_rate = 0.01

initial_learning_rate = 0.0001
decay_steps = 1280
decay_rate = 0.96

# ----------------------------------------------------------------------------------------
# Import the libraries
from turtle import shape
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# process data
# Import the dataset
dataset_in = pd.read_csv(csv_file_name)

import talib

# extract features
X_raw = dataset_in[features]
X_raw['MA20'] = talib.MA(dataset_in['close'], timeperiod = 20)
X_raw.iloc[:20,5] = X_raw.iloc[:20,3] 
num_of_features += 1

X_raw['MA5'] = talib.MA(dataset_in['close'], timeperiod = 5)
X_raw.iloc[:5,6] = X_raw.iloc[:5,3] 
num_of_features += 1

print(X_raw)
X_raw.to_csv('testout.csv')
# extract targets
close = dataset_in['close'].values.tolist()
open = dataset_in['open'].values.tolist()

y_raw = [0]
for i in range(1, len(open)):
    diff = close[i] - close[i-1]
    margin = close[i-1] * margin_rate
    if diff > margin :
        y_raw.append(2)
    elif diff < - margin :
        y_raw.append(0)
    else:
        y_raw.append(1)


print("len of open: ", len(open))

# calculate total num of data
total_num_data = len(X_raw)


# split training / testing
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size = test_size_portion, shuffle = False)


# scaling
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
scale_X_train = MinMaxScaler(feature_range = (0, 1))
X_train_scale = scale_X_train.fit_transform(X_train_raw)

y_train_label = to_categorical(y_train_raw)

scale_X_test = MinMaxScaler(feature_range = (0, 1))
X_test_scale = scale_X_test.fit_transform(X_test_raw)

y_test_label = to_categorical(y_test_raw)

print("len of y test label", len(y_test_label))


# generate epochs
X_train = []   #預測點的前 timesteps 天的資料
y_train = []   #預測點
for i in range(timesteps, len(X_train_scale) - days_forward):
    X_train.append( X_train_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_train.append( y_train_label [ (i + days_forward) ] )                                                                                              # data of the to_categorical

X_test = []
y_test = []
for i in range(timesteps, len(X_test_scale) - days_forward):
    X_test.append( X_test_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_test.append( y_test_label [ (i + days_forward) ] )                                                                                                # data of the to_categorical

print("len of y_test: ", len(y_test))

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
#from keras.layers import BatchNormalization


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
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(dropout_rate))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 128))
model.add(Dropout(dropout_rate))

# Adding the output layer
##model.add(Dense(units = 1))
model.add(Dense(3, activation='softmax'))



from keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers

#lr_schedule = LearningRateScheduler(lr)
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=True)


# Compiling
model.compile(optimizer = optimizers.Adam(learning_rate = lr_schedule), loss = 'categorical_crossentropy', metrics = ['acc'])


# ----------------------------------------------------------------------------------------
# checkpoint / learning rate / callbacks set up
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor = 'val_loss', 
    mode = 'min', # {'auto', 'min', 'max'}
    save_best_only = True, # False: also save the model itself (structure)
    verbose = 1 # 0: silent, 1: displays messages when the callback takes an action
    )
callbacks_list = [checkpoint]


# ----------------------------------------------------------------------------------------
# run model
history = model.fit(X_train, y_train, validation_split = validation_split_portion, shuffle = False, epochs = num_of_epochs, batch_size = num_of_batch_size, callbacks = callbacks_list)


# ----------------------------------------------------------------------------------------
### predict
predicted_y_test = model.predict(X_test)
predicted_y_test = np.argmax(predicted_y_test, axis = 1)[:]   # to get the original scale

# load best model
model.load_weights("weights.best.hdf5")

results = model.evaluate(X_test, y_test)



# ----------------------------------------------------------------------------------------
# shift right timesteps
'''
predicted_y_test_shifted = []
for i in range(0, timesteps):
    predicted_y_test_shifted.append(None)
for i in predicted_y_test:
    predicted_y_test_shifted.append(i)
'''



data = [[0,0,0],
        [0,0,0],
        [0,0,0]]


for i in range(0, len(predicted_y_test)):
    data[ abs(y_test_raw[i]-2) ] [ predicted_y_test[i] ] += 1




# ----------------------------------------------------------------------------------------
# visualize 

import matplotlib.pyplot as plt  # for ploting results


plot_loss = plt.subplot2grid((2, 2), (0, 0))
plot_accu = plt.subplot2grid((2, 2), (0, 1))
plot_test = plt.subplot2grid((2, 2), (1, 0))

# Visualising the loss
plot_loss.plot(history.history['loss'])
plot_loss.plot(history.history['val_loss'])
plot_loss.set_title('model loss')
plot_loss.set_ylabel('loss')
plot_loss.set_xlabel('epoch')
plot_loss.legend(['Train', 'Validation'], loc='upper left')


# Visualising the accuracy
plot_accu.plot(history.history['acc'])
plot_accu.plot(history.history['val_acc'])
plot_accu.set_title('model accuracy')
plot_accu.set_ylabel('accuracy')
plot_accu.set_xlabel('epoch')
plot_accu.legend(['Train', 'Validation'], loc='upper left')



test_statistics = "test loss: " + str(results[0]) + "\ntest acc:" + str(results[1]) + "\nMargin rate: " + str(margin_rate)

plot_test.axis('tight')
plot_test.axis('off')
plot_test.table(cellText=data,colLabels=['pred down', 'pred flat', 'pred up'], rowLabels=['real up', 'real flat', 'real down'], loc="center")


plt.figtext(0.9, 0.15, test_statistics, horizontalalignment = 'right', verticalalignment = 'bottom', wrap = True, fontsize = 12)





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

