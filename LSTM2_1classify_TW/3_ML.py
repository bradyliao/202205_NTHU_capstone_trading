# agenda

# learning rate

# early stoping

# Feature (加 delta)

# combine models (Functional API )
# Stationary (arima)
# https://zh.wikipedia.org/zh-tw/時間序列

# optimizer
# adam w -> momentum (may not need schedular)


import numpy as np
import pandas as pd

date_product_ = '20220530_TXFF2_'
csv_file_name = date_product_ + '2processed.csv'

dataset_in = pd.read_csv(csv_file_name)

# testing (very few data)
# dataset_in = dataset_in[:][:500]


# for reference
# column_header = [ '3th_buy_qty', '3th_buy_price', '2th_buy_qty', '2th_buy_price', '1th_buy_qty', '1th_buy_price', 'future_sell_to_now_buy_trend', 'future_buy_trend', 'time_diff', 'future_sell_trend', 'future_buy_to_now_sell_trend', '1th_sell_price', '1th_sell_qty', '2th_sell_price', '2th_sell_qty', '3th_sell_price', '3th_sell_qty', '1th_buy_price_MACDhist', '1th_sell_price_MACDhist' ]


features = [ '3th_buy_qty', '3th_buy_price', '2th_buy_qty', '2th_buy_price', '1th_buy_qty', '1th_buy_price', 'time_diff', '1th_sell_price', '1th_sell_qty', '2th_sell_price', '2th_sell_qty', '3th_sell_price', '3th_sell_qty', '1th_buy_price_MACDhist', '1th_sell_price_MACDhist' ]
# targets = [ 'future_sell_to_now_buy_trend', 'future_buy_trend', 'future_sell_trend', 'future_buy_to_now_sell_trend']
targets = ['future_buy_to_now_sell_trend']


# extract features ---------------------------------------------------------------------------
raw_x = dataset_in[features]
# extract targets
raw_y = dataset_in[targets]

print("raw_x.shape: ", raw_x.shape)
print("raw_y.shape: ", raw_y.shape)


# scale features ---------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(raw_x)

scaled_x = scaler_x.transform(raw_x)
# scaled_x -> numpy.ndarray

# scale targets ---------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
label_in_order = [ ['flat_or_down'], ['up'] ]
y_Encoder =  OneHotEncoder(sparse=False)
y_Encoder.fit(label_in_order)

scaled_y =  y_Encoder.transform(raw_y) 
# scaled_y -> numpy.ndarray     [ [1. 0.] , [0. 1.], ... ]
#                            flat_or_down , up

print("scaled_x.shape: ", scaled_x.shape)
print("scaled_y.shape: ", scaled_y.shape)



# pack data ---------------------------------------------------------------------------
lookback = 30
packed_x = []
packed_y = []
for i in range(lookback, len(scaled_y) ):
    packed_x.append( scaled_x[i-lookback:i+1] )
    packed_y.append( scaled_y[i] )

packed_x = np.array(packed_x)
packed_y = np.array(packed_y)

print("packed_x.shape: ", packed_x.shape)
print("packed_y.shape: ", packed_y.shape)



# reduce data ---------------------------------------------------------------------------
reduced_x = []
reduced_y = []
i = 0
expand_portation = 1
while i < len(packed_y):
    j = i
    if (   np.array_equal(  packed_y[i]  ,  y_Encoder.transform([['up']]) [0]  )   ):
        while True:
            j += 1
            if (   np.array_equal(  packed_y[j]  ,  y_Encoder.transform([['flat_or_down']]) [0]  )   ):
                break
        expansion = int ((j - i) * expand_portation )
        
        if i-expansion > 0   and   j+expansion < len(packed_y):
            for k in range(i-expansion, j+expansion):
                reduced_x.append( packed_x[k] )
                reduced_y.append( packed_y[k] )
    else:
        j += 1
        
    i = j
    
    
reduced_x = np.array(reduced_x)
reduced_y = np.array(reduced_y)

print("reduced_x.shape: ", reduced_x.shape)
print("reduced_y.shape: ", reduced_y.shape)


count_up = 0
count_flat_or_down = 0
for i in range(0, len(reduced_y)):
    if (   np.array_equal(  reduced_y[i]  ,  y_Encoder.transform([['up']])[0]  )   ):
        count_up += 1
    else:
        count_flat_or_down += 1
print("up:", count_up)
print("flat_or_down:", count_flat_or_down)



# split training / testing ---------------------------------------------------------------------------
test_size_portion = 0.1
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(reduced_x, reduced_y, test_size = test_size_portion, shuffle = False)

# convert to numpy array --------------------------------------------------------------------------- (may not need)
train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)   # 轉成numpy array的格式，以利輸入 RNN
print("train_x.shape: ", train_x.shape)
print("train_y.shape: ", train_y.shape)
print("test_x.shape: ", test_x.shape)
print("test_y.shape: ", test_y.shape)

# reshape data X to 3-dimension for model ---------------------------------------------------------------------------
assert len(features) == train_x.shape[2]
train_x = np.reshape( train_x, ( train_x.shape[0], train_x.shape[1], train_x.shape[2] ) )

assert len(features) == test_x.shape[2]
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2]))
print("train_x.shape: ", train_x.shape)
print("train_y.shape: ", train_y.shape)
print("test_x.shape: ", test_x.shape)
print("test_y.shape: ", test_y.shape)




# import sys
# sys.exit(0)



# setup model ---------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
dropout_rate = 0.2

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
assert len(features) == train_x.shape[2]
model.add(LSTM(units = 512, return_sequences = True, input_shape = (train_x.shape[1], train_x.shape[2])))
model.add(Dropout(dropout_rate))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 256, return_sequences = True))
model.add(Dropout(dropout_rate))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(dropout_rate))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 64))
model.add(Dropout(dropout_rate))

model.add(Dense(64, activation='relu'))
model.add(Dropout(dropout_rate))

# Adding the output layer
model.add(Dense(2, activation='sigmoid'))


# Compiling
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])




# checkpoint / callbacks set up ---------------------------------------------------------------------------
from keras.callbacks import ModelCheckpoint
filepath="model_weights_best.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor = 'val_loss', 
    mode = 'min', # {'auto', 'min', 'max'}
    save_best_only = True, # False: also save the model itself (structure)
    verbose = 1 # 0: silent, 1: displays messages when the callback takes an action
    )
callbacks_list = [checkpoint]



# run model (train) ---------------------------------------------------------------------------
validation_split_portion = 0.1
num_of_epochs = 40
num_of_batch_size = 256

history = model.fit(train_x, train_y, validation_split = validation_split_portion, shuffle = False, epochs = num_of_epochs, batch_size = num_of_batch_size, callbacks = callbacks_list)



# save model ---------------------------------------------------------------------------
# load best model
model.load_weights("model_weights_best.h5")
results = model.evaluate(test_x, test_y)

# save the whole model
model.save("model_whole.h5")

# test model ---------------------------------------------------------------------------
predicted_y = model.predict(test_x)

# convert back to original label
predicted_y = y_Encoder.inverse_transform( predicted_y )
test_y = y_Encoder.inverse_transform( test_y )
# predicted_y, test_y ->      [ ['flat_or_down'] , ['up'], ... ]

# output result
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_true=test_y, y_pred=predicted_y, labels = [ 'flat_or_down', 'up' ])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [ 'flat_or_down', 'up' ])
import matplotlib.pyplot as plt
cm_display.plot()
plt.show()



