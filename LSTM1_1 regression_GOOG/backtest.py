'''

from cgi import test
import sys

backtest_file_name = sys.argv[1]

# import pandas as pd
# import numpy as np



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
dataset_in = pd.read_csv(backtest_file_name, index_col=False)






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




# X_raw = X_raw[33:]
X_raw = X_raw.dropna()

X_raw.to_csv("backtest_indicator_dropna.csv", index=False)
X_raw = pd.read_csv("backtest_indicator_dropna.csv", index_col=False)



# extract targets
y_raw = X_raw[target]

print(X_raw.shape[0])
print(y_raw.shape[0])

# calculate total num of data
total_num_data = len(X_raw)




# scaling
from sklearn.preprocessing import MinMaxScaler
scale_X = MinMaxScaler(feature_range = (0, 1))
X_scale = scale_X.fit_transform(X_raw)

scale_y = MinMaxScaler(feature_range = (0, 1))
y_scale = scale_y.fit_transform(y_raw)




# generate data in timesteps
X_data = []   #預測點的前 timesteps 天的資料
y_data = []   #預測點
for i in range(timesteps, len(X_scale) - days_forward):
    X_data.append( X_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_data.append( y_scale [ (i + days_forward) , 0] ) # data of the target value
    
    
# convert to numpy array
X_data, y_data = np.array(X_data), np.array(y_data)   # 轉成numpy array的格式，以利輸入 RNN


# reshape data X to 3-dimension for model
assert num_of_features == X_data.shape[2]
X_data = np.reshape( X_data, ( X_data.shape[0], X_data.shape[1], X_data.shape[2] ) )






import tensorflow as tf
from tensorflow import keras


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


model = keras.models.load_model("model_whole", custom_objects={'custo_loss': custo_loss, 'custo_acc': custo_acc })



# predict
predicted_stock_price = model.predict(X_data)
predicted_stock_price = scale_y.inverse_transform(predicted_stock_price)  # to get the original scale


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




# Visualising the test results
real_stock_price = scale_y.inverse_transform(y_scale)
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





a = pd.DataFrame(X_raw)
b = pd.DataFrame(predicted_stock_price_shifted)
b.columns = ['predicted']
c= pd.concat([a, b], axis=1)
print(c)
c.to_csv("backtest_predicted.csv", index=False)




'''



#temp
import pandas as pd
timesteps = 60
days_forward = 20









predicted_df = pd.read_csv("backtest_predicted.csv", index_col=False)


lookbacks = 60
lookback_threshold_rate_buy  = 0.0005
lookback_threshold_rate_sell = 0.0005
holding_units_limit = 10
skips_after_transcation = 0



lookback_average = 0

fee_fixed = 20
fee_rate = 0.00002
buy_unit = 1
sell_unit = 1
price_each_point = 2 #the price in csv is already *100
holding_units_current = 0
holding_units_max = 0

transaction_total_buy = 0
transaction_total_sell = 0

cost_current = 0
cost_max = 0
revenue = 0
profit = 0


f = open("test.txt", "w") 


for row in range(timesteps+lookbacks, predicted_df.shape[0]-days_forward+1):
    
    
    
    
    lookback_average = 0
    for i in range(1,lookbacks+1):
        lookback_average = lookback_average + predicted_df.iloc[row-i, 10]
    lookback_average = lookback_average/lookbacks
    
    
    
    
    
    # buy
    if predicted_df.iloc[row, 10] > lookback_average*(1+lookback_threshold_rate_buy):
        if holding_units_current >= holding_units_limit:
            continue
        
        
        transaction_total_buy += buy_unit
        holding_units_current += buy_unit
        cost_current = cost_current + ( buy_unit * price_each_point * predicted_df.iloc[row+1, 1] * (1+fee_rate) + fee_fixed )
        
        if holding_units_current > holding_units_max:
            holding_units_max = holding_units_current
        if cost_current > cost_max:
            cost_max = cost_current
        
        row += skips_after_transcation
        # f.write("buy" + \n")
    
    
    # sell
    if predicted_df.iloc[row, 10] < lookback_average*(1-lookback_threshold_rate_buy):
        if holding_units_current <= -holding_units_limit:
            continue
        
        
        transaction_total_sell += sell_unit
        holding_units_current -= sell_unit
        cost_current = cost_current - ( sell_unit * price_each_point * predicted_df.iloc[row+1, 1] * (1+fee_rate) + fee_fixed )
        
        if holding_units_current < -holding_units_max:
            holding_units_max = holding_units_current
        if cost_current < -cost_max:
            cost_max = cost_current
        
        row += skips_after_transcation
        # f.write("buy\n")

        
    # # sell
    # elif holding_units_current > 0 and predicted_df.iloc[row, 10] < lookback_average*(1-lookback_threshold_rate_sell):
    #     transaction_total_sell += holding_units_current
        
    #     revenue = holding_units_current * price_each_point * predicted_df.iloc[row+1, 1] * (1-fee_rate) - fee_fixed
    #     profit = profit + (revenue - cost_current)
        
    #     cost_current = 0
    #     holding_units_current = 0
        
    #     row += skips_after_transcation
        
    else:
        continue
    
    
    
# clear holding with the last price of the day
if holding_units_current > 0:
    transaction_total_sell += holding_units_current
        
    revenue = holding_units_current * price_each_point * predicted_df.iloc[-1, 1] * (1-fee_rate) - fee_fixed
    profit = profit + (revenue - cost_current)
        
    cost_current = 0
    holding_units_current = 0
    
if holding_units_current < 0:
    transaction_total_buy -= holding_units_current
        
    revenue = holding_units_current * price_each_point * predicted_df.iloc[-1, 1] * (1-fee_rate) - fee_fixed
    profit = profit + (revenue - cost_current)
        
    cost_current = 0
    holding_units_current = 0
    
    
    
    
    
print("lookbacks:               ", lookbacks)
print("skips after transcation: ", skips_after_transcation)
print("buy each transcation:    ", buy_unit)
print("lookback threshold buy:  ", lookback_threshold_rate_buy)
print("lookback threshold sell: ", lookback_threshold_rate_sell)
print("cost max:                ", cost_max)
print("holding_units_max:       ", holding_units_max)
print("transcation total buy:   ", transaction_total_buy)
print("transcation total sell:  ", transaction_total_sell)
print("profit:                  ", profit)

