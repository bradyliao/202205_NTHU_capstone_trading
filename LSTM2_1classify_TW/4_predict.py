import numpy as np
import pandas as pd

date_product_ = '20220531_TXFF2_'
csv_file_name = date_product_ + '2processed.csv'

dataset_in = pd.read_csv(csv_file_name)

# testing (very few data)
dataset_in = dataset_in[:][:10000]


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
#                             flat_or_down, up

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




from keras import models
model = models.load_model("model_whole.h5")


# predict
predicted_y = model.predict(packed_x)
predicted_y = y_Encoder.inverse_transform( predicted_y )

packed_y = y_Encoder.inverse_transform( packed_y )


print("packed_y.shape: ", packed_y.shape)
print("predicted_y.shape: ", predicted_y.shape)



# output result
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_true=packed_y, y_pred=predicted_y, labels = [ 'flat_or_down', 'up' ])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [ 'flat_or_down', 'up' ])
import matplotlib.pyplot as plt
cm_display.plot()
plt.show()



