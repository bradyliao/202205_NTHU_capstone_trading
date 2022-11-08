import pandas as pd
import talib

date_product_ = '20220530_TXFF2_'

csv_file_name = date_product_ + '1extracted.csv'
data_in = pd.read_csv(csv_file_name)


start_index = 179
next_few = 30
# next_few_half = int(next_few/2)



newData = list()
future_buy_to_now_sell_up = 0
future_buy_to_now_sell_flat_or_down = 0


for i in range(start_index, data_in.shape[0]-next_few):
# for i in range(start_index, 200000):
    # time_in_ms
    time_list = data_in['info_time'][i].split(":")
    hr, minute, sec, ms = time_list[0], time_list[1], time_list[2], time_list[3]
    time_in_ms = int(hr) * 3600000 + int(minute) * 60000 + int(sec) * 1000 + int(ms)
    # time_in_sec = int(hr) * 3600 + int(minute) * 60 + int(sec)
    
    # time_diff
    if i == start_index:
        time_diff = 0
    else:
        time_diff = time_in_ms - time_in_ms_prev
    time_in_ms_prev = time_in_ms
    
    
    # future buy sell relationship
    future_buy_to_now_sell_trend = 'flat_or_down'
    future_buy_trend = 'flat_or_up'
    
    future_sell_to_now_buy_trend = 'flat_or_up'
    future_sell_trend = 'flat_or_down'
    
    for j in range(1, next_few):
        if data_in[' 1th_buy_price'][i+j] > data_in[' 1th_sell_price'][i]: # and future_buy_to_now_sell_trend == 'flat_or_down':
            future_buy_to_now_sell_trend = 'up'
        
        if data_in[' 1th_buy_price'][i+j] < data_in[' 1th_buy_price'][i]: # and future_buy_trend == 'flat_or_down':
            future_buy_trend = 'down'
            
        if data_in[' 1th_sell_price'][i+j] < data_in[' 1th_buy_price'][i]: # and future_sell_to_now_buy_trend == 'flat_or_up':
            future_sell_to_now_buy_trend = 'down'
        
        if data_in[' 1th_sell_price'][i+j] > data_in[' 1th_sell_price'][i]: # and future_sell_trend == 'flat_or_down':
            future_sell_trend = 'up'
        
    # # future buy sell relationship - stastics
    # if future_buy_to_now_sell_trend == 'up':
    #     future_buy_to_now_sell_up += 1
    # else:
    #     future_buy_to_now_sell_flat_or_down += 1
    
    
    
    
    
    # newData.append([ data_in['info_time'][i], time_in_ms, data_in[' 1th_buy_price'][i]/100, data_in[' 1th_sell_price'][i]/100, future_buy_to_now_sell_trend, future_buy_trend, future_sell_to_now_buy_trend, future_sell_trend ])
    newData.append([   int( data_in[' 3th_buy_qty'][i] ) , int( data_in[' 3th_buy_price'][i] / 100 )  ,  int( data_in[' 2th_buy_qty'][i] ) , int( data_in[' 2th_buy_price'][i] / 100 )  ,  int( data_in[' 1th_buy_qty'][i] ) , int( data_in[' 1th_buy_price'][i] / 100 )  ,
                        future_sell_to_now_buy_trend , future_buy_trend  ,  time_diff  ,  future_sell_trend , future_buy_to_now_sell_trend ,
                        int( data_in[' 1th_sell_price'][i] / 100 ) , int( data_in[' 1th_sell_qty'][i] )  ,  int( data_in[' 2th_sell_price'][i] / 100 ) , int( data_in[' 2th_sell_qty'][i] )  ,  int( data_in[' 3th_sell_price'][i] / 100 ) , int( data_in[' 3th_sell_qty'][i] )  
                    ] )



output_df = pd.DataFrame(data=newData)
# output_df.columns = ['time_list', 'time_in_sec', '1th_buy_price', '1th_sell_price', 'future_buy_to_now_sell_trend', 'future_buy_trend', 'future_sell_to_now_buy_trend', 'future_sell_trend']
output_df.columns = ['3th_buy_qty', '3th_buy_price', '2th_buy_qty', '2th_buy_price', '1th_buy_qty', '1th_buy_price', 'future_sell_to_now_buy_trend', 'future_buy_trend', 'time_diff', 'future_sell_trend', 'future_buy_to_now_sell_trend', '1th_sell_price', '1th_sell_qty', '2th_sell_price', '2th_sell_qty', '3th_sell_price', '3th_sell_qty']

# talib
temp_macd, temp_macdsignal, output_df['1th_buy_price_MACDhist'] = talib.MACD(output_df['1th_buy_price'], fastperiod=12, slowperiod=26, signalperiod=9)
temp_macd, temp_macdsignal, output_df['1th_sell_price_MACDhist'] = talib.MACD(output_df['1th_sell_price'], fastperiod=12, slowperiod=26, signalperiod=9)
output_df.drop(index = output_df.index[:33])



output_df.to_csv( date_product_ + '2processed.csv')#, index=False)


# # future buy sell relationship - stastics
# print("next_few: ", next_few)
# print("future_buy_to_now_sell_up: ", future_buy_to_now_sell_up)
# print("future_buy_to_now_sell_flat_or_down: ", future_buy_to_now_sell_flat_or_down)