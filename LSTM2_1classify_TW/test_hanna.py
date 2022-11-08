ast_sma_10 = 0
last_sma_30 = 0
last_k = 0
last_d = 0
last_close = 0
last_DIF = 0
last_DEM = 0
last_cci = 0



flag = 0
tot = 0
budget = 10000000
cur = 0
cnt_act = 0
act_buy = 0
act_sell = 0
for i in df.index:
    if(flag != 0):
        cnt_act += 1
        
    if(last_close < df["Close"][i]):
        print("Date", i, "up")
        if(flag == 1):
            tot += 1
        elif(flag == -1):
            tot -= 1
    else:
        print("Date", i, "down")
        if(flag == -1):
            tot += 1
        elif(flag == 1):
            tot -= 1
    print(budget)
    flag = 0
    """
    if((df["sma_10"][i] > df["sma_30"][i]) & (last_sma_10 < last_sma_30)): #MA黃金交叉：短均線向上突破長均線
        print("sma_1")
        flag = 1
    if((df["sma_10"][i] < df["sma_30"][i]) & (last_sma_10 > last_sma_30)): #MA死亡交叉：短均線向下突破長均線
        print("sma_-1")
        flag = -1
    """
    if((df['k'][i] > df['d'][i]) & (last_k < last_d)): #KD黃金交叉：K值向上突破D值
        print("kd_1")
        flag = 1
    if((df['k'][i] < df['d'][i]) & (last_k > last_d)): #KD死亡交叉：K值向下跌破D值
        print("kd_-1")
        flag = -1
        
    
    if((df["DIF"][i] > df["DEM"][i]) & (last_DIF < last_DEM)):
        print("MACD_1")
        flag = 1
    elif((df["DIF"][i] < df["DEM"][i]) & (last_DIF > last_DEM)):
        print("MACD_-1")
        flag = -1
      
    if(df['k'][i] < 20):
        print("kd_1")
        flag = 1
    elif(df['k'][i] > 80):
        print("kd_-1")
        flag = -1



    if(df['Open'][i] == df['Close'][i]):
        if(df['Open'][i] == df['High'][i]): #open=close=high: 買盤
            flag = 1
        elif(df['Open'][i] == df['Low'][i]): #open=close=low: 下跌
            flag = -1
    elif(df['Open'][i] < df['Close'][i]): #red
        up_line = (df['High'][i] - df['Close'][i])
        down_line = (df['Open'][i] - df['Low'][i])
        if( up_line == 0 ): # 買
            flag = 1
        elif( down_line == 0 ): # 賣
            flag = -1
    elif(df['Open'][i] > df['Close'][i]):
        up_line = (df['High'][i] - df['Open'][i])
        down_line = (df['Close'][i] - df['Low'][i])
        if( down_line == 0 ): # 賣
            flag = -1
        elif( up_line == 0 ): # 買
            flag = 1
    
    if((df['CCI'][i] > 100) & (last_cci < 100)): #CCI值向上突破100
        print("cci_1")
        flag = 1
    if((df['CCI'][i] < -100) & (last_cci > -100)): #CCI值向下跌破-100
        print("cci_-1")
        flag = -1
    elif((df['CCI'][i] > -100) & (last_cci < -100)): #CCI值向上突破-100
        print("cci_1")
        flag = 1
        
    if(flag == 1): #buy one
        budget -= df['Close'][i]
        cur += 1
        act_buy+=1
        print("Date: ",i, ", buy 1, cur", cur,", budget", budget)
    elif(flag == -1): # sell all
        if(cur != 0):
            act_sell+=1
            budget += df['Close'][i]*cur
            print("Date: ",i, ", sell", cur, ", budget", budget, ", gain", budget-10000000)
            cur = 0
            
    
    last_sma_10 = df["sma_10"][i]
    last_sma_30 = df["sma_30"][i]
    last_k = df["k"][i]
    last_d = df["d"][i]
    last_close = df["Close"][i]
    last_DIF = df['DIF'][i]
    last_DEM = df['DEM'][i]
    last_DEM = df['CCI'][i]