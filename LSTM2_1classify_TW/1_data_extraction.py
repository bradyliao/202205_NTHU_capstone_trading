import pandas as pd

date_ = '20220531_'

csv_file_name = date_ + '0original.csv'
data_in = pd.read_csv(csv_file_name)


print( date_ + "TXFF2: ", data_in[' product_id'].value_counts()[' TXFF2               '])
df_TXFF2 = pd.DataFrame(data_in[data_in[' product_id'] == ' TXFF2               '])
df_TXFF2.to_csv( date_ + 'TXFF2' + '_1extracted.csv', index=False )


print( date_ + "TXFG2: ", data_in[' product_id'].value_counts()[' TXFG2               '])
df_TXFG2 = pd.DataFrame(data_in[data_in[' product_id'] == ' TXFG2               '])
df_TXFG2.to_csv( date_ + 'TXFG2' + '_1extracted.csv', index=False )



# column_names = list(data_in.columns.values)
# print(column_names)
