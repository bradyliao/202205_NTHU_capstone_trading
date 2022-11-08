

fee_rate = 0.00002
buy_price = 19000
sell_price = 19001

cost = 200 * buy_price *  (1+fee_rate) + 20
revenue = 200 * sell_price * (1-fee_rate) - 20

profit = revenue - cost

print(profit)

