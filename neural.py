from fetch import Fetch
# from strategy import Strategy
from utility import Utility
# from orders import Orders
# from indicators import Indicators
# from analyze import Analyze

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

from datetime import datetime as dt

def trainingData(data):
    [peaks, troughs] = Utility.findExtrema(list(data['Smooth']), endsOpt=False)
    # extrema = np.asarray(sorted(np.concatenate((peaks, troughs)), key=lambda x: x[0])).transpose()
    # print(len(extrema[0]))

    dates = [np.asarray(extrema).transpose()[0].astype(int) for extrema in [troughs,peaks]]
    prices = [data.Close[np.asarray(extrema).transpose()[0].astype(int)] for extrema in [troughs,peaks]]

    if dates[0][0] > dates[1][0]:
        dates[1] = dates[1][1:]
        prices[1] = prices[1][1:]

    ind = 0
    lastTrans = data.Close[0]
    nextTrans = prices[0][0]
    buy,sell = [],[]
    for n in range(len(data)):
        if n in dates[0]: #buy date
            buy.append(1)
            sell.append(-1)

            lastTrans = prices[0][ind]
            if ind+1 < len(prices[0]):
                nextTrans = prices[1][ind]
            else:
                nextTrans = data.Close[-1]
        elif n in dates[1]: #sell date
            buy.append(-1)
            sell.append(1)

            lastTrans = prices[1][ind]
            if ind+1 < len(prices[0]):
                nextTrans = prices[0][ind+1]
            else:
                nextTrans = data.Close[-1]
            
            ind += 1
        else:
            if lastTrans > nextTrans: #next buy
                # print('waiting to buy')
                # print(str(lastTrans),'-',str(data.Close[n]),'-',str(nextTrans))
                norm = Utility.normalize(data.Close[n],lastTrans,nextTrans,-1,1)
                # print(norm)
                buy.append(norm)
                sell.append(-1*norm)
            else: #next sell
                # print('waiting to sell')
                # print(str(lastTrans),'-',str(data.Close[n]),'-',str(nextTrans))
                norm = Utility.normalize(data.Close[n],lastTrans,nextTrans,-1,1)
                # print(norm)
                buy.append(-1*norm)
                sell.append(norm)  

    return buy,sell

startDate = pd.Timestamp(year=dt.now().year-4,month=dt.now().month,day=dt.now().day)

tickers = ['SPY']
stockFetch = Fetch(tickers, startDate)
Data = stockFetch.getPrices()

data = Data[tickers[0]]
data['Smooth'] = Utility.smooth(list(data['Close']),avgType='simple',window=5,iterations=5)

buy,sell = trainingData(data)

_, ax = plt.subplots()
ax.plot(data['Close'],linewidth=0.5)
ax.plot(data['Smooth'],color='black',linewidth=1)
# ax.scatter(data.index.values[extrema[0].astype(int)],extrema[1])

ax2 = ax.twinx()
ax2.plot(data.index.values,buy,color='green')
ax2.plot(data.index.values,sell,color='red')
ax2.set_ylim(-5,5)

plt.show()