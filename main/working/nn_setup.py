from fetch import Fetch_Alpha
# from strategy import Indicators
from utility import Utility
# from orders import Orders
# from strategies import Strategy
# from analyze import Analyze

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt

from typing import Optional,Any

def trainingData(data,minHold:Optional[int]=None,minDelta:Optional[float]=None):
    def deleteIndex(index,*arrs):
        newArrs = []
        for arr in arrs:
            newArrs.append(np.delete(arr,index))

        return newArrs
        
    """ GET PEAKS/TROUGHS OF SMOOTH DATA """
    peaksTroughs = Utility.findExtrema(list(data.Smooth), endsOpt=False)
    peaks = peaksTroughs.peaks.dropna()
    peaks = list(zip(peaks.index.values,peaks.to_list()))
    troughs = peaksTroughs.troughs.dropna()  
    troughs = list(zip(troughs.index.values,troughs.to_list()))  
    # extrema = np.asarray(sorted(np.concatenate((peaks, troughs)), key=lambda x: x[0])).transpose()
    # print(len(extrema[0]))

    """ RE-ORGANIZE PEAKS/TROUGHS INTO SINGLE ARRAY """
    dates = [np.asarray(extrema).transpose()[0].astype(int) for extrema in [troughs,peaks]]
    prices = [data.Close.iloc[np.asarray(extrema).transpose()[0].astype(int)].to_numpy() for extrema in [troughs,peaks]]

    """ REMOVE FIRST POINT IF A PEAK """
    if dates[0][0] > dates[1][0]:
        dates[1] = dates[1][1:]
        prices[1] = prices[1][1:]

    """ REMOVE DATA THAT DOESN'T MEET THRESHOLDS """
    if not minHold is None:
        n = 0
        while True:
            if dates[1][n] - dates[0][n] < minHold: # position not held long enough
                [dates[1],prices[1]] = deleteIndex(n,dates[1],prices[1])

                if len(dates[0]) > n + 1:
                    [dates[0],prices[0]] = deleteIndex(n+1,dates[0],prices[0])

            if len(dates[0]) > n + 1:
                if dates[0][n+1] - dates[1][n] < minHold: # hold before purchase not long enough
                    [dates[0],prices[0]] = deleteIndex(n+1,dates[0],prices[0])
                    [dates[1],prices[1]] = deleteIndex(n+1,dates[1],prices[1])

            n += 1
            if len(dates[0]) <= n:
                break

    if not minDelta is None:
        n = 0
        while True:
            isDelete = False

            if (prices[1][n] - prices[0][n]) / prices[0][n] < minDelta: # gain on holdings below threshold
                [dates[1],prices[1]] = deleteIndex(n,dates[1],prices[1])

                if len(dates[0]) > n + 1:
                    [dates[0],prices[0]] = deleteIndex(n+1,dates[0],prices[0])

                isDelete = True

            if len(dates[0]) < n - 1:
                if abs((prices[1][n] - prices[0][n+1]) / prices[1][n]) < minDelta: # next buy not large enough drop in price
                    [dates[0],prices[0]] = deleteIndex(n+1,dates[0],prices[0])
                    [dates[1],prices[1]] = deleteIndex(n,dates[1],prices[1])

                    isDelete = True

            n += 1 if isDelete is False else 0
            if len(dates[0]) <= n:
                break

    """ SET BUY/SELL VALUES """
    ind = 0
    lastTrans = data.Close.iloc[0]
    nextTrans = prices[0][0]
    buy,sell = [],[]
    for n in range(len(data)):
        if n in dates[0]: #buy date
            """ IF A BUY DATE, SET BUY AND UPDATE NEXT/LAST TRANSACTION """
            buy.append(1)
            sell.append(-1)

            lastTrans = prices[0][ind]
            if ind+1 < len(prices[0]):
                nextTrans = prices[1][ind]
            else:
                nextTrans = data.Close.iloc[-1]
        elif n in dates[1]: #sell date
            """ IF A SELL DATE, SET SELL AND UPDATE NEXT/LAST TRANSACTION """
            buy.append(-1)
            sell.append(1)

            lastTrans = prices[1][ind]
            if ind+1 < len(prices[0]):
                nextTrans = prices[0][ind+1]
            else:
                nextTrans = data.Close.iloc[-1]
            
            ind += 1
        else:
            """ GET RELATIVE BUY/SELL VALUE AND SET APPROPRIATELY """
            norm = Utility.normalize(data.Close.iloc[n],lastTrans,nextTrans,-1,1)
            if lastTrans > nextTrans: #next buy
                buy.append(norm)
                sell.append(-1*norm)
            else: #next sell
                buy.append(-1*norm)
                sell.append(norm)  

    return buy,sell,dates,prices

""" SET TIMEFRAME AND FETCH """
startDate = pd.Timestamp(year=dt.now().year-1,month=dt.now().month,day=dt.now().day)
startDateWk = pd.Timestamp(year=dt.now().year-3,month=dt.now().month,day=dt.now().day)
startDateMth = pd.Timestamp(year=dt.now().year-5,month=dt.now().month,day=dt.now().day)

tickers = ['VTI']

stockFetch = Fetch_Alpha(tickers, startDate)
dataDay = stockFetch.getPrices()[tickers[0]]

stockFetch.startDate = startDateWk
dataWk = stockFetch.getPrices(increment='weekly')[tickers[0]]
dataWk = dataWk[dataWk.index < startDate]

stockFetch.startDate = startDateMth
dataMth = stockFetch.getPrices(increment='monthly')[tickers[0]]
dataMth = dataMth[dataMth.index < startDateWk]

data = pd.concat((dataMth,dataWk,dataDay))

# data = Data[tickers[0]]
data['Smooth'] = Utility.smooth(list(data.Close),avgType='simple',window=5,iterations=5)

""" SET TRAINING DATA """
buy,sell,dates,prices = trainingData(data)#,minDelta=0.05)

""" PLOT """
xvals = np.arange(len(data))

ax:list[plt.Axes]
_, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax[0].plot(data.index.values,xvals,data.Close,linewidth=0.5)
ax[0].plot(data.index.values,xvals,data.Smooth,color='black',linewidth=1)
ax[0].scatter(data.index.values[dates[0]],prices[0],c='g')
ax[0].scatter(data.index.values[dates[1]],prices[1],c='r')

ax[1].scatter(data.index.values,buy,c=buy,cmap='RdYlGn')
# ax2.plot(xvals,sell,color='red')
ax[1].set_ylim(-5,5)

plt.show()