from indicators import Indicators
from utility import Utility

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt


tickers = ['SPY']

# startDate = pd.Timestamp(year=2006,month=1,day=1)
startDate = pd.Timestamp(year=dt.now().year-9,month=dt.now().month,day=dt.now().day)
endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)

initialFunds = 10000

for ticker in tickers:
    data = pd.read_excel('data.xlsx',sheet_name=ticker,index_col=0)
    data = data.loc[startDate:]
        
    data['SMA20'] = Indicators.movingAverage(data['Close'],window=10,avgType='simple')
    # data['SMA50'] = Indicators.movingAverage(data['Close'],window=50,avgType='simple')
    # data['SMA200'] = Indicators.movingAverage(data['Close'],window=200,avgType='simple')
    # data['SMAe'] = Indicators.movingAverage(data['Close'],window=20,avgType='exponential',steepness=3)
    # data['BB'] = Indicators.bollingerBands(data['Close'],window=20)
    data['RSI'] = Indicators.rsi(data['Close'],window=14,avgType='simple')
    data['ATR'] = Indicators.atr(data,window=14,avgType='exponential')
    # data['MACD'] = Indicators.macd(data['Close'],fast=5,slow=10,sig=7,avgType='simple')
    # data['AD'] = Indicators.accDist(data)
    # data['VAP'] = Indicators.volumeAtPrice(data,numBins=25)
    # data['AVG'] = Indicators.avgPrice(data['ATR'])
    
    data['Smooth'] = Utility.smooth(list(data['Close']),avgType='simple',window=5,iterations=1)
    data['Regression'] = Indicators.regression(data['Close'],curveType='logarithmic')
    
    ##################################################################
    ##################################################################
    
    fig, axs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
    
    ##################################################################
    
    ax = axs[0]
    Utility.setPlot(ax,logscale=True,xlimits=[data.index.values[0],data.index.values[-1]])
    
    ax.plot(data['Close'],linewidth=0.5)
    ax.plot(data['Smooth'],color='black',linewidth=1)
    
    data['SMA'] = Indicators.movingAverage(data['Close'],window=20,avgType='simple',ax=axs[0:2],plotOpt=True)
    
    # Indicators.supportResistance(data['Smooth'],thresh=0.03,minNum=5,minDuration=10,style='both',ax=ax,plotOpt=True)            
    # for i in [1,2,3,4]:
    pattern = Indicators.trend(data['Close'],numIters=1,direction='both',segments=3,ax=ax,plotOpt=True)

    # data['BB'] = Indicators.bollingerBands(data['Close'],window=20,avgType='simple',ax=None,plotOpt=True)
    # data['VAP'] = Indicators.volumeAtPrice(data,numBins=30,volumeType='all',integrated=True,ax=ax,plotOpt=True)
    
    #################################################################
    
    # ax = axs[1]
    # Utility.setPlot(ax)
    # ax.plot(data['ATR'])
    
    
    #################################################################
    
    ax = axs[2]
    Utility.setPlot(ax)
    ax.plot(data['RSI'])

plt.show()