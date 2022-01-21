from fetch import Fetch
from strategy import Strategy
from utility import Utility
from orders import Orders
from indicators import Indicators

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt
from time import time
import math

# import multiprocessing as mp
# from multiprocessing.pool import Pool

import warnings
        
class Backtest:
    
    def __init__(self):
        self.tickers = ['SPY']
        # self.tickers = ['BLDR']#,'LYB']
        
        # self.startDate = pd.Timestamp(year=2020,month=3,day=23)
        self.startDate = pd.Timestamp(year=2006,month=1,day=1)
        # self.startDate = pd.Timestamp(year=dt.now().year-3,month=dt.now().month,day=dt.now().day)
        self.endDate = pd.Timestamp(year=2012,month=1,day=1)
        # self.endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)
        
        self.initialFunds = 10000
        
        self.run(True,False)
    
    ###########################################################################
    def run(self,fetchOpt,writeOpt):            
        if fetchOpt:
            stockFetch = Fetch(self.tickers,self.startDate)
            Data = stockFetch.getPrices()
            
            if writeOpt:
                with pd.ExcelWriter('data.xlsx') as writer:
                    for ticker in self.tickers:
                        data = Data[ticker]
                        data.to_excel(writer, sheet_name=ticker)
    
        ######################################################################
        
        info = {}
        Params = {}
        Funds = {}
        if not fetchOpt:
            Data = {}
        
        f = 1
        for ticker in self.tickers:
            # tic = time()
            
            if not fetchOpt:
                data = pd.read_excel('data.xlsx',sheet_name=ticker,index_col=0)
                data = data.loc[self.startDate:self.endDate]
                Data[ticker] = data
            else:
                data = Data[ticker]
            
            # data['SMA20'] = Strategy.movingAverage(data,window=10,avgType='simple')
            # data['SMA50'] = Strategy.movingAverage(data,window=50,avgType='simple')
            # data['SMA200'] = Strategy.movingAverage(data,window=200,avgType='simple')
            # data['SMAe'] = Strategy.movingAverage(data,window=20,avgType='exponential',steepness=3)
            # data['BB'] = Strategy.bollingerBands(data,window=20)
            # data['RSI'] = Strategy.rsi(data,window=14,avgType='simple')
            # data['ATR'] = Strategy.atr(data,window=14,avgType='exponential')
            # data['MACD'] = Strategy.macd(data,fast=5,slow=10,sig=7,avgType='simple')
            # data['AD'] = Strategy.accDist(data)
            # data['VAP'] = Strategy.volumeAtPrice(data,numBins=25)
            # data['AVG'] = Strategy.avgPrice(data['ATR'])
            
            # Strategy.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
            # Strategy.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
            # Strategy.extremaGaps(data['Smooth'],minDuration=10,minPerc=0.1)
            
            data['Smooth'] = Utility.smooth(list(data['Close']),avgType='simple',window=5,iterations=1)
            data['Regression'] = Strategy.regression(data['Close'],curveType='logarithmic')
            
            # toc = time() - tic
            # print('Elapsed time for ' + ticker + ': ' + '{:.2f}'.format(toc) + ' sec')
            
            ###################################################################
            ###################################################################
            
# Setup plots
            Figure = {}
            Figure['Indicator'], indAxs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
            Figure['Value'], valAxs = plt.subplots()
            
            Utility.setPlot(valAxs,logscale=False,xlimits=[data.index.values[0],data.index.values[-1]])
            
            valAxs.set_yscale('log')
            valAxs.grid()
            valAxs.grid(axis='x',linestyle='--')
            
            ###################################################################
            
            # PLOT 1
            ax = indAxs[0]
            Utility.setPlot(ax,logscale=False,xlimits=[data.index.values[0],data.index.values[-1]])
            
            ax.plot(data['Close'],linewidth=0.5)
            # ax.plot(data['Smooth'],color='black',linewidth=1)
            
            data['SMA'] = Strategy.movingAverage(data['Close'],
                                                  window=20,
                                                  avgType='simple',
                                                  ax=indAxs[1],plotDelta=True,plotOpt=True)
            
            # Strategy.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
            # Strategy.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
            # data['BB'] = Strategy.bollingerBands(data['Close'],window=20,avgType='simple',ax=ax,plotOpt=True)
            # data['VAP'] = Strategy.volumeAtPrice(data,numBins=15,volumeType='all',integrated=True,ax=ax,plotOpt=True)
            
            ##################################################################
            
            # PLOT 2
            ax = indAxs[1]
            Utility.setPlot(ax)
            
            ##################################################################
            
            # PLOT 3
            ax = indAxs[2]
            Utility.setPlot(ax)
            data['MACD'] = Strategy.macd(data['Close'],
                                         fast=2,slow=15,sig=4,
                                         avgType='logarithmic',
                                         ax=ax,plotOpt=True)
            
            data['MACD_avg'] = Strategy.avgPrice(pd.Series(list(zip(*data['MACD']))[0],index=data.index.values),colors='tab:blue',ax=ax,plotDev=True,plotOpt=True)
            info['delay'] = 5
            
            ##################################################################
            ##################################################################
            
# Null and Optimized funds
            [self.optFunds,self.nullFunds] = self.compare(data)
            
            nulldates = [null[0] for null in self.nullFunds]
            nullvalue = [null[1] for null in self.nullFunds]
            
# Strategy funds
            Funds[ticker] = None
            Params[ticker] = None
            
            # self.stratFunds = self.strategy(data,info)
            
            inputs = []
            for avgType in ['simple','exponential','logarithmic']:
                for fast in range(2,15):
                    for slow in range(5,30):
                        if slow < fast:
                            continue
                        
                        for sig in range(3,15):
                            inputs.append((avgType,fast,slow,sig))

            inputs = inputs[:5]
            inputs = list(zip(*inputs))
            self.outputs = self.exploration(data,info,avgType=inputs[0],fast=inputs[1],slow=inputs[2],sig=inputs[3])
            
            outputs = self.outputs
            del outputs['maxFunds']
            
            outputDf = pd.DataFrame.from_dict(outputs)
            outputDf.to_excel("MACD_test.xlsx")

# Plotting                
            # fig = plt.figure()
            # ax = plt.gca()
            # ax = plt.axes(projection='3d')
            
            # scatter = ax.scatter3D(x, y, z, c=v, s=np.asarray(d)*5, cmap='viridis')
            
            # ax.set_xlabel('fast')
            # ax.set_ylabel('slow')
            # ax.set_zlabel('value')
            # cb = fig.colorbar(scatter,ax=ax)
            # cb.set_label('signal')
        
            # stratdates = [strat[0] for strat in self.outputs['maxFunds']]
            # stratvalue = [strat[1] for strat in self.outputs['maxFunds']]
            
            # ax = valAxs
            # ax.plot(nulldates,nullvalue)
            # ax.plot(stratdates,stratvalue)
            
            f += 1
                        
        self.Data = Data
        # self.Params = Params
        # self.Funds = Funds
        # self.Figure = Figure

    ###########################################################################        
    def exploration(self,data,info,**kwargs): 
        avgType = kwargs['avgType']
        fast = kwargs['fast']
        slow = kwargs['slow']
        sig = kwargs['sig']
        
        outputs = {}
        x,y,z,d,a,f = [],[],[],[],[],[]
        
        maxValue = 0
        for avgType,fast,slow,sig in zip(kwargs['avgType'],kwargs['fast'],kwargs['slow'],kwargs['sig']):
            
            data['MACD'] = Strategy.macd(data['Close'],
                                         fast=fast,slow=slow,sig=sig,
                                         avgType=avgType)
            
            data['MACD_avg'] = Strategy.avgPrice(pd.Series(list(zip(*data['MACD']))[0],index=data.index.values))
            
            
            for delay in range(1,10):
                info['delay'] = delay
                self.stratFunds = self.strategy(data,info)
                
                if self.stratFunds[-1][1] > maxValue:                
                    maxValue = self.stratFunds[-1][1]
                    maxFunds = self.stratFunds
                    
                    # print('Strategy Funds:  $' + '{:,.0f}'.format(self.stratFunds[-1][1]))
                    # print('Fast:   ' + str(fast))
                    # print('Slow:   ' + str(slow))
                    # print('Signal: ' + str(sig))
                    # print('Delay:  ' + str(delay))
                    # print('Avg:    ' + str(avgType))
                    
                x.append(fast)
                y.append(slow)
                z.append(sig)
                d.append(delay)
                a.append(avgType)
                f.append(self.stratFunds[-1][1])
            
        outputs['x'] = x
        outputs['y'] = y
        outputs['z'] = z
        outputs['d'] = d
        outputs['a'] = a
        outputs['f'] = f
        outputs['maxFunds'] = maxFunds
                                
        return outputs
    
    ###########################################################################
    def strategy(self,data,info=None):
# Set buy/sell indicators
        indicator = {}
        # indicator['BB'] = Indicators.BB(data['Close'],data['BB'])
        # indicator['MACD'] = Indicators.MACD(data['MACD'],data['MACD_avg'])
        indicator['MACD'] = Indicators.MACD_Delta(data['MACD'],delay=info['delay'],avg=data['MACD_avg'])
        # indicator['SMA'] = Indicators.SMA(data['SMAe'])
        # indicator['ATR'] = Indicators.ATR(data)
        
# Initiate orders
        order = Orders(self.initialFunds)
        
        seekBuy = True
        seekSell = False
        
        buy = False
        sell = False
        for i in range(2,len(data)):
            date = data.index.values[i]
            
            if seekBuy == True:
                if indicator['MACD'].buy(i):
                    buy = True
                    seekBuy = False
            
            if seekSell == True:
                if indicator['MACD'].sell(i):
                    sell = True
                    seekSell = False
                    
                # if (data['Close'].iloc[i-1] - order.buyPrice) / order.buyPrice < -0.03:
                #     sell = True
                #     seekSell = False
            
            if sell == True:
                order.sell(data['Open'].iloc[i],date)                  
                
                sell = False
                seekBuy = True
            elif buy == True:
                order.buy(data['Open'].iloc[i],date)
                
                buy = False  
                seekSell = True
            else:
                order.hold(data['Close'].iloc[i],date)
                
            # print(date[:10] + ' --- ' + '{:,.0f}'.format(order.value[-1][1]))
        
        stratFunds = order.value
        print('Strategy Funds:  $' + '{:,.0f}'.format(stratFunds[-1][1]))
        
        return stratFunds
    
    ###########################################################################
    def compare(self,data):
        prices = data['Smooth']
        
        [peaks,troughs]  = Utility.findExtrema(list(prices),endsOpt=False)            
        extrema = np.asarray(sorted(np.concatenate((peaks,troughs)),key=lambda x:x[0]))
        
        optOrder = Orders(self.initialFunds)  
        nullOrder = Orders(self.initialFunds)  
        
        for i in range(len(data)):
            date = data.index.values[i]
            
            if i == 0:
                nullOrder.buy(data['Open'].iloc[i],date)
                optOrder.hold(0,date)
            else:
                nullOrder.hold(data['Close'].iloc[i],date)
                
                if i in extrema[:,0]:
                    if prices.iloc[i] < prices.iloc[i-1]:
                        optOrder.buy(data['Open'].iloc[i],date)
                    else:
                        if optOrder.shares > 0:
                            optOrder.sell(data['Open'].iloc[i],date)
                else:
                    optOrder.hold(data['Close'].iloc[i],date)

        nullFunds = nullOrder.value
        print('Null Funds:      $' + '{:,.0f}'.format(nullFunds[-1][1]))
        
        optFunds = optOrder.value
        print('Optimized Funds: $' + '{:,.0f}'.format(optFunds[-1][1]))
        
        return [optFunds,nullFunds]

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    backtest = Backtest()
    data = backtest.Data
    outputs = backtest.outputs
    
    # optFunds = backtest.optFunds
    # nullFunds = backtest.nullFunds
    
    # backtest.Figure['Indicator']