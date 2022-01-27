from fetch import Fetch
from strategy import Strategy
from utility import Utility
from orders import Orders
from indicators import Indicators
# from analyze import Analyze

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from datetime import datetime as dt
from time import time
import math

import multiprocessing as mp
from multiprocessing import Pool

import warnings
        
class Backtest:
    
    def __init__(self):
        self.tickers = ['SPY']#,'QQQ','DIA']
        
        # self.startDate = pd.Timestamp(year=2020,month=3,day=23)
        self.startDate = pd.Timestamp(year=2006,month=1,day=1)
        # self.startDate = pd.Timestamp(year=dt.now().year-3,month=dt.now().month,day=dt.now().day)
        
        self.endDate = pd.Timestamp(year=2012,month=1,day=1)
        # self.endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)
        
        self.initialFunds = 10000
        
        self.run(True,False)
    
    ###########################################################################
    def test(self):
        stockFetch = Fetch(self.tickers,self.startDate)
        self.Data = stockFetch.getPrices()
        
    ###########################################################################
    def run(self,fetchOpt,writeOpt):            
        if fetchOpt:
            stockFetch = Fetch(self.tickers,self.startDate)
            Data = stockFetch.getPrices()
            
            if writeOpt:
                with pd.ExcelWriter('data.xlsx') as writer:
                    for ticker in self.tickers:
                        self.data = Data[ticker]
                        self.data.to_excel(writer, sheet_name=ticker)
    
        ######################################################################
        
        Info = {}
        # Params = {}
        # Funds = {}
        if not fetchOpt:
            Data = {}

        for ticker in self.tickers:
            # tic = time()
            
            if not fetchOpt:
                self.data = pd.read_excel('data.xlsx',sheet_name=ticker,index_col=0)
                self.data = self.data.loc[self.startDate:self.endDate]
                Data[ticker] = data
            else:
                self.data = Data[ticker]
            
            self.info = {'delay':[],
                         'sma':{'avg':[],
                                'std':[]}}
            
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
            
            self.data['Smooth'] = Utility.smooth(list(self.data['Close']),avgType='simple',window=5,iterations=1)
            self.data['Regression'] = Strategy.regression(self.data['Close'],curveType='logarithmic')
            
            # toc = time() - tic
            # print('Elapsed time for ' + ticker + ': ' + '{:.2f}'.format(toc) + ' sec')
            
            ###################################################################
            ###################################################################
            
# Setup plots
            Figure = {}
            Figure['Indicator'], indAxs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
            Figure['Value'], valAxs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
            
            Utility.setPlot(valAxs[0],logscale=False,xlimits=[self.data.index.values[0],self.data.index.values[-1]])
            valAxs[0].set_yscale('log')
            
            for ax in valAxs:
                ax.grid()
                ax.grid(axis='x',linestyle='--')
            
            ###################################################################
            
            # PLOT 1
            ax = indAxs[0]
            Utility.setPlot(ax,logscale=False,xlimits=[self.data.index.values[0],self.data.index.values[-1]])
            
            ax.plot(self.data['Close'],linewidth=0.5)
            # ax.plot(data['Smooth'],color='black',linewidth=1)
            
            self.data['SMAs'] = Strategy.movingAverage(self.data['Close'],
                                                      window=130,
                                                      avgType='exponential',
                                                      outputAll=True,
                                                      colors=('tab:green'),ax=ax,plotOpt=True)
            
            self.data['SMAf'] = Strategy.movingAverage(self.data['Close'],
                                                      window=40,
                                                      avgType='exponential',
                                                      outputAll=True,
                                                      colors=('tab:orange'),ax=ax,plotOpt=True)
            
            # Strategy.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
            # Strategy.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
            # data['BB'] = Strategy.bollingerBands(data['Close'],window=20,avgType='simple',ax=ax,plotOpt=True)
            # data['VAP'] = Strategy.volumeAtPrice(data,numBins=15,volumeType='all',integrated=True,ax=ax,plotOpt=True)
            
            ##################################################################
            
            # PLOT 2
            ax = indAxs[1]
            Utility.setPlot(ax)
            
            self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
            [avg,std] = Strategy.avgPrice(self.data['SMA_Diff'],outputAll=True)
            
            ax.plot(self.data.index.values,self.data['SMA_Diff']) 
            
            self.info['sma']['avg'] = avg[0]
            self.info['sma']['std'] = std
            
            ##################################################################

            # PLOT 3
            ax = indAxs[2]
            Utility.setPlot(ax)
            self.data['MACD'] = Strategy.macd(self.data['Close'],
                                              fast=7,slow=8,sig=3,
                                              avgType='logarithmic',
                                              ax=ax,plotOpt=True)
            
            self.data['MACD_avg'] = Strategy.avgPrice(pd.Series(list(zip(*self.data['MACD']))[0],index=self.data.index.values),
                                                      colors='tab:blue',ax=ax,plotDev=True,plotOpt=True)
            self.info['delay'] = 2
            
            ##################################################################
            ##################################################################
            
# Null and Optimized funds
            [self.optFunds,self.nullFunds] = self.compare()
            
# Strategy funds
            # self.order = self.strategy()
            
            # stratFunds = self.order.value
            
            # stratdates = [strat[0] for strat in stratFunds]
            # stratvalue = [strat[1] for strat in stratFunds]
            
            # nulldates = [null[0] for null in self.nullFunds]
            # nullvalue = [null[1] for null in self.nullFunds]
            
            # valAxs[0].plot(nulldates,nullvalue,linestyle='dashed')
            # valAxs[0].plot(stratdates,stratvalue)
            
            # dates = list(zip(*self.order.value))[0]
            
            # diff = self.order.info.indexDiff
            # valAxs[1].plot(dates,diff,color='tab:orange')
            
            # drawdown = self.order.info.drawdown
            # nullDrawdown = self.order.info.nullDrawdown
            # valAxs[2].plot(dates,nullDrawdown,linestyle='dashed')
            # valAxs[2].plot(dates,drawdown)
            
            ###################################################################
            # df = pd.read_csv('Files/MACD_test3.xlsx',index_col=0)
            # df = df[df['f']>nullvalue[-1]]
            # df = df[df['numSell']>10]
        
            # self.outputs = self.exploration(data, info, avgType=df['a'], fast=df['x'], slow=df['y'], sig=df['z'], delay=df['d'])
            
            # outputs = self.outputs
            
            # outputDf = pd.DataFrame.from_dict(outputs)
            # outputDf.to_excel("MACD_test2.xlsx")
            
            ###################################################################
            self.setupExplore(ticker)
            # self.allResults = self.exploration(avgType=['simple'],windowSlow=[50],windowFast=[20],steepness=[3])
            
            Info[ticker] = self.info
                        
        self.Data = Data
        self.Info = Info
        # self.Params = Params
        # self.Funds = Funds
        # self.Figure = Figure
        
    ###########################################################################
    def setupExplore(self,ticker):
        inputs = {'avgType':[],
                  'windowSlow':[],
                  'windowFast':[],
                  'steepness':[],
                  'fast':[],
                  'slow':[],
                  'sig':[],
                  'delay':[]}
        
        wSlowRange = list(range(50,200,25))
        wFastRange = list(range(10,50,10))
        fastRange = list(range(2,12,2))
        slowRange = list(range(7,28,3))
        sigRange = list(range(2,15,2))
        delayRange = list(range(1,7))
        
        for avgType in ['simple','exponential','logarithmic']:
            for windowSlow in wSlowRange:
                for windowFast in wFastRange:
                    for fast in fastRange:
                        for slow in slowRange:
                            if slow < fast:
                                continue
                            for sig in sigRange:
                                for delay in delayRange:
                                    if avgType != 'simple':
                                        for steepness in range(1,6):
                                            inputs['avgType'].append(avgType)
                                            inputs['windowSlow'].append(windowSlow)
                                            inputs['windowFast'].append(windowFast)
                                            inputs['steepness'].append(steepness)
                                            inputs['fast'].append(fast)
                                            inputs['slow'].append(slow)
                                            inputs['sig'].append(sig)
                                            inputs['delay'].append(delay)
                                    else:
                                        inputs['avgType'].append(avgType)
                                        inputs['windowSlow'].append(windowSlow)
                                        inputs['windowFast'].append(windowFast)
                                        inputs['steepness'].append(0)
                                        inputs['fast'].append(fast)
                                        inputs['slow'].append(slow)
                                        inputs['sig'].append(sig)
                                        inputs['delay'].append(delay)
        
        dfIn = pd.DataFrame.from_dict(inputs)
        dfIn = dfIn.iloc[59000:]
        buffer = 59
        
        tic = time()
        dfStep = 1000
        for i,x in enumerate(range(0,len(dfIn),dfStep)):
            if x > len(dfIn) - dfStep:
                df = dfIn.iloc[x:]
            else:
                df = dfIn.iloc[x:x+dfStep]
            # df = dfIn
                
            with Pool(processes=mp.cpu_count()) as pool:
                results = []
                stepSize = int(np.ceil(len(df) / mp.cpu_count()))
                for n in range(0,len(df),stepSize):
                    if n > len(df) - stepSize:
                        ins = df.iloc[n:]
                    else:
                        ins = df.iloc[n:n+stepSize]
                
                    keywords = {'avgType':ins.avgType.tolist(),
                                'windowSlow':ins.windowSlow.tolist(),
                                'windowFast':ins.windowFast.tolist(),
                                'steepness':ins.steepness.tolist(),
                                'fast':ins.fast.tolist(),
                                'slow':ins.slow.tolist(),
                                'sig':ins.sig.tolist(),
                                'delay':ins.delay.tolist()}
                    
                    results.append(pool.apply_async(self.exploration, kwds=keywords))
               
                pool.close()
                pool.join() 
                
            [result.wait() for result in results]
            allResults = [r.get() for r in results]
            
            self.outputDf = Utility.dicts2df(allResults)
            self.outputDf.to_csv('Files/MACD_SMA/'+ticker+'_'+str(i+buffer)+'.csv')
        
        toc = time() - tic
        print('Runtime: ' + str(toc)) 

    ###########################################################################        
    def exploration(self,**kwargs): 
        outputs = {'avgType':[],
                   'windowSlow':[],
                   'windowFast':[],
                   'steepness':[],
                   
                   'fast':[],
                   'slow':[],
                   'sig':[],
                   'delay':[],
                   
                   'numBuy':[],
                   'numSell':[],
                   'avgEarn':[],
                   # 'drawdown':[],
                   # 'winLoss':[],
                   'value':[]}
                   # 'zzz':[]}
        
        for avgType,windowSlow,windowFast,steepness,fast,slow,sig,delay in \
            zip(kwargs['avgType'],kwargs['windowSlow'],kwargs['windowFast'], \
                kwargs['steepness'],kwargs['fast'],kwargs['slow'], \
                kwargs['sig'],kwargs['delay']):
            
            self.data['MACD'] = Strategy.macd(self.data['Close'],
                                              fast=fast,slow=slow,
                                              sig=sig,avgType=avgType)
            
            self.data['MACD_avg'] = Strategy.avgPrice(pd.Series(list(zip(*self.data['MACD']))[0],index=self.data.index.values))
            
            self.info['delay'] = delay
            
            self.data['SMAs'] = Strategy.movingAverage(self.data['Close'],
                                                       window=windowSlow,
                                                       avgType=avgType,
                                                       steepness=steepness,
                                                       outputAll=True)
            
            self.data['SMAf'] = Strategy.movingAverage(self.data['Close'],
                                                       window=windowFast,
                                                       avgType=avgType,
                                                       steepness=steepness,
                                                       outputAll=True)
            
            self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
            [avg,std] = Strategy.avgPrice(self.data['SMA_Diff'],outputAll=True)
            
            self.info['sma']['avg'] = avg[0]
            self.info['sma']['std'] = std
            
            order = self.strategy()
        
            outputs['avgType'].append(avgType)
            outputs['windowSlow'].append(windowSlow)
            outputs['windowFast'].append(windowFast)
            outputs['steepness'].append(steepness)
            
            outputs['fast'].append(fast)
            outputs['slow'].append(slow)
            outputs['sig'].append(sig)
            outputs['delay'].append(delay)
            
            outputs['numBuy'].append(order.info.numBuys)
            outputs['numSell'].append(order.info.numSells)
            outputs['avgEarn'].append(order.info.avgEarn)
            # outputs['drawdown'].append(order.info.drawdown)
            # outputs['winLoss'].append(order.info.winLoss)
            outputs['value'].append(order.value[-1][1])
            # outputs['zzz'].append(avgType + '-' + str(windowSlow) + '-' + str(windowFast) + '-' + str(steepness))
                                
        return outputs
    
    ###########################################################################
    def strategy(self):
# Set buy/sell indicators
        indicator = {}
        # indicator['BB'] = Indicators.BB(data['Close'],data['BB'])
        # indicator['MACD'] = Indicators.MACD(data['MACD'],data['MACD_avg'])
        # indicator['SMA'] = Indicators.SMA(self.data['SMA'])
        # indicator['ATR'] = Indicators.ATR(data)
        
        indicator['SMA'] = Indicators.SMA_Delta(self.data['SMAs'],self.data['SMAf'],self.data['SMA_Diff'],self.info['sma'])
        indicator['MACD'] = Indicators.MACD_Delta(self.data['MACD'],delay=self.info['delay'],avg=self.data['MACD_avg'])
        
# Initiate orders
        order = Orders(self.initialFunds,runNull=True)
        
        seekBuy = True
        seekSell = False
        
        buy = False
        sell = False
        for i in range(2,len(self.data)):
            date = self.data.index.values[i]
            
            if seekBuy == True:
                if indicator['MACD'].buy(i) and indicator['SMA'].buy(i):
                    buy = True
                    seekBuy = False
            
            if seekSell == True:
                if indicator['MACD'].sell(i) or indicator['SMA'].sell(i):
                    sell = True
                    seekSell = False
                    
                # if (data['Close'].iloc[i-1] - order.buyPrice) / order.buyPrice < -0.03:
                #     sell = True
                #     seekSell = False
            
            if sell == True:
                order.sell(self.data['Open'].iloc[i],date)                  
                
                sell = False
                seekBuy = True
            elif buy == True:
                order.buy(self.data['Open'].iloc[i],date)
                
                buy = False  
                seekSell = True
            else:
                order.hold(self.data['Close'].iloc[i],date)
                
            # print(date[:10] + ' --- ' + '{:,.0f}'.format(order.value[-1][1]))
        
        order.analyze()
        
        stratFunds = order.value
        print('Strategy Funds:  $' + '{:,.0f}'.format(stratFunds[-1][1]))
        
        return order
    
    ###########################################################################
    def compare(self):
        prices = self.data['Smooth']
        
        [peaks,troughs]  = Utility.findExtrema(list(prices),endsOpt=False)            
        extrema = np.asarray(sorted(np.concatenate((peaks,troughs)),key=lambda x:x[0]))
        
        optOrder = Orders(self.initialFunds)  
        nullOrder = Orders(self.initialFunds)  
        
        for i in range(len(self.data)):
            date = self.data.index.values[i]
            
            if i == 0:
                nullOrder.buy(self.data['Open'].iloc[i],date)
                optOrder.hold(0,date)
            else:
                nullOrder.hold(self.data['Close'].iloc[i],date)
                
                if i in extrema[:,0]:
                    if prices.iloc[i] < prices.iloc[i-1]:
                        optOrder.buy(self.data['Open'].iloc[i],date)
                    else:
                        if optOrder.shares > 0:
                            optOrder.sell(self.data['Open'].iloc[i],date)
                else:
                    optOrder.hold(self.data['Close'].iloc[i],date)

        nullFunds = nullOrder.value
        print('Null Funds:      $' + '{:,.0f}'.format(nullFunds[-1][1]))
        
        optFunds = optOrder.value
        # print('Optimized Funds: $' + '{:,.0f}'.format(optFunds[-1][1]))
        
        return [optFunds,nullFunds]
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Running...')
    
    backtest = Backtest()
    data = backtest.Data
    order = backtest.order
    # output = backtest.outputsDf