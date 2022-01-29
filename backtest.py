from fetch import Fetch
from strategy import Strategy
from utility import Utility
from orders import Orders
from indicators import Indicators
from analyze import Analyze

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            
            ax.set_title('Price')
            ax.plot(self.data['Close'],linewidth=0.5)
            # ax.plot(data['Smooth'],color='black',linewidth=1)
            
            self.data['SMAs'] = Strategy.movingAverage(self.data['Close'],
                                                      window=100,
                                                      avgType='logarithmic',
                                                      steepness=2,
                                                      outputAll=True,
                                                      colors=('tab:green'),ax=ax,plotOpt=True)
            
            self.data['SMAf'] = Strategy.movingAverage(self.data['Close'],
                                                      window=20,
                                                      avgType='logarithmic',
                                                      steepness=2,
                                                      outputAll=True,
                                                      colors=('tab:orange'),ax=ax,plotOpt=True)
            
            ax.legend()
            
            # Strategy.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
            # Strategy.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
            # data['BB'] = Strategy.bollingerBands(data['Close'],window=20,avgType='simple',ax=ax,plotOpt=True)
            # data['VAP'] = Strategy.volumeAtPrice(data,numBins=15,volumeType='all',integrated=True,ax=ax,plotOpt=True)
            
            ##################################################################
            
            # PLOT 2
            ax = indAxs[1]
            Utility.setPlot(ax)
            
            self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
            self.data['SMA_Diff_Avg'] = Strategy.movingAverage(self.data['SMA_Diff'],
                                                               window=10,
                                                               avgType='exponential',
                                                               ax=ax,plotOpt=True)
            
            [avg,std] = Strategy.avgPrice(self.data['SMA_Diff'],outputAll=True,ax=ax,plotDev=True,plotOpt=True)
            
            ax.set_title('[MAf-MAs] % of Price')
            ax.plot(self.data.index.values,self.data['SMA_Diff']) 
            
            self.info['sma']['avg'] = avg[0]
            self.info['sma']['std'] = std*1.8
            
            ##################################################################

            # PLOT 3
            ax = indAxs[2]
            Utility.setPlot(ax)
            
            ax.set_title('ATR')
            self.data['MACD'] = Strategy.macd(self.data['Close'],
                                              fast=7,slow=8,sig=3,
                                              avgType='logarithmic')#,
                                              # ax=ax,plotOpt=True)
            
            self.data['MACD_avg'] = Strategy.avgPrice(pd.Series(list(zip(*self.data['MACD']))[0],index=self.data.index.values))#,
                                                      # colors='tab:blue',ax=ax,plotDev=True,plotOpt=True)
            self.info['delay'] = 2
            
            
            self.data['ATR'] = Strategy.atr(self.data,
                                            window=10,
                                            avgType='exponential',
                                            ax=ax,plotOpt=True)
            
            # ax.legend()
            
            ##################################################################
            ##################################################################
            
# Null and Optimized funds
            [self.optFunds,self.nullFunds] = self.compare()
            
# Strategy funds
            ###################################################################
            # self.order = self.strategy()
            
            # stratFunds = self.order.value
            
            # stratdates = [strat[0] for strat in stratFunds]
            # stratvalue = [strat[1] for strat in stratFunds]
            
            # nulldates = [null[0] for null in self.nullFunds]
            # nullvalue = [null[1] for null in self.nullFunds]
            
            # valAxs[0].set_title('Value')
            # valAxs[0].plot(nulldates,nullvalue,linestyle='dashed',label='Null')
            # valAxs[0].plot(stratdates,stratvalue,label='Strategy')
            # valAxs[0].legend()
            
            # dates = list(zip(*self.order.value))[0]
            
            # diff = self.order.info.indexDiff
            # valAxs[1].set_title('% Diff to Null')
            # valAxs[1].plot(dates,diff,color='tab:orange')
            
            # drawdown = self.order.info.drawdown
            # nullDrawdown = self.order.info.nullDrawdown
            # valAxs[2].set_title('% Drawdown')
            # valAxs[2].plot(dates,nullDrawdown,linestyle='dashed',label='Null')
            # valAxs[2].plot(dates,drawdown,label='Strategy')
            # valAxs[2].legend()
            
            ###################################################################
            self.inputs = {'avgType':[],
                           'window':[]}
            self.setupExplore(ticker)
            # self.allResults = self.exploration(avgType=['simple'],windowSlow=[50],windowFast=[20],steepness=[3])
            
            Info[ticker] = self.info
            
            
                        
        self.Data = Data
        self.Info = Info
        # self.Params = Params
        # self.Funds = Funds
        # self.Figure = Figure
        
    ###########################################################################
    def setupInputs2(self):
        inputs = {'avgType':[],
                  'windowSlow':[],
                  'windowFast':[],
                  'steepness':[],
                  'fast':[],
                  'slow':[],
                  'sig':[],
                  'delay':[]}
        
        df = Analyze.extractData()
        
        inputs['avgType'] = df.avgType
        inputs['windowSlow'] = df.windowSlow
        inputs['windowFast'] = df.windowFast
        inputs['steepness'] = df.steepness
        inputs['fast'] = df.fast
        inputs['slow'] = df.slow
        inputs['sig'] = df.sig
        inputs['delay'] = df.delay
        
        dfIn = pd.DataFrame.from_dict(inputs)
        
        return dfIn
        
    ###########################################################################
    def setupInputs(self):
        inputs = self.inputs
        
        windowRange = list(range(3,16))
        windowRange.extend(list(range(20,50,5)))
        
        for avgType in ['simple','exponential','logarithmic']:
            for window in windowRange:
                inputs['avgType'].append(avgType)
                inputs['window'].append(window)
        
        dfIn = pd.DataFrame.from_dict(inputs)
        
        return dfIn
    
    ###########################################################################
    def setupExplore(self,ticker):
        tic = time()
        
        dfIn = self.setupInputs()
        
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
                                'window':ins.window.tolist()}
                    
                    results.append(pool.apply_async(self.exploration, kwds=keywords))
               
                pool.close()
                pool.join() 
                
            [result.wait() for result in results]
            allResults = [r.get() for r in results]
            
            self.outputDf = Utility.dicts2df(allResults)
            self.outputDf.to_csv('Files/MACD_SMAdiff/'+ticker+'_'+str(i)+'.csv')
        
        toc = time() - tic
        print('Runtime: ' + str(toc)) 

    ###########################################################################        
    def exploration(self,**kwargs): 
        outputs = {'avgType':[],
                   'window':[],
                   
                   'numSell':[],
                   'exposure':[],
                   'drawdown':[],
                   'winLoss':[],
                   'value':[],
                   'zzz':[]}
        
        for avgType,window in zip(kwargs['avgType'],kwargs['window']):
            
            # self.data['SMAs'] = Strategy.movingAverage(self.data['Close'],
            #                                            window=windowSlow,
            #                                            avgType=avgType,
            #                                            steepness=steepness,
            #                                            outputAll=True)
            
            # self.data['SMAf'] = Strategy.movingAverage(self.data['Close'],
            #                                            window=windowFast,
            #                                            avgType=avgType,
            #                                            steepness=steepness,
            #                                            outputAll=True)
            
            # self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
            # self.data['SMA_Diff_Avg'] = Strategy.movingAverage(self.data['SMA_Diff'],
            #                                                    window=10,
            #                                                    avgType='exponential')
            # [avg,std] = Strategy.avgPrice(self.data['SMA_Diff'],outputAll=True)
            
            # self.info['sma']['avg'] = avg[0]
            # self.info['sma']['std'] = std*stdev
            
            self.data['ATR'] = Strategy.atr(self.data,
                                            window=window,
                                            avgType=avgType)
            
            order = self.strategy()
        
            outputs['avgType'].append(avgType)
            outputs['window'].append(window)
            
            outputs['numSell'].append(order.info.numSells)
            outputs['exposure'].append(order.info.exposure)
            outputs['drawdown'].append(order.info.maxDrawdown)
            outputs['winLoss'].append(order.info.winLoss)
            outputs['value'].append(order.value[-1][1])
            outputs['zzz'].append(avgType + '-' + str(window))
                                
        return outputs
    
    ###########################################################################
    def strategy(self):
# Set buy/sell indicators
        indicator = {}
        # indicator['BB'] = Indicators.BB(data['Close'],data['BB'])
        # indicator['MACD'] = Indicators.MACD(data['MACD'],data['MACD_avg'])
        # indicator['SMA'] = Indicators.SMA(self.data['SMA'])
        # indicator['ATR'] = Indicators.ATR(data)
        
        indicator['SMA'] = Indicators.SMA_Delta(self.data['SMAs'],
                                                self.data['SMAf'],
                                                self.data['SMA_Diff'],
                                                self.data['SMA_Diff_Avg'],
                                                self.info['sma'])
        indicator['MACD'] = Indicators.MACD_Delta(self.data['MACD'],
                                                  delay=self.info['delay'],
                                                  avg=self.data['MACD_avg'])
        
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
    # order = backtest.order
    # output = backtest.outputsDf