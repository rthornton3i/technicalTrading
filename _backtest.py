from fetch import Fetch
from indicators import Indicators
from utility import Utility
from orders import Orders
from analyze import Analyze
from strategy import Strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt
from time import time
# import math
import os

import multiprocessing as mp
from multiprocessing import Pool

import warnings


class Backtest:

    def __init__(self):
        self.tickers = ['SPY'] #['SPY','QQQ','DIA']

        # Slow
        # self.startDate = pd.Timestamp(year=2012,month=4,day=1)
        # self.endDate   = pd.Timestamp(year=2014,month=3,day=1)

        # Medium
        # self.startDate = pd.Timestamp(year=2013,month=8,day=1)
        # self.endDate   = pd.Timestamp(year=2018,month=1,day=1)

        # Fast
        # self.startDate = pd.Timestamp(year=2017,month=6,day=1)
        # self.endDate   = pd.Timestamp(year=2020,month=2,day=1)

        # COVID
        # self.startDate = pd.Timestamp(year=2019,month=7,day=1)
        # self.endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)

        self.startDate = pd.Timestamp(year=2013,month=7,day=8)
        # self.startDate = pd.Timestamp(year=dt.now().year-10,month=dt.now().month,day=dt.now().day)

        self.endDate = pd.Timestamp(year=2023,month=7,day=7)
        # self.endDate = pd.Timestamp(year=dt.now().year, month=dt.now().month, day=dt.now().day)

        self.initialFunds = 10000
        self.inputs = self.Inputs()

        self.run(False, False)

    #=================================================================#
    def run(self, fetchOpt, writeOpt):
        if fetchOpt:
            fetch = Fetch(self.tickers, self.startDate, self.endDate)
            Data = fetch.getPrices()

            if writeOpt:
                with pd.ExcelWriter('data.xlsx') as writer:
                    for ticker in self.tickers:
                        self.data = Data[ticker]
                        self.data.to_excel(writer, sheet_name=ticker)
        else:
            Data = {}

        Figure = {}

        #=================================================================#

        for ticker in self.tickers:
            if not fetchOpt:
                self.data = pd.read_excel('Files/data.xlsx', sheet_name=ticker, index_col=0)
                self.data = self.data.loc[self.startDate:self.endDate]
                Data[ticker] = self.data
            else:
                self.data = Data[ticker]                        

            self.plotOpt = False
            # Figure[ticker] = self.plotStrategy()

            tic = time()

            #=================================================================#

            """ EXECUTE EXPLORATION """

            # self.exploreIndicators = ['MACD']

            # self.setupInputs()
            # self.setupExplore(ticker,'MACD')

            #=================================================================#

            """ EXECUTE STRATEGY """

            self.executeIndicators()
            self.executeStrategy()
            self.executeOrders()
            self.executeRetrospective()

            #=================================================================#

            toc = time() - tic
            print('Elapsed time for ' + ticker + ': ' + '{:.2f}'.format(toc) + ' sec')

        self.Data = Data
        self.Figure = Figure

    #=================================================================#

    def plotStrategy(self):
        """SETUP PLOTS"""

        Figure = {}
        ax = None
        if self.plotOpt:
            plt.close('all')

            Figure['Indicators'], self.indAxs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
            Figure['Value'], self.valAxs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

            Utility.setPlot(self.valAxs[0], logscale=False,
                            xlimits=[self.data.index.values[0], self.data.index.values[-1]])
            self.valAxs[0].set_yscale('log')

            for ax in self.valAxs:
                ax.grid()
                ax.grid(axis='x', linestyle='--')

        ###################################################################

        # PLOT 1
        if self.plotOpt:
            ax = self.indAxs[0]
            Utility.setPlot(ax, logscale=False, xlimits=[self.data.index.values[0], self.data.index.values[-1]])

            ax.set_title('Price')
            ax.plot(self.data['Close'], linewidth=0.5)
            # ax.plot(self.data['Smooth'],color='black',linewidth=1)
            
            ax.legend()

            ax = self.indAxs[1]
            Utility.setPlot(ax)

            ax = self.indAxs[2]
            Utility.setPlot(ax)

        return Figure

    #=================================================================#
    
    def executeIndicators(self):
        """EXECUTE INDICATORS FOR STRATEGY"""
        macd = self.inputs.macd
        atr = self.inputs.atr

        self.data['Smooth'] = Utility.smooth(list(self.data['Close']), avgType='simple', window=5, iterations=1)
        self.data['Regression'] = Indicators.regression(self.data['Close'], curveType='logarithmic')

        self.data['MACD'] = Indicators.macd(self.data,fast=macd.fast,slow=macd.slow,sig=macd.sig,avgType='simple')
        self.data['ATR'] = Indicators.atr(self.data,window=atr.window,avgType='exponential')
        self.data['SMA'] = Indicators.movingAverage(self.data,window=100,avgType='logarithmic',steepness=2,outputAll=True)#, colors=('tab:green'),ax=ax,plotOpt=self.plotOpt)
        # self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
        # self.data['SMA_Diff_Avg'] = Indicators.movingAverage(self.data['SMA_Diff'],window=10,avgType='exponential')# ax=ax,plotOpt=self.plotOpt)
        self.data['SMAe'] = Indicators.movingAverage(self.data,window=20,avgType='exponential',steepness=3)
        self.data['BB'] = Indicators.bollingerBands(self.data,window=20)
        self.data['RSI'] = Indicators.rsi(self.data,window=14,avgType='simple')
        self.data['AD'] = Indicators.accDist(self.data)
        self.data['VAP'] = Indicators.volumeAtPrice(self.data,numBins=25)
        self.data['AVG'] = Indicators.avgPrice(self.data['ATR'])

        [_, mean, high] = list(zip(*self.data['BB']))
        self.data['ATR_BB'] = [m + ((h - m) * 1.75) for m, h in zip(mean, high)]

        self.data['MACD_Avg'] = Indicators.avgPrice(pd.Series(list(zip(*self.data['MACD']))[0], index=self.data.index.values))#,colors='tab:blue', ax=ax, plotDev=True, plotOpt=self.plotOpt)

        Indicators.supportResistance(self.data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both')#,ax=ax,plotOpt=True)            
        Indicators.trend(self.data['Close'],direction='up')#,ax=ax,plotOpt=True)
        Indicators.extremaGaps(self.data['Smooth'],minDuration=10,minPerc=0.1)

    def executeStrategy(self):
        """CALCULATE INDICTORS"""

        self.strategy = {}
        
        self.strategy['ATR'] = Strategy.ATR(self.data,self.inputs.atr)
        self.strategy['ATR_BB'] = Strategy.ATR_BB(self.data,self.inputs.atr)
        self.strategy['BB'] = Strategy.BB(self.data,self.inputs.bb)
        self.strategy['MACD'] = Strategy.MACD(self.data,self.inputs.macd)
        self.strategy['MACD_Delta'] = Strategy.MACD_Delta(self.data,self.inputs.macd)
        self.strategy['RSI'] = Strategy.RSI(self.data,self.inputs.rsi)
        self.strategy['SMA'] = Strategy.SMA(self.data,self.inputs.sma)
        self.strategy['SMA_Crossing'] = Strategy.SMA_Crossing(self.data,self.inputs.sma)

    def executeOrders(self):
        """EXECUTE ORDERS"""

        self.orders = Orders(self.initialFunds, runNull=True)
        
        seekBuy = True
        seekSell = False

        buy = False
        sell = False
        for i in range(2, len(self.data)):
            date = self.data.index.values[i]

            if seekBuy:
############### [USER INPUT]: SET BUY CONSTRAINT
                if self.strategy['MACD_Delta'].buy(i):
                    buy = True
                    seekBuy = False

            if seekSell:
############### [USER INPUT]: SET BUY CONSTRAINT
                if self.strategy['MACD_Delta'].sell(i):
                    sell = True
                    seekSell = False

                """TRAILING STOP %"""
                # if (self.data['Close'].iloc[i-1] - orders.buyPrice) / orders.buyPrice < -0.0:
                #     sell = True
                #     seekSell = False

            if sell:
                self.orders.sell(self.data['Open'].iloc[i], date)

                sell = False
                seekBuy = True
            elif buy:
                self.orders.buy(self.data['Open'].iloc[i], date)

                buy = False
                seekSell = True
            else:
                self.orders.hold(self.data['Close'].iloc[i], date)

    def executeRetrospective(self):
        """ANALYZE ORDERS"""

        analyze = Analyze(self.orders)
        self.orders.info = analyze.analyze()

        stratFunds = self.orders.value
        nullFunds = self.orders.nullValue

        stratdates = [strat[0] for strat in stratFunds]
        stratvalue = [strat[1] for strat in stratFunds]

        nulldates = [null[0] for null in nullFunds]
        nullvalue = [null[1] for null in nullFunds]

        if self.plotOpt:
            dates = list(zip(*self.orders.value))[0]

            """NULL AND STRATEGY FUNDS"""
            self.valAxs[0].set_title('Value')
            self.valAxs[0].plot(nulldates, nullvalue, linestyle='dashed', label='Null')
            self.valAxs[0].plot(stratdates, stratvalue, label='Strategy')
            self.valAxs[0].legend()

            """DIFFERENCE TO NULL (S&P)"""
            diff = self.orders.info.indexDiff

            self.valAxs[1].set_title('% Diff to Null')
            self.valAxs[1].plot(dates, diff, color='tab:orange')

            """DRAWDOWN vs. NULL (S&P) DRAWDOWN"""
            drawdown = self.orders.info.drawdown
            nullDrawdown = self.orders.info.nullDrawdown

            self.valAxs[2].set_title('% Drawdown')
            self.valAxs[2].plot(dates, nullDrawdown, linestyle='dashed', label='Null')
            self.valAxs[2].plot(dates, drawdown, label='Strategy')
            self.valAxs[2].legend()

        print('Strategy Funds:  $' + '{:,.0f}'.format(stratvalue[-1]))
        print('Null Funds:      $' + '{:,.0f}'.format(nullvalue[-1]))

    #=================================================================#
    
    def setupInputs(self):
        combos = []
        for indicator in self.exploreIndicators:
            obj = getattr(self.inputs,indicator.lower())

            for field,_ in obj.__dict__.items():
                combos.append(getattr(obj,field))

        combinations = Utility.combinations(combos)

        # REMOVE NONCOMPLIANT DATA
        combinationsCleaned = []
        for ind,combo in enumerate(combinations):
            if not combo[1] < combo[0]:
                combinationsCleaned.append(combinations[ind])

        unzippedCombinations = list(zip(*combinationsCleaned))
        self.iters = len(unzippedCombinations[0])

        self.exploreInputs = []
        for i in range(self.iters):
            self.exploreInputs.append(self.Inputs())

            for indicator in self.exploreIndicators:
                obj = getattr(self.exploreInputs[i],indicator.lower())

                for j,(field,_) in enumerate(obj.__dict__.items()):
                    setattr(obj, field, unzippedCombinations[j][i])

    def setupExplore(self,ticker,path):
        outdir = 'Files/' + path
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        self.inputs = []

        # SINGLE PROCESSOR
        self.step = self.iters
        self.allResults = self.exploration(0)

        self.outputDf = Utility.obj2df(self.allResults)
        self.outputDf.to_csv('Files/' + path + '/' + ticker + '.csv',index=False)

        # MULTIPLE PROCESSOR
        # with Pool(processes=mp.cpu_count()) as pool:
        #     results = []
        #     self.step = int(np.ceil(self.iters / mp.cpu_count()))
        #     for n in range(0, self.iters, self.step):
        #         results.append(pool.apply_async(self.exploration, [n]))

        #     pool.close()
        #     pool.join()

        #     [result.wait() for result in results]
        #     results = [r.get() for r in results]

        #     allResults = []
        #     for result in results:
        #         allResults.extend(result)

        #     self.outputDf = Utility.obj2df(allResults)
        #     self.outputDf.to_csv('Files/' + path + '/' + ticker + '.csv',index=False)

    def exploration(self,segment):
        out = []

        for i in range(segment,self.step+segment):
            print(str(i+1),'/',str(self.step+segment))

            if i >= self.iters:
                break

            self.inputs = self.exploreInputs[i]

            self.executeIndicators()
            self.executeStrategy()
            self.executeOrders()
            self.executeRetrospective()

            indicators = []
            for indicator in self.exploreIndicators:
                indicators.append(getattr(self.inputs,indicator.lower()))

            outputs = self.Outputs(indicators)

########### [USER INPUT]: SET FIELDS
            outputs.fields.fast = self.inputs.macd.fast
            outputs.fields.slow = self.inputs.macd.slow
            outputs.fields.sig = self.inputs.macd.sig
            outputs.fields.delay = self.inputs.macd.delay

            outputs.numSell = self.orders.info.numSells
            outputs.exposure = self.orders.info.exposure
            outputs.drawdown = self.orders.info.maxDrawdown
            outputs.winLoss = self.orders.info.winLoss
            outputs.value = self.orders.value[-1][1]

            out.append(outputs)

        return out
    
    #=================================================================#

    def optimize(self):
        prices = self.data['Smooth']

        [peaks, troughs] = Utility.findExtrema(list(prices), endsOpt=False)
        extrema = np.asarray(sorted(np.concatenate((peaks, troughs)), key=lambda x: x[0]))

        optOrder = Orders(self.initialFunds)

        for i in range(len(self.data)):
            date = self.data.index.values[i]

            if i == 0:
                optOrder.hold(0, date)
            else:
                if i in extrema[:, 0]:
                    if prices.iloc[i] < prices.iloc[i - 1]:
                        optOrder.buy(self.data['Open'].iloc[i], date)
                    else:
                        if optOrder.shares > 0:
                            optOrder.sell(self.data['Open'].iloc[i], date)
                else:
                    optOrder.hold(self.data['Close'].iloc[i], date)

        optFunds = optOrder.value
        # print('Optimized Funds: $' + '{:,.0f}'.format(optFunds[-1][1]))

        return optFunds

    #=================================================================#
    #=================================================================#
    
    class Inputs:
        def __init__(self):
            self.atr = self.ATR()
            self.bb = self.BB()
            self.macd = self.MACD()
            self.rsi = self.RSI()
            self.sma = self.SMA()

        class ATR:
            def __init__(self):
                self.avg = 14
                self.std = 3
                self.window = 10

                # self.avg = [14]
                # self.std = [3]
                # self.window = [10]

        class BB:
            def __init__(self):
                pass

        class MACD:
            def __init__(self):
                self.delay = 1
                self.fast = 2
                self.slow = 5
                self.sig = 3

                # self.delay = np.arange(1,6)
                # self.fast = np.arange(2,21,2)
                # self.slow = np.arange(5,35,4)
                # self.sig = np.arange(3,19,3)

                # self.delay = np.arange(1,2)
                # self.fast = np.arange(2,7)
                # self.slow = np.arange(5,13,4)
                # self.sig = np.arange(7,15,2)
        
        class RSI:
            def __init__(self):
                pass

        class SMA:
            def __init__(self):
                pass

    class Outputs:
        def __init__(self,*objs):
            # EXTRACT INPUTS FROM LIST OF INPUTS     
            if len(objs) == 1:
                objs = objs[0]

            self.fields = self.Fields(objs)
            self.numSell = []
            self.exposure = []
            self.drawdown = []
            self.winLoss = []
            self.value = []

        class Fields:
            def __init__(self,*objs):
                # EXTRACT INPUTS FROM LIST OF INPUTS     
                if len(objs) == 1:
                    objs = objs[0]

                for obj in objs:                
                    for att, _ in obj.__dict__.items():
                        setattr(self,att,[])

#=================================================================#
#=================================================================#

if __name__ == '__main__':
    import logging, sys

    # warnings.filterwarnings("ignore")
    print('Running...')
    
    # logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    # logging.debug('A debug message!')

    backtest = Backtest()
    data = backtest.Data
    # order = backtest.order
    # output = backtest.outputsDf

    plt.show()
