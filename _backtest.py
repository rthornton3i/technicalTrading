from fetch import Fetch
from indicators import Indicators
from utility import Utility
from orders import Orders
from strategy import Strategy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt
from time import time
import math
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

        # self.startDate = pd.Timestamp(year=2020,month=3,day=23)
        # self.startDate = pd.Timestamp(year=2006, month=1, day=1)
        # self.startDate = pd.Timestamp(year=2019,month=12,day=11)
        self.startDate = pd.Timestamp(year=dt.now().year-10,month=dt.now().month,day=dt.now().day)

        # self.endDate = pd.Timestamp(year=2012,month=1,day=1)
        self.endDate = pd.Timestamp(year=dt.now().year, month=dt.now().month, day=dt.now().day)

        self.initialFunds = 10000

        # self.test()
        self.run(True, True)

    ###########################################################################
    def test(self):
        stockFetch = Fetch(self.tickers, self.startDate)
        self.Data = stockFetch.getPrices()

    ###########################################################################
    def run(self, fetchOpt, writeOpt):
        if fetchOpt:
            stockFetch = Fetch(self.tickers, self.startDate, self.endDate)
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
                self.data = pd.read_excel('Files/data.xlsx', sheet_name=ticker, index_col=0)
                self.data = self.data.loc[self.startDate:self.endDate]
                Data[ticker] = self.data
            else:
                self.data = Data[ticker]

            self.info = {'delay': [],
                         'sma': {'avg': [],
                                 'std': []},
                         'atr': {'avg': [],
                                 'std': []}}

            # data['SMA20'] = Indicators.movingAverage(data,window=10,avgType='simple')
            # data['SMA50'] = Indicators.movingAverage(data,window=50,avgType='simple')
            # data['SMA200'] = Indicators.movingAverage(data,window=200,avgType='simple')
            # data['SMAe'] = Indicators.movingAverage(data,window=20,avgType='exponential',steepness=3)
            # data['BB'] = Indicators.bollingerBands(data,window=20)
            # data['RSI'] = Indicators.rsi(data,window=14,avgType='simple')
            # data['ATR'] = Indicators.atr(data,window=14,avgType='exponential')
            # data['MACD'] = Indicators.macd(data,fast=5,slow=10,sig=7,avgType='simple')
            # data['AD'] = Indicators.accDist(data)
            # data['VAP'] = Indicators.volumeAtPrice(data,numBins=25)
            # data['AVG'] = Indicators.avgPrice(data['ATR'])

            # Indicators.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
            # Indicators.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
            # Indicators.extremaGaps(data['Smooth'],minDuration=10,minPerc=0.1)

            # self.data['Smooth'] = Utility.smooth(list(self.data['Close']), avgType='simple', window=5, iterations=1)
            # self.data['Regression'] = Indicators.regression(self.data['Close'], curveType='logarithmic')

            # toc = time() - tic
            # print('Elapsed time for ' + ticker + ': ' + '{:.2f}'.format(toc) + ' sec')

            ###################################################################
            # Indicators funds
            # plotOpt = True
            # Figure = self.plotStrategy(plotOpt)
            # self.executeStrategy(plotOpt)

            ###################################################################
            # self.inputs = {'stdev': []}

            # self.setupExplore(ticker,'MACD_ATR_BBdiff')
            # self.allResults = self.exploration(stdev=[0.5,0.75,1,1.25,1.4,1.5,1.6,1.75])

            # Info[ticker] = self.info

        self.Data = Data
        # self.Info = Info
        # self.Params = Params
        # self.Funds = Funds
        # self.Figure = Figure

    ###########################################################################
    def setupInputs(self):
        inputs = self.inputs

        # fastRng = list(range(3,14,2))
        # sigRng = list(range(3,12,2))
        # delayRng = list(range(1,5))
        # slowRng = np.unique([int(np.round(n/10 * fast)) for n in range(11,22,2)])

        for stdev in [0.5, 0.75, 1, 1.25, 1.4, 1.5, 1.6, 1.75, 2, 2.25, 2.5]:
            var = [stdev]
            for i, attr in enumerate(inputs):
                inputs[attr].append(var[i])

        dfIn = pd.DataFrame.from_dict(inputs)

        return dfIn

    ###########################################################################
    def setupExplore(self, ticker, path):
        tic = time()

        outdir = 'Files/' + path
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        dfIn = self.setupInputs()

        start = 0
        end = len(dfIn)
        buffer = 0

        dfStep = 1000
        for i, x in enumerate(range(start, end, dfStep)):
            if x > len(dfIn) - dfStep:
                df = dfIn.iloc[x:]
            else:
                df = dfIn.iloc[x:x + dfStep]
            # df = dfIn

            with Pool(processes=mp.cpu_count()) as pool:
                results = []
                stepSize = int(np.ceil(len(df) / mp.cpu_count()))
                for n in range(0, len(df), stepSize):
                    if n > len(df) - stepSize:
                        ins = df.iloc[n:]
                    else:
                        ins = df.iloc[n:n + stepSize]

                    keywords = {}
                    for attr in self.inputs:
                        keywords[attr] = getattr(ins, attr).tolist()

                    results.append(pool.apply_async(self.exploration, kwds=keywords))

                pool.close()
                pool.join()

            [result.wait() for result in results]
            allResults = [r.get() for r in results]

            self.outputDf = Utility.dicts2df(allResults)
            self.outputDf.to_csv('Files/' + path + '/' + ticker + '_' + str(i + buffer) + '.csv')

        toc = time() - tic
        print('Runtime: ' + str(toc))

        ###########################################################################

    def exploration(self, **kwargs):
        outputs = {'stdev': []}

        outputs['numSell'] = []
        outputs['exposure'] = []
        outputs['drawdown'] = []
        outputs['winLoss'] = []
        outputs['value'] = []
        outputs['zzz'] = []

        for stdev in kwargs['stdev']:
            # zip(kwargs['stdev']):

            # self.data['SMAs'] = Indicators.movingAverage(self.data['Close'],
            #                                            window=windowSlow,
            #                                            avgType=avgType,
            #                                            steepness=steepness,
            #                                            outputAll=True)

            # self.data['SMAf'] = Indicators.movingAverage(self.data['Close'],
            #                                            window=windowFast,
            #                                            avgType=avgType,
            #                                            steepness=steepness,
            #                                            outputAll=True)

            # self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
            # self.data['SMA_Diff_Avg'] = Indicators.movingAverage(self.data['SMA_Diff'],
            #                                                    window=10,
            #                                                    avgType='exponential')
            # [avg,std] = Indicators.avgPrice(self.data['SMA_Diff'],outputAll=True)

            # self.info['sma']['avg'] = avg[0]
            # self.info['sma']['std'] = std*stdev

            self.data['MACD'] = Indicators.macd(self.data['Close'],
                                              fast=5, slow=8, sig=3,
                                              avgType='logarithmic')

            self.data['MACD_avg'] = Indicators.avgPrice(
                pd.Series(list(zip(*self.data['MACD']))[0], index=self.data.index.values))

            self.info['delay'] = 1

            self.data['ATR'] = Indicators.atr(self.data,
                                            window=12,
                                            avgType='logarithmic')

            self.data['BB'] = Indicators.bollingerBands(self.data['ATR'],
                                                      window=125,
                                                      avgType='median')

            [_, mean, high] = list(zip(*self.data['BB']))
            self.data['ATR_BB'] = [m + ((h - m) * stdev) for m, h in zip(mean, high)]

            order = self.strategy()

            v = [stdev]
            var = [stdev,
                   order.info.numSells,
                   order.info.exposure,
                   order.info.maxDrawdown,
                   order.info.winLoss,
                   order.value[-1][1],
                   '-'.join([str(n) for n in v])]

            for i, attr in enumerate(outputs):
                outputs[attr].append(var[i])

        return outputs

    def executeStrategy(self, plotOpt):
        self.order = self.strategy(runNull=True)

        stratFunds = self.order.value
        nullFunds = self.order.nullValue

        stratdates = [strat[0] for strat in stratFunds]
        stratvalue = [strat[1] for strat in stratFunds]

        nulldates = [null[0] for null in nullFunds]
        nullvalue = [null[1] for null in nullFunds]

        if plotOpt:
            self.valAxs[0].set_title('Value')
            self.valAxs[0].plot(nulldates, nullvalue, linestyle='dashed', label='Null')
            self.valAxs[0].plot(stratdates, stratvalue, label='Indicators')
            self.valAxs[0].legend()

        dates = list(zip(*self.order.value))[0]

        diff = self.order.info.indexDiff
        if plotOpt:
            self.valAxs[1].set_title('% Diff to Null')
            self.valAxs[1].plot(dates, diff, color='tab:orange')

        drawdown = self.order.info.drawdown
        nullDrawdown = self.order.info.nullDrawdown
        if plotOpt:
            self.valAxs[2].set_title('% Drawdown')
            self.valAxs[2].plot(dates, nullDrawdown, linestyle='dashed', label='Null')
            self.valAxs[2].plot(dates, drawdown, label='Indicators')
            self.valAxs[2].legend()

        print('Indicators Funds:  $' + '{:,.0f}'.format(stratvalue[-1]))
        print('Null Funds:      $' + '{:,.0f}'.format(nullvalue[-1]))

    ###########################################################################
    def strategy(self, runNull=False):
        # Set buy/sell strategies
        indicator = {}
        # indicator['BB'] = Strategy.BB(data['Close'],data['BB'])
        # indicator['MACD'] = Strategy.MACD(data['MACD'],data['MACD_avg'])
        # indicator['SMA'] = Strategy.SMA(self.data['SMA'])
        # indicator['ATR'] = Strategy.ATR(data)

        # indicator['SMA'] = Strategy.SMA_Delta(self.data['SMAs'],
        #                                         self.data['SMAf'],
        #                                         self.data['SMA_Diff'],
        #                                         self.data['SMA_Diff_Avg'],
        #                                         self.info['sma'])
        indicator['MACD'] = Strategy.MACD_Delta(self.data,self.info)
        indicator['ATR'] = Strategy.ATR_BB(self.data)

        # Initiate orders
        order = Orders(self.initialFunds, runNull=runNull)

        seekBuy = True
        seekSell = False

        buy = False
        sell = False
        for i in range(2, len(self.data)):
            date = self.data.index.values[i]

            if seekBuy:
                if indicator['MACD'].buy(i) and indicator['ATR'].buy(i):
                    buy = True
                    seekBuy = False

            if seekSell:
                if indicator['MACD'].sell(i) or indicator['ATR'].sell(i):
                    sell = True
                    seekSell = False

                # if (self.data['Close'].iloc[i-1] - order.buyPrice) / order.buyPrice < -0.0:
                #     sell = True
                #     seekSell = False

            if sell:
                order.sell(self.data['Open'].iloc[i], date)

                sell = False
                seekBuy = True
            elif buy:
                order.buy(self.data['Open'].iloc[i], date)

                buy = False
                seekSell = True
            else:
                order.hold(self.data['Close'].iloc[i], date)

            # print(date[:10] + ' --- ' + '{:,.0f}'.format(order.value[-1][1]))

        order.info.analyze()

        return order

    def plotStrategy(self, plotOpt=True):
        # Setup plots
        Figure = {}
        ax = None
        if plotOpt:
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
        if plotOpt:
            ax = self.indAxs[0]
            Utility.setPlot(ax, logscale=False, xlimits=[self.data.index.values[0], self.data.index.values[-1]])

            ax.set_title('Price')
            ax.plot(self.data['Close'], linewidth=0.5)
            # ax.plot(data['Smooth'],color='black',linewidth=1)

        # self.data['SMAs'] = Indicators.movingAverage(self.data['Close'],
        #                                           window=100,
        #                                           avgType='logarithmic',
        #                                           steepness=2,
        #                                           outputAll=True,
        #                                           colors=('tab:green'),ax=ax,plotOpt=plotOpt)

        # self.data['SMAf'] = Indicators.movingAverage(self.data['Close'],
        #                                           window=20,
        #                                           avgType='logarithmic',
        #                                           steepness=2,
        #                                           outputAll=True,
        #                                           colors=('tab:orange'),ax=ax,plotOpt=plotOpt)

        if plotOpt:
            pass
            # ax.legend()

        # Indicators.supportResistance(data['Smooth'],thresh=0.05,minNum=3,minDuration=10,style='both',ax=ax,plotOpt=True)            
        # Indicators.trend(data['Close'],direction='up',ax=ax,plotOpt=True)
        # data['BB'] = Indicators.bollingerBands(data['Close'],window=20,avgType='simple',ax=ax,plotOpt=True)
        # data['VAP'] = Indicators.volumeAtPrice(data,numBins=15,volumeType='all',integrated=True,ax=ax,plotOpt=True)

        ##################################################################

        # PLOT 2
        if plotOpt:
            ax = self.indAxs[1]
            Utility.setPlot(ax)

            ax.set_title('ATR')

        # self.data['SMA_Diff'] = [n / self.data['Close'].iloc[i] for i,n in enumerate(pd.Series(list(zip(*self.data['SMAf']))[0]) - pd.Series(list(zip(*self.data['SMAs']))[0]))]
        # self.data['SMA_Diff_Avg'] = Indicators.movingAverage(self.data['SMA_Diff'],
        #                                                    window=10,
        #                                                    avgType='exponential',
        #                                                    ax=ax,plotOpt=plotOpt)

        # [avg,std] = Indicators.avgPrice(self.data['SMA_Diff'],outputAll=True,ax=ax,plotDev=True,plotOpt=plotOpt)

        # self.info['sma']['avg'] = avg[0]
        # self.info['sma']['std'] = std*1.8

        # Slow: 0.857
        # Medium: 1.283
        # Fast: 2.416
        # Covid: 5.1
        self.data['ATR'] = Indicators.atr(self.data,
                                        window=12,
                                        avgType='logarithmic',
                                        ax=ax, plotOpt=plotOpt)

        print('Mean ATR: ' + str(np.nanmean(self.data['ATR'])))

        self.data['BB'] = Indicators.bollingerBands(self.data['ATR'],
                                                  window=125,
                                                  avgType='median',
                                                  ax=ax, plotOpt=plotOpt)

        [_, mean, high] = list(zip(*self.data['BB']))
        self.data['ATR_BB'] = [m + ((h - m) * 1.75) for m, h in zip(mean, high)]

        if plotOpt:
            ax.plot(self.data.index.values, self.data['ATR_BB'])

            ##################################################################

        # PLOT 3
        if plotOpt:
            ax = self.indAxs[2]
            Utility.setPlot(ax)

            ax.set_title('MACD')

        # Overall: 5-8-3-1
        # Slow: 9-17-3-3 (9-14-7-4)
        # Medium: 5-10-5-3 (5-8-5-3)
        # Fast: 13-14-3-2 (13-17-7-3)
        # Covid: 3-6-3-1 (3-4-5-1)
        self.data['MACD'] = Indicators.macd(self.data['Close'],
                                          fast=13, slow=17, sig=7,
                                          avgType='logarithmic',
                                          ax=ax, plotOpt=plotOpt)

        self.data['MACD_avg'] = Indicators.avgPrice(
            pd.Series(list(zip(*self.data['MACD']))[0], index=self.data.index.values),
            colors='tab:blue', ax=ax, plotDev=True, plotOpt=plotOpt)

        self.info['delay'] = 3

        if plotOpt:
            ax.legend()

        return Figure

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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Running...')

    backtest = Backtest()
    data = backtest.Data
    # order = backtest.order
    # output = backtest.outputsDf

    plt.show()
