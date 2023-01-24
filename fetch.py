from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData

# from datapackage import Package

# package = Package('https://datahub.io/core/finance-vix/datapackage.json')

import numpy as np
import pandas as pd
from datetime import datetime as dt
from time import sleep

"""
REFERENCES:
    
1) Alpha Vantage Functions: https://github.com/RomelTorres/alpha_vantage/tree/develop/alpha_vantage
2) Alpha Vantage Documentation: https://www.alphavantage.co/documentation/
    
3) MPL Finance Tutorial: https://github.com/matplotlib/mplfinance#contents-and-tutorials
    
"""

class Fetch:
    
    def __init__(self,tickers,startDate,endDate=None):
        self.key = 'WKSR42529JVOYXM9'
        
        self.ts = TimeSeries(self.key, output_format='pandas')
        self.ti = TechIndicators(self.key)
        self.fd = FundamentalData(self.key)
        
        self.tickers = tickers
        self.startDate = startDate
        
        if endDate is None:
            self.endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)
        else:
            self.endDate = endDate
        
    def getPrices(self,splitOpt=False,divOpt=False):
        data = {}
        for ticker in self.tickers:
            while True:
                # data[ticker],_ = self.ts.get_daily(symbol=ticker,outputsize='full')
                try:
                    data[ticker],_ = self.ts.get_daily_adjusted(symbol=ticker,outputsize='full')
                    print('Fetched: ' + ticker + '...')
                    break
                except ValueError:
                    print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)
            
            data[ticker].index.name = 'Date'
            data[ticker].columns = ['Open','High','Low','Close','Adjusted','Volume','Dividend','Split']
            data[ticker] = data[ticker].reindex(index=data[ticker].index[::-1])
            
            data[ticker] = data[ticker].loc[self.startDate:self.endDate]
            
            if splitOpt:
                split = 1
                for date in data[ticker].index.values[::-1]:
                    for key in ['Open','High','Low','Close']:
                        data[ticker][key].loc[date] /= split
                        
                    split *= data[ticker]['Split'].loc[date]
                    
            if divOpt:
                div = 0
                for date in data[ticker].index.values[::-1]:
                    for key in ['Open','High','Low','Close']:
                        data[ticker][key].loc[date] -= div
                        
                    div += data[ticker]['Dividend'].loc[date]
            
        return data
    
    def getSearch(self,keywords):
        symbols,_ = self.ts.get_symbol_search(keywords=keywords)
            
        return symbols
    
    def getMovAvg(self,style='sma',period=20):
        data = {}
        for ticker in self.tickers:
            vals = {}
            if style.lower() == 'sma':
                while True:
                    try:
                        movAvg,_ = self.ti.get_sma(symbol=ticker,time_period=period)
                        break
                    except ValueError:
                        # print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                        sleep(5)
                        
                ident = 'SMA'
            elif style.lower() == 'ema':
                while True:
                    try:
                        movAvg,_ = self.ti.get_ema(symbol=ticker,time_period=period)
                        break
                    except ValueError:
                        # print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                        sleep(5)
                        
                ident = 'EMA'
            else:
                raise Exception('ERROR: Invalid style specified.')
            
            dates = [dt.strptime(key.split(' ')[0],"%Y-%m-%d") for key in movAvg.keys()]
            vals['Moving Average'] = [float(movAvg[key][ident]) for key in movAvg.keys()]
            
            data[ticker] = pd.DataFrame(vals,index=dates)
            data[ticker].index.name = 'Date'
            data[ticker] = data[ticker].reindex(index=data[ticker].index[::-1])
            
            data[ticker] = data[ticker].loc[self.startDate:]
        
        return data
    
    def getMACD(self,periods=[12,26,9],absrange=False):
        data = {}
        for ticker in self.tickers:
            vals = {}
            while True:
                try:
                    macd,_ = self.ti.get_macd(symbol=ticker,fastperiod=periods[0],slowperiod=periods[1],signalperiod=periods[2])
                    break
                except ValueError:
                    # print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)
                    
            dates = [dt.strptime(key.split(' ')[0],"%Y-%m-%d") for key in macd.keys()]
            vals['MACD']   = [float(macd[key]['MACD']) for key in macd.keys()]
            vals['Signal'] = [float(macd[key]['MACD_Signal']) for key in macd.keys()]
            vals['Hist']   = [float(macd[key]['MACD_Hist']) for key in macd.keys()]
                
            data[ticker] = pd.DataFrame(vals,index=dates)
            data[ticker].index.name = 'Date'
            data[ticker] = data[ticker].reindex(index=data[ticker].index[::-1])
            
            data[ticker] = data[ticker].loc[self.startDate:]
            
            if absrange:
                maxVal = max(abs(min(data[ticker]['MACD'])),max(data[ticker]['MACD']))
                data[ticker]['MACD']   = [val / maxVal for val in data[ticker]['MACD']]
                data[ticker]['Signal'] = [val / maxVal for val in data[ticker]['Signal']]
                data[ticker]['Hist']   = [m-s for m,s in zip(data[ticker]['MACD'],data[ticker]['Signal'])]
        
        return data
    
    def getRSI(self,period=20):
        data = {}
        for ticker in self.tickers:
            vals = {}
            while True:
                try:
                    rsi,_ = self.ti.get_rsi(symbol=ticker,time_period=period)
                    break
                except ValueError:
                    # print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)
            
            dates = [dt.strptime(key.split(' ')[0],"%Y-%m-%d") for key in rsi.keys()]
            vals['RSI'] = [float(rsi[key]['RSI']) for key in rsi.keys()]
            
            data[ticker] = pd.DataFrame(vals,index=dates)
            data[ticker].index.name = 'Date'
            data[ticker] = data[ticker].reindex(index=data[ticker].index[::-1])
            
            data[ticker] = data[ticker].loc[self.startDate:]
        
        return data