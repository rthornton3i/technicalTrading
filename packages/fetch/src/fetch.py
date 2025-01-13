from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData

from polygon import RESTClient

# from datapackage import Package

# package = Package('https://datahub.io/core/finance-vix/datapackage.json')

import numpy as np
import pandas as pd
from datetime import datetime as dt
from time import sleep

from pandas import DataFrame

"""
REFERENCES:
    
1) Alpha Vantage Functions: https://github.com/RomelTorres/alpha_vantage/tree/develop/alpha_vantage
2) Alpha Vantage Documentation: https://www.alphavantage.co/documentation/
    
3) MPL Finance Tutorial: https://github.com/matplotlib/mplfinance#contents-and-tutorials
    
"""

class Fetch_Alpha:
    
    def __init__(self,tickers:list[str],startDate:pd.Timestamp,endDate:pd.Timestamp=None):
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

    def fetch(self, fetchOpt:bool=False, writeOpt:bool=False) -> dict[str,DataFrame]:
        if fetchOpt:
            Data = self.getPrices()

            if writeOpt:
                with pd.ExcelWriter('inputs/data.xlsx') as writer:
                    for ticker in self.tickers:
                        data = Data[ticker]
                        data.to_excel(writer, sheet_name=ticker)
        else:
            Data = {}
            for ticker in self.tickers:
                if not fetchOpt:
                    data = pd.read_excel('inputs/data.xlsx', sheet_name=ticker, index_col=0)
                    data = data.loc[self.startDate:self.endDate]
                    Data[ticker] = data

        return Data
        
    def getPrices(self,increment='daily',splitOpt=False,divOpt=False):
        data = {}
        for ticker in self.tickers:
            while True:
                # data[ticker],_ = self.ts.get_daily_adjusted(symbol=ticker,outputsize='full')
                try:
                    if increment == 'daily':
                        data[ticker],_ = self.ts.get_daily(symbol=ticker,outputsize='full')
                    elif increment == 'weekly':
                        data[ticker],_ = self.ts.get_weekly(symbol=ticker)
                    elif increment == 'monthly':
                        data[ticker],_ = self.ts.get_monthly(symbol=ticker)

                    print('Fetched: ' + ticker + '...')
                    break
                except ValueError:
                    print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)
            
            data[ticker].index.name = 'Date'
            # data[ticker].columns = ['Open','High','Low','Close','Adjusted','Volume','Dividend','Split']
            data[ticker].columns = ['Open','High','Low','Close','Volume']
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
    
"""
REFERENCES:
    
    
"""

class Fetch_Polygon:
    
    def __init__(self):#,tickers:list[str],startDate:pd.Timestamp,endDate:pd.Timestamp=None):
        self.key = 'ez7NxSEphpNlHZFjsp6MSANJxs1UNOGv'
        
        client = RESTClient(self.key)

        aggs = []
        for a in client.list_aggs(
            "AAPL",
            1,
            "minute",
            "2022-01-01",
            "2023-02-03",
            limit=50000,
        ):
            aggs.append(a)

        print(aggs)
        
        # self.tickers = tickers
        # self.startDate = startDate
        
        # if endDate is None:
        #     self.endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)
        # else:
        #     self.endDate = endDate

    def fetch(self, fetchOpt:bool=False, writeOpt:bool=False) -> dict[str,DataFrame]:
        if fetchOpt:
            Data = self.getPrices()

            if writeOpt:
                with pd.ExcelWriter('inputs/data.xlsx') as writer:
                    for ticker in self.tickers:
                        data = Data[ticker]
                        data.to_excel(writer, sheet_name=ticker)
        else:
            Data = {}
            for ticker in self.tickers:
                if not fetchOpt:
                    data = pd.read_excel('inputs/data.xlsx', sheet_name=ticker, index_col=0)
                    data = data.loc[self.startDate:self.endDate]
                    Data[ticker] = data

        return Data
        
    def getPrices(self,increment='daily',splitOpt=False,divOpt=False):
        data = {}
        for ticker in self.tickers:
            while True:
                # data[ticker],_ = self.ts.get_daily_adjusted(symbol=ticker,outputsize='full')
                try:
                    if increment == 'daily':
                        data[ticker],_ = self.ts.get_daily(symbol=ticker,outputsize='full')
                    elif increment == 'weekly':
                        data[ticker],_ = self.ts.get_weekly(symbol=ticker)
                    elif increment == 'monthly':
                        data[ticker],_ = self.ts.get_monthly(symbol=ticker)

                    print('Fetched: ' + ticker + '...')
                    break
                except ValueError:
                    print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)
            
            data[ticker].index.name = 'Date'
            # data[ticker].columns = ['Open','High','Low','Close','Adjusted','Volume','Dividend','Split']
            data[ticker].columns = ['Open','High','Low','Close','Volume']
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
    

fetch = Fetch_Polygon()