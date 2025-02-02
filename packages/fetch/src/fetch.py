from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.fundamentaldata import FundamentalData

from polygon import RESTClient

import numpy as np
import pandas as pd
from datetime import datetime as dt
from time import sleep

from pandas import DataFrame, Timestamp
from typing import Any, Optional

"""
REFERENCES:
    
1) Alpha Vantage Functions: https://github.com/RomelTorres/alpha_vantage/tree/develop/alpha_vantage
2) Alpha Vantage Documentation: https://www.alphavantage.co/documentation/
    
3) MPL Finance Tutorial: https://github.com/matplotlib/mplfinance#contents-and-tutorials
    
"""

class Fetch:

    def __init__(self,tickers:list[str],startDate:Timestamp,endDate:Optional[Timestamp]=None):
        self.tickers = tickers
        self.startDate = startDate
        
        if endDate is None:
            self.endDate = Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)
        else:
            self.endDate = endDate

    def fetch(self, fetchOpt:bool=False, writeOpt:bool=False) -> dict[str,DataFrame]:
        if fetchOpt:
            Data = self.getPrices()

            if writeOpt:
                cont = bool(input("Are you sure you'd like to overwrite data? (0/1)"))
                if cont:
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
    
    def getPrices(self) -> dict[str,DataFrame]:
        pass

class Fetch_Alpha(Fetch):
    
    def __init__(self,tickers:list[str],startDate:Timestamp,endDate:Optional[Timestamp]=None):
        self.key = 'WKSR42529JVOYXM9'
        
        self.ts = TimeSeries(self.key, output_format='pandas')
        self.ti = TechIndicators(self.key)
        self.fd = FundamentalData(self.key)

        super().__init__(tickers,startDate,endDate)
        
    def getPrices(self,increment='daily',splitOpt=False,divOpt=False) -> dict[str,DataFrame]:
        data:dict[str,DataFrame] = {}
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

################################################################################   

"""
REFERENCES:
    
    
"""

class Fetch_Polygon(Fetch):
    
    def __init__(self,tickers:list[str],startDate:Timestamp,endDate:Optional[Timestamp]=None):
        self.key = 'ez7NxSEphpNlHZFjsp6MSANJxs1UNOGv'
        
        self.client = RESTClient(self.key)
        
        super().__init__(tickers,startDate,endDate)
        
    def getPrices(self,increment:str='day') -> dict[str,DataFrame]:
        data:dict[str,DataFrame] = {}
        for ticker in self.tickers:
            while True:
                # data[ticker],_ = self.ts.get_daily_adjusted(symbol=ticker,outputsize='full')
                try:
                    aggs = self.client.get_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan=increment,
                        from_=self.startDate,
                        to=self.endDate,
                        adjusted=True
                    )
                    
                    data[ticker] = pd.DataFrame([vars(a) for a in aggs],columns=list(vars(aggs[0]).keys()))
                    data[ticker]['Date'] = pd.Series([dt.fromtimestamp(timestamp) for timestamp in data[ticker].loc[:,'timestamp']/1000])
                    data[ticker].drop(['timestamp','transactions','otc','vwap'], axis=1, inplace=True) 

                    print('Fetched: ' + ticker + '...')
                    break
                except ValueError:
                    print('WARNING: Call limit per minute exceeded, waiting 5 seconds...')
                    sleep(5)

            data[ticker].set_index('Date',inplace=True)
            data[ticker].columns = ['Open','High','Low','Close','Volume']
            
        return data