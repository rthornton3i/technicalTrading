from utility import Utility
from strategy import Strategy

import numpy as np
import matplotlib.pyplot as plt

class Indicators:
    
    class MACD:
        
        def __init__(self,tech):
            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            
        def buy(self,i):
            buyOpt = False
            
            if self.macdLine[i-1] < 0 and self.signalLine[i-1] < 0:
                if self.macdLine[i-2] < self.signalLine[i-2] and self.macdLine[i-1] > self.signalLine[i-1]:
                    buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.macdLine[i-1] > 0 and self.signalLine[i-1] > 0:
                if self.macdLine[i-2] > self.signalLine[i-2] and self.macdLine[i-1] < self.signalLine[i-1]:
                    sellOpt = True
                    
            return sellOpt
        
    class MACD_Delta:
        
        def __init__(self,data):
            tech = Strategy.macd(data,fast=5,slow=10,sig=7,avgType='simple')
            
            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            
            self.backdays = 3
            
        def buy(self,i):
            buyOpt = False
            
            if np.isnan(self.macdLine[i-1]) or np.isnan(self.signalLine[i-1]):
                return buyOpt
        
            if self.macdLine[i-1] < 0 and self.signalLine[i-1] < 0:
                if self.diff[i-1] < 0:
                    for n in range(self.backdays,1,-1):
                        if abs(self.diff[i-n-1]) < abs(self.diff[i-n]):
                            buyOpt = True
                        else:
                            buyOpt = False
                            break
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if np.isnan(self.macdLine[i-1]) or np.isnan(self.signalLine[i-1]):
                return sellOpt
            
            if self.macdLine[i-1] > 0 and self.signalLine[i-1] > 0:
                if self.diff[i-1] > 0:
                    for n in range(self.backdays,1,-1):
                        if abs(self.diff[i-n-1]) < abs(self.diff[i-n]):
                            sellOpt = True
                        else:
                            sellOpt = False
                            break
                    
            return sellOpt
    
    class MACD_DeltaMod:
        
        def __init__(self,data):
            tech = Strategy.macd(data,fast=5,slow=10,sig=7,avgType='simple')
            
            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            
            self.backdays = 2
            
        def buy(self,i):
            buyOpt = False
            
            if self.macdLine[i-1] < 0 and self.signalLine[i-1] < 0:
                if self.diff[i-1] < 0:
                    for n in range(self.backdays,1,-1):
                        if abs(self.diff[i-n-1]) < abs(self.diff[i-n]):
                            buyOpt = True
                        else:
                            buyOpt = False
                            break
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.macdLine[i-2] > 0 and self.signalLine[i-2] > 0:
                if self.macdLine[i-2] > self.signalLine[i-2] and self.macdLine[i-1] < self.signalLine[i-1]:
                    sellOpt = True
                    
            return sellOpt
        
    class RSI:
        
        def __init__(self,tech):
            self.tech = tech
            
            [peaks,troughs]  = Utility.findExtrema(list(self.tech),endsOpt=False)            
            self.extrema = np.asarray(sorted(np.concatenate((peaks,troughs)),key=lambda x:x[0]))
            
        def buy(self,i):
            buyOpt = False
            
            if i in self.extrema[:,0]:
                if self.tech.iloc[i] < 30 and self.tech.iloc[i] < self.tech.iloc[i+1]:
                    buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if i in self.extrema[:,0]:
                if self.tech.iloc[i] > 70 and self.tech.iloc[i] > self.tech.iloc[i+1]:
                    sellOpt = True
                    
            return sellOpt
    
    class SMA:
        
        def __init__(self,tech):
            self.tech = tech
            
        def buy(self,i):
            buyOpt = False
            
            if self.tech.iloc[i-3] > self.tech.iloc[i-2] and self.tech.iloc[i-2] < self.tech.iloc[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.tech.iloc[i-3] < self.tech.iloc[i-2] and self.tech.iloc[i-2] > self.tech.iloc[i-1]:
                sellOpt = True
                    
            return sellOpt
    
    class SMA_Crossing:
        
        def __init__(self,settings):#data,tech):
            self.window = settings.window
            # self.openPrice = data['Open']
            # self.closePrice = data['Close']
            # self.tech = tech
            
        def buy(self,i):
            buyOpt = False
            
            if self.tech.iloc[i-3] > self.tech.iloc[i-2] and self.tech.iloc[i-2] < self.tech.iloc[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.tech.iloc[i-3] < self.tech.iloc[i-2] and self.tech.iloc[i-2] > self.tech.iloc[i-1]:
                sellOpt = True
                    
            return sellOpt
        
    class BB:
        
        def __init__(self,price,tech):
            self.price = price
            self.tech = tech
            
        def buy(self,i):
            buyOpt = False
            
            if self.price[i-1] < self.tech[i-1][0] and self.price[i] >= self.tech[i][0]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.price[i] > self.tech[i][1]:
                sellOpt = True
                
            return sellOpt
        
    class ATR:
        
        def __init__(self,data):
            self.atr = Strategy.atr(data,window=14,avgType='simple')
            self.avgAtr = Utility.smooth(self.atr,window=10,avgType='exponential',trailing=True)
            
            self.atrVal = 3
            # plt.figure(99)
            # plt.plot(self.avgAtr)
            
        def buy(self,i):
            buyOpt = False
            
            if self.atr[i] < self.atrVal:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.atr[i] > self.atrVal:
                sellOpt = True
                
            return sellOpt