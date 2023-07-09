from utility import Utility
from indicators import Indicators

import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    
    class MACD:
        
        def __init__(self,tech,avg=None):
            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            
            if avg is None:
                self.avg = np.zeros((1,len(self.macdLine)))
            else:
                self.avg = avg
            
        def buy(self,i):
            buyOpt = False
            
            if self.macdLine[i-1] < self.avg[i-1] and self.signalLine[i-1] < self.avg[i-1]:
                if self.diff[i-2] < 0 and self.diff[i-1] > 0:
                    buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.macdLine[i-1] > self.avg[i-1] and self.signalLine[i-1] > self.avg[i-1]:
                if self.diff[i-2] > 0 and self.diff[i-1] < 0:
                    sellOpt = True
                    
            return sellOpt
        
    class MACD_Delta:
        
        def __init__(self,data,info):
            tech = data['MACD']
            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            self.avg = None

            self.delay = info['delay']
            
            if self.avg is None:
                self.avg = np.zeros((len(self.macdLine),1))
            else:
                self.avg = data['MACD_avg']
            
        def buy(self,i):
            buyOpt = False
            
            if np.isnan(self.macdLine[i-1]) or np.isnan(self.signalLine[i-1]):
                return buyOpt
        
            if self.macdLine[i-1] < self.avg[i-1] and self.signalLine[i-1] < self.avg[i-1]:
                for n in range(1,self.delay+1):
                    if abs(self.diff[i-n]) < abs(self.diff[i-n-1]):
                        continue
                    else:
                        return buyOpt
                    
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if np.isnan(self.macdLine[i-1]) or np.isnan(self.signalLine[i-1]):
                return sellOpt
            
            if self.macdLine[i-1] > self.avg[i-1] and self.signalLine[i-1] > self.avg[i-1]:
                for n in range(1,self.delay+1):
                    if abs(self.diff[i-n]) < abs(self.diff[i-n-1]):
                        continue
                    else:
                        return sellOpt
                    
                sellOpt = True
                    
            return sellOpt
    
    class SMA_Delta:
        
        def __init__(self,slow,fast,diff,diffAvg,info):
            self.slow = {}
            self.fast = {}
            
            tech = list(zip(*slow))
            self.slow['tech'] = tech[0]
            self.slow['delta'] = tech[1]
            self.slow['slope'] = tech[2]
            
            tech = list(zip(*fast))
            self.fast['tech'] = tech[0]
            self.fast['delta'] = tech[1]
            self.fast['slope'] = tech[2]
            
            self.diff = diff
            self.diffAvg = diffAvg
            self.avg = info['avg']
            self.std = info['std']
            
            self.inStd = False
            
        def buy(self,i):
            buyOpt = False
            
            slopeDown = self.diffAvg[i-2] > self.diffAvg[i-1]
            belowDev = self.diffAvg[i-1] < self.avg - self.std
            
            if not slopeDown and not belowDev:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            slopeDown = self.diffAvg[i-2] > self.diffAvg[i-1]
            belowDev = self.diffAvg[i-1] < self.avg - self.std
            
            if slopeDown and belowDev:
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
            tech = list(zip(*tech))
            self.tech = tech[0]
            self.delta = tech[1]
            self.slope = tech[2]
            
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
        
        def __init__(self,tech,info):
            self.tech = tech
            
            self.avg = info['avg']
            self.std = info['std']
            
        def buy(self,i):
            buyOpt = False
            
            if self.tech[i-1] < self.avg + self.std:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.tech[i-1] > self.avg + self.std:
                sellOpt = True
                
            return sellOpt
        
    class ATR_BB:
        
        def __init__(self,data):
            self.tech = data['ATR']
            self.bb = data['ATR_BB']
            
        def buy(self,i):
            buyOpt = False
            
            if self.tech[i-1] < self.bb[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.tech[i-1] > self.bb[i-1]:
                sellOpt = True
                
            return sellOpt