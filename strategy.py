from utility import Utility
from indicators import Indicators

import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    
    class MACD:
        
        def __init__(self,data,avg=None):
            self.macdLine = [n[0] for n in data]
            self.signalLine = [n[1] for n in data]
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
            data = data['MACD']
            self.macdLine = [n[0] for n in data]
            self.signalLine = [n[1] for n in data]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            self.avg = None

            self.delay = info.delay
            
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
            
            data = list(zip(*slow))
            self.slow['data'] = data[0]
            self.slow['delta'] = data[1]
            self.slow['slope'] = data[2]
            
            data = list(zip(*fast))
            self.fast['data'] = data[0]
            self.fast['delta'] = data[1]
            self.fast['slope'] = data[2]
            
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
        
        def __init__(self,data):
            self.data = data
            
            [peaks,troughs]  = Utility.findExtrema(list(self.data),endsOpt=False)            
            self.extrema = np.asarray(sorted(np.concatenate((peaks,troughs)),key=lambda x:x[0]))
            
        def buy(self,i):
            buyOpt = False
            
            if i in self.extrema[:,0]:
                if self.data.iloc[i] < 30 and self.data.iloc[i] < self.data.iloc[i+1]:
                    buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if i in self.extrema[:,0]:
                if self.data.iloc[i] > 70 and self.data.iloc[i] > self.data.iloc[i+1]:
                    sellOpt = True
                    
            return sellOpt
    
    class SMA:
        
        def __init__(self,data):
            data = list(zip(*data))
            self.data = data[0]
            self.delta = data[1]
            self.slope = data[2]
            
        def buy(self,i):
            buyOpt = False
            
            if self.data.iloc[i-3] > self.data.iloc[i-2] and self.data.iloc[i-2] < self.data.iloc[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.data.iloc[i-3] < self.data.iloc[i-2] and self.data.iloc[i-2] > self.data.iloc[i-1]:
                sellOpt = True
                    
            return sellOpt
    
    class SMA_Crossing:
        
        def __init__(self,settings):#data,data):
            self.window = settings.window
            # self.openPrice = data['Open']
            # self.closePrice = data['Close']
            # self.data = data
            
        def buy(self,i):
            buyOpt = False
            
            if self.data.iloc[i-3] > self.data.iloc[i-2] and self.data.iloc[i-2] < self.data.iloc[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.data.iloc[i-3] < self.data.iloc[i-2] and self.data.iloc[i-2] > self.data.iloc[i-1]:
                sellOpt = True
                    
            return sellOpt
        
    class BB:
        
        def __init__(self,price,data):
            self.price = price
            self.data = data
            
        def buy(self,i):
            buyOpt = False
            
            if self.price[i-1] < self.data[i-1][0] and self.price[i] >= self.data[i][0]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.price[i] > self.data[i][1]:
                sellOpt = True
                
            return sellOpt
        
    class ATR:
        
        def __init__(self,data,info):
            self.data = data.ATR
            
            self.avg = info.avg
            self.std = info.std
            
        def buy(self,i):
            buyOpt = False
            
            if self.data[i-1] < self.avg + self.std:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.data[i-1] > self.avg + self.std:
                sellOpt = True
                
            return sellOpt
        
    class ATR_BB:
        
        def __init__(self,data):
            self.data = data['ATR']
            self.bb = data['ATR_BB']
            
        def buy(self,i):
            buyOpt = False
            
            if self.data[i-1] < self.bb[i-1]:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.data[i-1] > self.bb[i-1]:
                sellOpt = True
                
            return sellOpt