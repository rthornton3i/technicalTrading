from utility import Utility
from indicators import Indicators

import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    
    class MACD:
        
        def __init__(self,data,inputs):
            tech = data['MACD']

            self.macdLine = [n[0] for n in tech]
            self.signalLine = [n[1] for n in tech]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            
            self.avg = data['MACD_Avg']
            
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
        
        def __init__(self,data,inputs):
            data = data['MACD']
            self.macdLine = [n[0] for n in data]
            self.signalLine = [n[1] for n in data]
            self.diff = [m-s for m,s in zip(self.macdLine,self.signalLine)]
            self.avg = None

            self.delay = inputs.delay
            
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
        
    class RSI:
        
        def __init__(self,data,inputs):
            self.data = data['RSI']
            
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
        
        def __init__(self,data,inputs):
            tech = data['SMA']
            
            tech = list(zip(*tech))
            self.data = tech[0]
            self.delta = tech[1]
            self.slope = tech[2]
            
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
        
        def __init__(self,data,inputs):
            tech = data['SMA']
            tech = list(zip(*tech))
            self.data = tech[0]
            
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
        
        def __init__(self,data,inputs):
            self.price = data['Close']
            self.data = data['BB']
            
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
        
        def __init__(self,data,inputs):
            self.atr = data.ATR
            
            self.avg = inputs.avg
            self.std = inputs.std
            
        def buy(self,i):
            buyOpt = False
            
            if self.atr[i-1] < self.avg + self.std:
                buyOpt = True
                    
            return buyOpt
        
        def sell(self,i):
            sellOpt = False
            
            if self.atr[i-1] > self.avg + self.std:
                sellOpt = True
                
            return sellOpt
        
    class ATR_BB:
        
        def __init__(self,data,inputs):
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