import numpy as np
from orders import Orders

class Analyze:

    def __init__(self, orders:Orders):
        self.orders = orders

        self.value = self.orders.value
        self.nullValue = self.orders.nullValue

    def analyze(self):
        info = self.orders.info
        info.avgEarn = np.mean(info.earnings)
        info.stdEarn = np.std(info.earnings)
        
        info.avgHold = np.mean(info.holdPeriod)
        info.exposure = np.sum(info.holdPeriod) / len(self.value)
        
        [info.drawdown, info.maxDrawdown] = self.calcDrawdown(self.value)
        info.winLoss = self.winLossRate(info.earnings)
        
        info.indexDiff = self.calcIndexDiff()
        [info.nullDrawdown, info.maxNullDrawdown] = self.calcDrawdown(self.nullValue)

        return info
            
    def calcIndexDiff(self):
        diff = []
        for val,nullVal in zip(self.value,self.nullValue):
            val = val[1]
            nullVal = nullVal[1]
            
            diff.append((val - nullVal) / val)
            
        return diff
    
    def calcDrawdown(self,value):
        localMax = 0
        
        drawdown = []
        maxDrawdown = 0
        
        for _,val in value:
            if val > localMax:
                localMax = val
            
            amt = -(localMax - val) / localMax
            drawdown.append(amt)
            if amt < maxDrawdown:
                maxDrawdown = amt
    
        return [drawdown,maxDrawdown]
    
    def winLossRate(self,earnings):
        wins = 0
        losses = 0
        for earn in earnings:
            if earn > 0:
                wins += 1
            else:
                losses += 1
        
        if losses > 0:
            winLoss = wins / losses
        else:
            winLoss = np.inf
        
        return winLoss