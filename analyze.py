import numpy as np

class Analyze:

    def __init__(self, orders):
        self.orders = orders
        self.info = orders.info

    def analyze(self):
        self.info.avgEarn = np.mean(self.info.earnings)
        self.info.stdEarn = np.std(self.info.earnings)
        
        self.info.avgHold = np.mean(self.info.holdPeriod)
        self.info.exposure = np.sum(self.info.holdPeriod) / len(self.orders.value)
        
        [self.info.drawdown, self.info.maxDrawdown] = self.calcDrawdown(self.orders.value)
        self.info.winLoss = self.winLossRate()
        
        if self.orders.runNull:
            self.info.indexDiff = self.calcIndexDiff()
            [self.info.nullDrawdown, self.info.maxNullDrawdown] = self.calcDrawdown(self.orders.nullValue)

        return self.info
            
    def calcIndexDiff(self):
        diff = []
        for val,nullVal in zip(self.orders.value,self.orders.nullValue):
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
    
    def winLossRate(self):
        wins = 0
        losses = 0
        for earn in self.info.earnings:
            if earn > 0:
                wins += 1
            else:
                losses += 1
        
        if losses > 0:
            winLoss = wins / losses
        else:
            winLoss = np.inf
        
        return winLoss