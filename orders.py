import math
import numpy as np

class Orders:
    
    def __init__(self,initialFunds,runNull=False):
        self.shares = 0
        self.cash = initialFunds
        self.value = []
        
        self.runNull = runNull
        self.nullShares = 0
        self.nullCash = initialFunds
        self.nullValue = []
        
        self.buyPrice = 0
        self.sellPrice = 0
        
        self.info = Info()
        
    def buy(self,buyPrice,date):
        self.buyPrice = buyPrice
        
        self.shares += math.floor(self.cash / buyPrice)
        self.cash -= self.shares * buyPrice
        self.value.append((date,self.cash + (self.shares * buyPrice)))
        
        self.calcNull(buyPrice,date)
        
        self.info.numBuys += 1
        self.info.buyDate = date
    
    def sell(self,sellPrice,date):
        self.sellPrice = sellPrice
        
        self.cash += self.shares * self.sellPrice
        self.shares = 0
        self.value.append((date,self.cash))
        
        self.calcNull(sellPrice,date)
        
        self.info.numSells += 1
        self.info.earnings.append((self.sellPrice - self.buyPrice) / self.buyPrice)
        self.info.holdPeriod.append(np.timedelta64(self.info.buyDate - date,'D').astype(int))
        
    def hold(self,curPrice,date):
        self.value.append((date,self.cash + (self.shares * curPrice)))
        
        self.calcNull(curPrice,date)
    
    def calcNull(self,price,date):
        if self.runNull:
            if self.nullShares == 0:
                self.nullShares += math.floor(self.nullCash / price)
                self.nullCash -= self.nullShares * price
                self.nullValue.append((date,self.nullCash + (self.nullShares * price)))
            else:
                self.nullValue.append((date,self.nullCash + (self.nullShares * price)))
            
    def analyze(self):
        self.info.avgEarn = np.mean(self.info.earnings)
        self.info.stdEarn = np.std(self.info.earnings)
        
        self.info.avgHold = np.mean(self.info.holdPeriod)
        self.info.exposure = np.sum(self.info.holdPeriod) / len(self.value)
        
        self.info.indexDiff = self.calcIndexDiff()
        [self.info.drawdown, self.info.maxDrawdown] = self.calcDrawdown(self.value)
        [self.info.nullDrawdown, self.info.maxNullDrawdown] = self.calcDrawdown(self.nullValue)
        self.info.winLoss = self.winLossRate()
    
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
            if amt > maxDrawdown:
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
            winLoss = 0
        
        return winLoss
        
class Info:
    
    def __init__(self):
        self.numBuys = 0
        self.numSells = 0
        
        self.earnings = []
        self.avgEarn = []
        self.stdEarn = []
        
        self.drawdown = []
        self.maxDrawdown = []
        self.nullDrawdown = []
        self.maxNullDrawdown = []
        
        self.winLoss = []
        
        self.buyDate = []
        self.holdPeriod = []
        self.avgHold = []
        self.exposure = []

        self.indexDiff = []   