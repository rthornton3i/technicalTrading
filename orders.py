import math
import numpy as np

class Orders:
    
    def __init__(self,initialFunds):
        self.shares = 0
        self.cash = initialFunds
        self.value = []
        
        self.buyPrice = 0
        self.sellPrice = 0
        
        self.info = Info()
        
    def buy(self,buyPrice,date):
        self.buyPrice = buyPrice
        
        self.shares += math.floor(self.cash / self.buyPrice)
        self.cash -= self.shares * self.buyPrice
        self.value.append((date,self.cash + (self.shares * buyPrice)))
        
        self.info.numBuys += 1
    
    def sell(self,sellPrice,date):
        self.sellPrice = sellPrice
        
        self.cash += self.shares * self.sellPrice
        self.shares = 0
        self.value.append((date,self.cash))
        
        self.info.numSells += 1
        self.info.earnings.append((self.sellPrice - self.buyPrice) / self.buyPrice)
        
    def hold(self,curPrice,date):
        self.value.append((date,self.cash + (self.shares * curPrice)))
        
    def analyze(self):
        self.info.avgEarn = np.mean(self.info.earnings)
        # self.info.drawdown = self.maxDrawdown()
        # self.info.winLoss = self.winLossRate()
    
    def maxDrawdown(self):
        localMax = 0
        localMin = 0
        drawdown = 0
        
        for _,val in self.value:
            if val > localMax:
                localMax = val
                localMin = val
                
            if val < localMin:
                localMin = val
                
            if localMax - localMin > drawdown:
                drawdown = localMax - localMin
    
        return drawdown
    
    def winLossRate(self):
        wins = 0
        losses = 0
        for earn in self.info.earnings:
            if earn > 0:
                wins += 1
            else:
                losses += 1
                
        winLoss = wins / losses
        
        return winLoss
        
class Info:
    
    def __init__(self):
        self.numBuys = 0
        self.numSells = 0
        
        self.earnings = []
        self.avgEarn = []
        
        self.drawdown = []
        self.winLoss = []
        