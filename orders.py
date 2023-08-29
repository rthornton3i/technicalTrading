import math
import numpy as np

class Orders:
    
    def __init__(self,initialFunds):
        self.shares = 0
        self.cash = initialFunds
        self.value = []
        
        self.initBuy = False
        self.nullShares = 0
        self.nullCash = initialFunds
        self.nullValue = []
        
        self.buyPrice = 0
        self.sellPrice = 0

        self.info = self.Info()
        
    def buy(self,buyPrice,date):
        self.buyPrice = buyPrice
        
        self.shares += math.floor(self.cash / buyPrice)
        self.cash -= self.shares * buyPrice
        self.value.append((date,self.cash + (self.shares * buyPrice)))
        
        self.info.numBuys += 1
        self.info.buyDate = date
    
    def sell(self,sellPrice,date):
        self.sellPrice = sellPrice
        
        self.cash += self.shares * self.sellPrice
        self.shares = 0
        self.value.append((date,self.cash))
        
        self.info.numSells += 1
        self.info.earnings.append((self.sellPrice - self.buyPrice) / self.buyPrice)
        self.info.holdPeriod.append(np.timedelta64(date - self.info.buyDate,'D').astype(int))
        
    def hold(self,curPrice,date):
        self.value.append((date,self.cash + (self.shares * curPrice)))
    
    def calcNull(self,price,date,buy=False,sell=False,hold=False):
        if buy:
            self.nullShares += math.floor(self.nullCash / price)
            self.nullCash -= self.nullShares * price
            self.nullValue.append((date,self.nullCash + (self.nullShares * price)))
        
        if hold:
            self.nullValue.append((date,self.nullCash + (self.nullShares * price)))
        
        if sell:
            self.nullCash += self.nullShares * price
            self.nullShares = 0
            self.nullValue.append((date,self.nullCash))

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