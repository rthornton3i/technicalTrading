import math

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
        self.winLossRate()
        self.maxDrawdown()
    
    def winLossRate(self):
        pass
    
    def maxDrawdown(self):
        maxdrawdown = 0
        
class Info:
    
    def __init__(self):
        self.numBuys = 0
        self.numSells = 0
        self.earnings = []