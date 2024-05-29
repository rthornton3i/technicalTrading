from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import os
import datetime

class Scrape:
    
    def __init__(self):
        pass
    
    def scrape(self,stock=None,url=None):
        if url is None:
            url = 'https://finance.yahoo.com/quote/' + stock + '?p=' + stock
            
        html = urlopen(url)
        soup = bs(html, 'lxml')
        code = soup.prettify()
        
        fileName = 'stock.txt'
        
        file = open(fileName,'w+')
        try:
            file.write(code)    
        except UnicodeEncodeError:
            file = open(fileName,'w+',encoding="utf-8")
            file.write(code)   
        file.close()    
        
        file = open(fileName,'r')
        self.lines = file.readlines()
        file.close()
        
        # os.remove(fileName)
        
    def attributeSearch(self,searchIndex,index):
        def lineSearch():
            lineIndex = 0
            for line in self.lines:
                if searchIndex in line:
                    searchResult = True
                    break
                else:
                    searchResult = False
                    
                lineIndex += 1
            
            lineIndex += index
            
            if searchResult == False:
                lineIndex = 0
            
            return lineIndex
        
        lineIndex = lineSearch()
    
        try:       
            att = float(self.lines[lineIndex].strip().replace(',',''))
        except ValueError:
            att = self.lines[lineIndex].strip().replace(',','')
            if '<!DOCTYPE html>' in att or 'N/A' in att.upper():
                att = 'N/a'
        
        return att

    def scrapePrice(self,stock):
        self.scrape(stock)
        
        # Ticker price
        tickerSearch = '<span class="Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)"'
        priceAtt = self.attributeSearch(tickerSearch,1)
        
        print(priceAtt)
    
    def scrapeVIX(self):
        yesterday = datetime.datetime.today() - datetime.timedelta(days = 1)
        date = (yesterday-datetime.datetime(1970,1,1)).total_seconds()
        
        url = 'https://finance.yahoo.com/quote/%5EVIX/history?period1=631238400&period2=' + str(date) + '&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
        self.scrape(url=url)
        
        # # Beta
        # betaSearch = 'BETA_3Y-value'
        # betaAtt = self.attributeSearch(betaSearch,2)
        
        # # PE (TTM)
        # peSearch = 'PE_RATIO-value'
        # peAtt = self.attributeSearch(peSearch,2)
        
        # # EPS
        # epsSearch = 'EPS_RATIO-value'
        # epsAtt = self.attributeSearch(epsSearch,2)
        
        # # Market Cap
        # mktCapSearch = 'MARKET_CAP-value'
        # mktCap = self.attributeSearch(mktCapSearch,2)
        # if not mktCap == 'N/a':
        #     mktCap = float(mktCap[:-1])
        # mktCapAtt = mktCap
        
        # # Dividend
        # divSearch = 'DIVIDEND_AND_YIELD-value'
        # divPerc = self.attributeSearch(divSearch,1)
        # if not divPerc == 'N/a':
        #     divPerc = round(float((divPerc[divPerc.find("(")+1:divPerc.find(")")])[:-1])/100,5)
        # divAtt = divPerc
        
        # # 52 Week Price Range
        # yrRangeSearch = 'FIFTY_TWO_WK_RANGE-value'
        # priceRange = self.attributeSearch(yrRangeSearch,1)
        # if not priceRange == 'N/a':
        #     priceRange = list(map(float,priceRange.strip().split(' - ')))
        # yrPriceRange = priceRange
        
scrape = Scrape()
scrape.scrapeVIX()