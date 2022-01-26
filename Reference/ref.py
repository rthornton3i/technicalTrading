from fetch import Fetch
from strategy import Strategy
from utility import Utility

import numpy as np
import pandas as pd
from datetime import datetime as dt

import matplotlib.pyplot as plt
# import mplfinance as mpf

import warnings

# def run():
warnings.filterwarnings("ignore")

tickers = ['SPY','QQQ','DIA','VTI']#'MMM','ABT','ABBV','ABMD','ACN','ATVI','ADBE','AMD','AAP','AES','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE','LNT','ALL','GOOGL','GOOG','MO','AMZN','AMCR','AEE','AAL','AEP','AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','ADI','ANSS','ANTM','AON','AOS','APA','AAPL','AMAT','APTV','ADM','ANET','AJG','AIZ','T','ATO','ADSK','ADP','AZO','AVB','AVY','BKR','BLL','BAC','BK','BAX','BDX','BRK.B','BBY','BIO','BIIB','BLK','BA','BKNG','BWA','BXP','BSX','BMY','AVGO','BR','BF.B','CHRW','COG','CDNS','CZR','CPB','COF','CAH','KMX','CCL','CARR','CTLT','CAT','CBOE','CBRE','CDW','CE','CNC','CNP','CERN','CF','SCHW','CHTR','CVX','CMG','CB','CHD','CI','CINF','CTAS','CSCO','C','CFG','CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','COP','ED','STZ','COO','CPRT','GLW','CTVA','COST','CCI','CSX','CMI','CVS','DHI','DHR','DRI','DVA','DE','DAL','XRAY','DVN','DXCM','FANG','DLR','DFS','DISCA','DISCK','DISH','DG','DLTR','D','DPZ','DOV','DOW','DTE','DUK','DRE','DD','DXC','EMN','ETN','EBAY','ECL','EIX','EW','EA','EMR','ENPH','ETR','EOG','EFX','EQIX','EQR','ESS','EL','ETSY','EVRG','ES','RE','EXC','EXPE','EXPD','EXR','XOM','FFIV','FB','FAST','FRT','FDX','FIS','FITB','FE','FRC','FISV','FLT','FLIR','FMC','F','FTNT','FTV','FBHS','FOXA','FOX','BEN','FCX','GPS','GRMN','IT','GNRC','GD','GE','GIS','GM','GPC','GILD','GL','GPN','GS','GWW','HAL','HBI','HIG','HAS','HCA','PEAK','HSIC','HSY','HES','HPE','HLT','HFC','HOLX','HD','HON','HRL','HST','HWM','HPQ','HUM','HBAN','HII','IEX','IDXX','INFO','ITW','ILMN','INCY','IR','INTC','ICE','IBM','IP','IPG','IFF','INTU','ISRG','IVZ','IPGP','IQV','IRM','JKHY','J','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY','KEYS','KMB','KIM','KMI','KLAC','KHC','KR','LB','LHX','LH','LRCX','LW','LVS','LEG','LDOS','LEN','LLY','LNC','LIN','LYV','LKQ','LMT','L','LOW','LUMN','LYB','MTB','MRO','MPC','MKTX','MAR','MMC','MLM','MAS','MA','MKC','MXIM','MCD','MCK','MDT','MRK','MET','MTD','MGM','MCHP','MU','MSFT','MAA','MHK','TAP','MDLZ','MPWR','MNST','MCO','MS','MOS','MSI','MSCI','NDAQ','NTAP','NFLX','NWL','NEM','NWSA','NWS','NEE','NLSN','NKE','NI','NSC','NTRS','NOC','NLOK','NCLH','NOV','NRG','NUE','NVDA','NVR','NXPI','ORLY','OXY','ODFL','OMC','OKE','ORCL','OTIS','PCAR','PKG','PH','PAYX','PAYC','PYPL','PENN','PNR','PBCT','PEP','PKI','PRGO','PFE','PM','PSX','PNW','PXD','PNC','POOL','PPG','PPL','PFG','PG','PGR','PLD','PRU','PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RL','RJF','RTX','O','REG','REGN','RF','RSG','RMD','RHI','ROK','ROL','ROP','ROST','RCL','SPGI','CRM','SBAC','SLB','STX','SEE','SRE','NOW','SHW','SPG','SWKS','SNA','SO','LUV','SWK','SBUX','STT','STE','SYK','SIVB','SYF','SNPS','SYY','TMUS','TROW','TTWO','TPR','TGT','TEL','TDY','TFX','TER','TSLA','TXN','TXT','TMO','TJX','TSCO','TT','TDG','TRV','TRMB','TFC','TWTR','TYL','TSN','UDR','ULTA','USB','UAA','UA','UNP','UAL','UNH','UPS','URI','UHS','UNM','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VFC','VIAC','VTRS','V','VNO','VMC','WRB','WAB','WMT','WBA','DIS','WM','WAT','WEC','WFC','WELL','WST','WDC','WU','WRK','WY','WHR','WMB','WLTW','WYNN','XEL','XLNX','XYL','YUM','ZBRA','ZBH','ZION','ZTS','AEZS','APHA','ATNF','ATOS','AZN','BBIG','BHAT','BIDU','BILI','BNGO','BOWX','BOXL','CAN','CDEV','CIDM','CLEU','CSCW','CTRM','DBX','DKNG','EBON','EEIQ','EVFM','FCEL','FTFT','FUTU','GEVO','GNUS','HOFV','IDEX','IMMP','IQ','ITRM','JD','KDP','LI','LIVX','MARA','METX','MKD','MRVL','NAKD','NEXT','NKLA','NNDM','OCGN','OGI','ONTX','OPEN','PDD','PHUN','PLUG','PRQR','RAIL','RIOT','ROOT','SHIP','SIRI','SLM','SNDL','SRNE','SV','TELL','TIGR','TLRY','TNXP','TXMD','VEON','VISL','VUZI','WAFU','WIMI','WKEY','WKHS','YVR','ZNGA']

# startDate = pd.Timestamp(year=dt.now().year-3,month=dt.now().month,day=dt.now().day)
# startDate = pd.Timestamp(year=2009,month=6,day=1)
startDate = pd.Timestamp(year=2020,month=3,day=23)
# startDate = pd.Timestamp(year=2020,month=5,day=1)
    
stockFetch = Fetch(tickers,startDate)
Data = stockFetch.getPrices()
# MovAvg = stockFetch.getMovAvg(style='sma',period=20)
# MACD = stockFetch.getMACD(periods=[12,26,9],absrange=True)
# RSI = stockFetch.getRSI(period=20)

for ticker in tickers:
    ## Get data
    data = Data[ticker]
    # movAvg = MovAvg[ticker]
    # macd = MACD[ticker]
    # rsi = RSI[ticker]
    
    data['Smooth'] = Utility.smooth(list(data['Close']),window=3,iterations=3)
    
    ## PLOT ##################################################################
    
    _, axs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2,1,1]})
    
    ##########################################################################
    
    ax = axs[0]
    Utility.setPlot(ax)#,xlabels=data.index.values)
    
    # mpf.plot(data,type='candle',style='charles')
    ax.plot(data.index.values,data['Close'],linewidth=1)
    # ax.plot(movAvg['Moving Average'])
    # ax.plot(data['Smooth'],linewidth=0.5)
    
    
    ## SUPPORT/RESISTANCE
    # pattern = Strategy.supportResistance(data['Smooth'],thresh=0.15,minNum=4,minDuration=10,style='resistance',ax=ax)
    # pattern = Strategy.supportResistance(data['Smooth'],thresh=0.15,minNum=4,minDuration=10,style='support',ax=ax)
    
    
    ## REGRESSION
    # pattern = Strategy.regression(data['Adjusted'],curveType='logarithmic',dev=1,devOpt=False,ax=ax)
    
    # i = 1
    # for dateOffset in [0]:#,6,9]:
    #     pattern = Strategy.regression(data['Adjusted'].loc[startDate+pd.DateOffset(months=dateOffset):],curveType='logarithmic',dev=1,colors=(1/i,0.6/i,0.2/i),devOpt=False,ax=ax)
    #     i += 1
    
    # pattern = Strategy.avgPrice(data['Adjusted'],dev=1)
    
    
    ## CHANNELS
    for direction in ['up','down']:
        pattern = Strategy.trend(data['Close'],direction=direction,minStd=0.5,ax=ax)
        # for dateOffset in [0]:#,6,9]:
        #     pattern = Strategy.trend(data['Smooth'].loc[startDate+pd.DateOffset(months=dateOffset):],direction=direction,minStd=0.5,ax=ax)
    
    
    ## PEAKS
    # pattern = Strategy.extremaGaps(data['Smooth'],minDuration=10,minPerc=0.1)
    
    # print('Num P2T: ' + str(len(pattern[0])))
    # print('Num T2P: ' + str(len(pattern[1])))
    # print('Avg p2t: ' + str(np.mean(pattern[0])))
    # print('Std p2t: ' + str(np.std(pattern[0])))
    # print('Avg t2p: ' + str(np.mean(pattern[1])))
    # print('Std t2p: ' + str(np.std(pattern[1])))
    
    # data['Adjusted'] = Utility.smooth(list(data['Adjusted']),style='mean',window=20,iterations=1)
    # plt.plot(data['Adjusted'],'k--')
    
    ##########################################################################
    
    ax = axs[1]
    Utility.setPlot(ax)
    
    # ax.plot(macd.index.values,macd['MACD'],'steelblue',linewidth=1)
    # ax.plot(macd.index.values,macd['Signal'],'orange',linewidth=1)
    # ax.bar(macd.index.values,macd['Hist'],color='grey')
    
    ##########################################################################
    
    ax = axs[2]
    Utility.setPlot(ax)#,ylimits=(0,100))
    
    # ax.plot(rsi.index.values,rsi['RSI'],'steelblue',linewidth=1)
    # ax.plot(rsi.index.values,np.ones(len(rsi['RSI']))*70,'lightgreen',linewidth=1,linestyle='dashed')
    # ax.plot(rsi.index.values,np.ones(len(rsi['RSI']))*30,'lightcoral',linewidth=1,linestyle='dashed')
        
# if __name__ == "__main__":
#     run()