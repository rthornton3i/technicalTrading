from backtest import Backtest

import pandas as pd
import matplotlib.pyplot as plt

tickers = ['SPY']#['SPY','QQQ','DIA']

# Slow
# startDate = pd.Timestamp(year=2012,month=4,day=1)
# endDate   = pd.Timestamp(year=2014,month=3,day=1)

# Medium
# startDate = pd.Timestamp(year=2013,month=8,day=1)
# endDate   = pd.Timestamp(year=2018,month=1,day=1)

# Fast
# startDate = pd.Timestamp(year=2017,month=6,day=1)
# endDate   = pd.Timestamp(year=2020,month=2,day=1)

# COVID
# startDate = pd.Timestamp(year=2019,month=7,day=1)
# endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)

# Housing Crisis
# startDate = pd.Timestamp(year=2007,month=1,day=1)
# endDate = pd.Timestamp(year=dt.now().year,month=dt.now().month,day=dt.now().day)

# Excel
startDate = pd.Timestamp(year=2013,month=7,day=8)
endDate = pd.Timestamp(year=2023,month=7,day=7)

initialFunds = 10000

backtest = Backtest(tickers,startDate,endDate,plotOpt=True)
# backtest.exploration()
backtest.run()

plt.show()