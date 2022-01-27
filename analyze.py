from backtest import Backtest

import pandas as pd
import numpy as np

from time import time 

class Analyze():
    
    def concatData():
        for i in range(2):
            dfi = pd.read_csv('Files/MACD_SMA/SPY_' + str(i) + '.csv',index_col=0)
            
            if i == 0:
                df = dfi
            else:
                df = df.append(dfi)
                
        df = df.sort_values(by=['value'],ascending=False)
        
        return df
    
if __name__ == '__main__':
    print('Running...')
    
    tic = time()
    
    backtest = Backtest()
    backtest.test()
    Data = backtest.Data
    
    df = Analyze.concatData()    

    toc = time() - tic
    print('Runtime: ' + str(toc))         