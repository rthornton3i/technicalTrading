# from backtest import Backtest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time 

class Analyze():
    
    def extractData():
        path = 'Files/MACD_SMAdiff/'
        for ticker in ['SPY','QQQ','DIA']:
            init = False
            for i in range(366):
                try:
                    dfi = pd.read_csv(path + ticker + '_' + str(i) + '.csv',index_col=0)
                except:
                    continue
                
                if not init:
                    df = dfi
                    init = True
                else:
                    df = df.append(dfi)
                
            df = df.sort_values(by=['value'],ascending=False)    
        
            df = df[df.numSell>0]
            # df = df[df.value>45000]
            
            df.to_csv(path + ticker + '_00.csv')
        
        # stdNumSell = np.std(df.numSell)
        # avgNumSell = np.mean(df.numSell)
        
        # df = df[df.numSell>(avgNumSell-(2*stdNumSell))]
        
        # return df
    
if __name__ == '__main__':
    print('Running...')
    
    tic = time()
    
    # backtest = Backtest()
    # backtest.test()
    # Data = backtest.Data
    
    Analyze.extractData()
    # df = Analyze.extractData()
    
    # x = df.fast
    # y = df.slow
    # z = df.value
    # v = df.sig
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    # scatter = ax.scatter3D(x, y, z, c=v, cmap='viridis')

    # ax.set_xlabel('fast')
    # ax.set_ylabel('slow')
    # ax.set_zlabel('value')
    # cb = fig.colorbar(scatter,ax=ax)
    # cb.set_label('sig')

    toc = time() - tic
    print('Runtime: ' + str(toc))         