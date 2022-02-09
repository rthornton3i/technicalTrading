# from backtest import Backtest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time 

class Analyze():
    
    def extractData(tickers,path):
        values = {}
        ranks = {}
        inds = {}
        for ticker in tickers:
            init = False
            for i in range(100):
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
        
            # df = df[df.numSell>0]
            # df = df[df.value>45000]
            
            # df.to_csv(path + ticker + '.csv')
            
            values[ticker] = df.value.tolist()
            inds[ticker] = df.zzz.tolist()
            
            # dfs[ticker] = df
        
        allRanks = []
        if len(tickers) > 1:
            tick = tickers[0]
            for ind,value in zip(inds[tick],values[tick]):
                valRanks = [value]
                for ticker in tickers[1:]:
                    n = inds[ticker].index(ind)
                    valRanks.append(values[ticker][n])
                    
                allRanks.append(valRanks)
            
            avgRanks = np.mean(allRanks,axis=1).tolist()
            
            bestN = []
            bestInd = []
            bestVal = []
            tempRanks = avgRanks.copy()
            for _ in range(10):
                bestN.append(tempRanks.index(max(tempRanks)))
                bestInd.append(inds[tick][bestN[-1]])
                bestVal.append([values[ticker][bestN[-1]] for ticker in tickers])
                
                tempRanks[bestN[-1]] = len(tempRanks)
        
        else:
            avgRanks = 0
            bestInd = 0
            
        return [allRanks,avgRanks,bestInd,bestVal]
                
    
if __name__ == '__main__':
    print('Running...')
    
    tic = time()
    
    # backtest = Backtest()
    # backtest.test()
    # Data = backtest.Data
    
    path = 'Files/MACD_ATR_Slow/'
    tickers = ['QQQC','QQQF','QQQM','QQQS']#['SPY','QQQ','DIA']
    [allRanks,avgRanks,bestInd,bestVal] = Analyze.extractData(tickers,path)
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