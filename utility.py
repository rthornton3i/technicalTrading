import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

class Utility:
    
    def findExtrema(data,endsOpt=True):
        peaks,troughs = [],[]
        for ind,val in enumerate(data):
            if ind > 0 and ind < len(data)-1:
                if val > data[ind-1] and val > data[ind+1]:
                    peaks.append((ind,val))
                elif val < data[ind-1] and val < data[ind+1]:
                    troughs.append((ind,val))
            elif endsOpt:
                peaks.append((ind,val))
                troughs.append((ind,val))
        
        return [peaks,troughs]
    
    def smooth(data,avgType='simple',steepness=3,window=3,iterations=1,trailing=False):
        if trailing:
            k = window
        else:
            k = int(window/2)    
        
        tempData = data
        for i in range(iterations):
            smoothData = []
            for j in range(len(tempData)):
                if trailing:
                    if j < k:
                        vals = tempData[:j+1]
                    else:
                        vals = tempData[j-k+1:j+1]
                else:
                    if j < k:
                        vals = tempData[:j+k+1]
                    elif j > len(tempData)-1-k:
                        vals = tempData[j-k:]
                    else:
                        vals = tempData[j-k:j+k+1]
                
                smoothData.append(Utility.avg(vals,avgType=avgType,steepness=steepness))
            
            tempData = smoothData
            
        return smoothData
    
    def expCurve(numel,steepness=3,reverse=False):
        steepness = steepness * (numel / 10)
        
        curve = np.exp(np.arange(numel)*(1/steepness))
        curve = (curve - min(curve)) / (max(curve) - min(curve))
        if reverse:
            curve = 1 - curve[::-1]
        
        if len(curve) == 1:
            curve = [1]
            
        return curve
    
    def avg(vals,avgType='simple',steepness=4):            
        if avgType.lower() == 'simple':
            mean = np.mean(vals)
        elif avgType.lower() == 'exponential':
            mean = np.average(vals,weights=Utility.expCurve(len(vals),steepness))
        elif avgType.lower() == 'logarithmic':
            mean = np.average(vals,weights=Utility.expCurve(len(vals),steepness,reverse=True))
        elif avgType.lower() == 'weighted':
            mean = np.average(vals,weights=np.arange(len(vals))+1)
        elif avgType.lower() == 'wilders':
            mean = 0
        elif avgType.lower() == 'median':
            mean = np.median(vals)
            
        return mean
    
    def rSquared(actualData,expectedData):
        rss = sum((actualData - expectedData) ** 2)
        tss = sum((actualData - np.mean(actualData)) ** 2)
        
        r2 = 1 - (rss / tss)
        
        return r2
    
    def setPlot(ax,logscale=False,xlabels=None,xlimits=None,ylimits=None,numxticks=10,numyticks=5):
        if logscale:
            ax.set_yscale('log',base=2)
            ax.yaxis.set_major_formatter(tick.ScalarFormatter())
        
        if not xlimits is None:
            ax.set_xlim(xlimits)
        
        if not xlabels is None:
            ax.set_xticks(np.arange(len(xlabels))) 
            # ax.set_xticklabels(xlabels)   
        
        if not ylimits is None:
            ax.set_ylim(ylimits)
            ax.yaxis.set_ticks(np.linspace(ylimits[0],ylimits[1],numyticks)) 
            
        ax.grid()
        ax.grid(axis='x',linestyle='--')
        
        ax.tick_params(axis='x',labelrotation=45)
        
    def nmspc2df(nmspc):
        def createDf(ns):
            tempDf = pd.DataFrame()
            
            for attr in dir(ns):
                if not callable(getattr(ns, attr)) and not attr.startswith("__"):
                    tempDf[attr] = getattr(ns,attr)
            
            return tempDf

        if isinstance(nmspc,list):
            for i,ns in enumerate(nmspc):
                if i == 0:
                    df = createDf(ns)
                else:
                    tempDf = createDf(ns)
                    df = df.append(tempDf,ignore_index=True)
        else:
            df = createDf(nmspc)
                
        return df
    
    def dicts2df(dicts):
        if isinstance(dicts,list):
            for i,dic in enumerate(dicts):
                if i == 0:
                    df = pd.DataFrame.from_dict(dic)
                else:
                    df = df.append(pd.DataFrame.from_dict(dic))
        else:
            df = pd.DataFrame.from_dict(dicts)
            
        return df