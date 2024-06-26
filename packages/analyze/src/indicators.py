from utility import Utility

import numpy as np
import pandas as pd
from datetime import datetime as dt
import math

import matplotlib.pyplot as plt
# import mplfinance as mpf

class Indicators:
    
    def avgPrice(data,dev=1,avgType='mean',colors=(0.25,0.25,0.25),outputAll=False,ax=None,plotDev=False,plotOpt=False):
        y = list(data)        
        std = np.nanstd(data)
        
        if avgType.lower() == 'mean':
            avg = np.nanmean(data)*np.ones((len(data),1))
        elif avgType.lower() == 'median':
            avg = np.nanmedian(data)*np.ones((len(data),1))
        
        if outputAll:
            pattern = [avg,std]
        else:
            pattern = avg
        
        if plotOpt: 
            if ax == None:
                ax = plt.gca()
                
            ax.plot(data.index.values,avg,color=colors,linestyle='dashed')
            
            if plotDev:
                ax.plot(data.index.values,avg+(dev*std),color=colors,linestyle='dotted')
                ax.plot(data.index.values,avg-(dev*std),color=colors,linestyle='dotted')
        
        return pattern
    
    def regression(data,curveType='linear',minStd=1,dev=1,colors=(0.25,0.25,0.25),style='solid',devOpt=False,devColors=((0.2,0.5,0.2),(0.75,0.2,0.2)),outputAll=False,ax=None,plotOpt=False):
        if curveType.lower() == 'linear':
            y = np.asarray(data)
        elif curveType.lower() == 'logarithmic':
            y = np.asarray(np.log10(data))
        else:
            raise Exception('ERROR: Invalid curve type specified.')
            
        x = np.asarray([(date - data.index.values[0]) / (data.index.values[-1] - data.index.values[0]) for date in data.index.values])
        
        Sxx = np.sum((x - np.nanmean(x))**2)
        # Syy = np.sum((y - np.nanmean(y))**2)
        Sxy = np.sum((x - np.nanmean(x)) * (y - np.nanmean(y)))
        
        m = Sxy / Sxx
        b = np.nanmean(y) - m * np.nanmean(x)
        # r = Sxy / (np.sqrt(Sxx) * np.sqrt(Syy))
        
        if curveType.lower() == 'linear':
            z = b + m * x
        elif curveType.lower() == 'logarithmic':
            r = 10 ** m
            A = 10 ** b
            z = A * r ** x
            
        std = np.std(y)
        r2 = Utility.rSquared(data,z)
        
        if outputAll:
            pattern = (z,std,r2)
        else:
            pattern = z
        
        if plotOpt:
            if ax == None:
                ax = plt.gca()
                
            threshAmt = minStd * (max(y) - min(y))
            if std < threshAmt:
                ax.plot(data.index.values,z,color=colors,linestyle=style,linewidth=1)
                
                if devOpt:
                    ax.plot(data.index.values,z+(dev*std),color=devColors[0],linestyle='dashed',linewidth=0.5)
                    ax.plot(data.index.values,z-(dev*std),color=devColors[1],linestyle='dashed',linewidth=0.5)
        
        return pattern
    
    def supportResistance(data,thresh=0.02,minNum=3,minDuration=0,style='resistance',ax=None,plotOpt=False):
        if style.lower() == 'both':
            styles = ['support','resistance']
        elif style.lower() == 'resistance' or style.lower() == 'support':
            styles = [style]
        else:
            raise Exception('ERROR: Invalid line specified.')
        
        pattern = {}
        for style in styles:
            extrema  = Utility.findExtrema(list(data),endsOpt=True)
            peaks = extrema.peaks.dropna() if style.lower() == 'resistance' else extrema.troughs.dropna()
            peakDates,peakPrices  = zip(*[(ind,peak) for ind,peak in peaks.items()])
            
            threshAmt = thresh * (max(data) - min(data))
            
            pattern[style] = []
            priceInds = []
            for priceInd,peakPrice in enumerate(peakPrices):
                if priceInd in priceInds:
                    continue
                
                threshPrices = [peakPrice - threshAmt,peakPrice + threshAmt]
                
                assocPeaks = []
                for ind,price in enumerate(peakPrices[priceInd:]):
                    if ind == 0:
                        assocPeaks.append((ind+priceInd,price))
                    else:
                        if style.lower() == 'resistance':
                            if price > threshPrices[1]:
                                break
                            elif price > threshPrices[0]:
                                assocPeaks.append((ind+priceInd,price))
                        elif style.lower() == 'support':
                            if price < threshPrices[0]:
                                break
                            elif price < threshPrices[1]:
                                assocPeaks.append((ind+priceInd,price))
                
                if len(assocPeaks) > 1:
                    timeDelta = peakDates[assocPeaks[-1][0]] - peakDates[assocPeaks[0][0]]
                    
                    if timeDelta > minDuration and len(assocPeaks) >= minNum:
                        pattern[style].append(assocPeaks)
                        priceInds.extend([peak[0] for peak in assocPeaks])
            
            if plotOpt:
                if ax == None:
                    ax = plt.gca()
                    
                if style.lower() == 'resistance':
                    color = 'g-'
                elif style.lower() == 'support':
                    color = 'r-'
                    
                for pat in pattern[style]:
                    dates = [data.index.values[peakDates[pt[0]]] for pt in pat]
                    prices = [pt[1] for pt in pat]
                    ax.plot(np.asarray(dates),np.ones(len(prices))*np.mean(prices),color,linewidth=1)
        
        return pattern
 
    def trend(data,minStd=1,numIters=1,direction='both',segments=1,ax=None,plotOpt=False):
        def getPeakPrices(segmentData,line):
            peakPrices,peakDates = [],[]
            for i in range(numIters):
                endOpts = False if i == 0 else True
                peakPrices = segmentData if i == 0 else peakPrices

                extrema  = Utility.findExtrema(peakPrices,endsOpt=endOpts)
                peaks = extrema.peaks.dropna()
                troughs = extrema.troughs.dropna()                  
                
                iter = peaks if line.lower() == 'upper' else troughs if line.lower() == 'lower' else None
                for ind,peak in iter.items():
                    peakDates = ind+(len(data)-per) if i == 0 else peakDates[ind]
                    peakPrices = peak
            
            plt.plot(data.iloc[-per:])
            plt.plot(data.index.values[peakDates],peakPrices)
            
            return peakDates,peakPrices

        def findBounds(peakDates,peakPrices,line):            
            maxPeaks = []
            iters = list(zip(peakDates,peakPrices))
            if line.lower() == 'upper':
                if direction.lower() == 'down':
                    iters = iters[::-1]
                maxPeak = 0
            elif line.lower() == 'lower':
                if direction.lower() == 'up':
                    iters = iters[::-1]
                maxPeak = 1e9
    
            for date,peak in iters:
                if line.lower() == 'upper' and peak > maxPeak:
                    maxPeak = peak
                    maxPeaks.append((date,peak))
                elif line.lower() == 'lower' and peak < maxPeak:
                    maxPeak = peak
                    maxPeaks.append((date,peak))
                
            boundDates  = [date for date,_ in maxPeaks]
            boundPrices = [peak for _,peak in maxPeaks]
            
            plt.plot(data.index.values[boundDates],boundPrices)
            plt.show()

            peakDf = pd.Series(np.transpose(boundPrices),index=data.index.values[boundDates])
            
            return peakDf
        
        # Set direction variable to define patterns to search
        match direction.lower():
            case ['up','down']:
                directions = [direction]
            case 'both':
                directions = ['up','down']
            case _:
                raise Exception('ERROR: Invalid direction specified')            
        
        # Get periods to assess patterns
        periods = np.round(Utility.expCurve(segments+1,steepness=3,reverse=False)[1:]*(len(data)-1)).astype(int)
        for per in periods:
            # pattern = {}
            for direction in directions:
                if direction.lower() == 'up':
                    linestyle = 'solid'
                elif direction.lower() == 'down':
                    linestyle = 'dashed'

                ## TOP
                peakDates,peakPrices = getPeakPrices(data.iloc[-per:],line='upper')
                peakDf = findBounds(peakDates,peakPrices,line='upper')
                peakReg = Indicators.regression(peakDf,minStd=minStd,colors=(0.2,0.5,0.2),style=linestyle,ax=ax,plotOpt=True)
                
                ## BOTTOM
                troughDates,troughPrices = getPeakPrices(data.iloc[-per:],line='lower')
                troughDf = findBounds(troughDates,troughPrices,line='lower')
                troughReg = Indicators.regression(troughDf,minStd=minStd,colors=(0.75,0.2,0.2),style=linestyle,ax=ax,plotOpt=True)
            
                # pattern[direction] = [peakReg,troughReg]
                # pattern = [peakR2,troughR2]
        
        # return pattern
    
    def extremaGaps(data,minPerc=0,minDuration=0,plotOpt=False):
        extrema  = Utility.findExtrema(list(data),endsOpt=False)
        peakDates,peakPrices  = zip(*[(ind,peak) for ind,peak in extrema.peaks.dropna().items()])
        troughDates,troughPrices  = zip(*[(ind,peak) for ind,peak in extrema.troughs.dropna().items()])
        
        # plt.scatter(data.index.values[troughDates],troughPrices)
        
        p2tInd = 0 if peakDates[0] < troughDates[0] else 1
        threshAmt = minPerc * (max(data) - min(data))
        
        dates  = [peakDates,troughDates]
        prices = [peakPrices,troughPrices]
        
        peak2trough = []
        trough2peaks = []
        ind = 0 if p2tInd == 0 else 1
        for n in range(min(len(dates[0]),len(dates[1]))-1):
            for i in range(2):
                if i == 0:
                    dTime  = (data.index.values[dates[abs(ind-1)][n]] - data.index.values[dates[ind][n]]).astype('timedelta64[D]') / np.timedelta64(1,'D')
                    dPrice = abs(prices[abs(ind-1)][n] - prices[ind][n])
                else:
                    dTime  = (data.index.values[dates[ind][n+1]] - data.index.values[dates[abs(ind-1)][n]]).astype('timedelta64[D]') / np.timedelta64(1,'D')
                    dPrice = abs(prices[ind][n+1] - prices[abs(ind-1)][n])
                    
                if dTime > minDuration and dPrice > threshAmt:
                    if ind == 0:
                        if i == 0:
                            peak2trough.append(dTime)
                            
                            if plotOpt:
                                plt.scatter(data.index.values[dates[0][n]],prices[0][n],c='g',marker='|',s=1000,linewidths=4)
                                plt.scatter(data.index.values[dates[1][n]],prices[1][n],c='r',marker='|',s=1000,linewidths=4) 
                            
                        else:
                            trough2peaks.append(dTime)
                            
                            if plotOpt:
                                plt.scatter(data.index.values[dates[1][n]],prices[1][n],c='r',marker='_',s=1000,linewidths=4) 
                                plt.scatter(data.index.values[dates[0][n+1]],prices[0][n+1],c='g',marker='_',s=1000,linewidths=4)
                    else:
                        if i == 0:
                            trough2peaks.append(dTime)
                            
                            if plotOpt:
                                plt.scatter(data.index.values[dates[1][n]],prices[1][n],c='r',marker='_',s=1000,linewidths=4) 
                                plt.scatter(data.index.values[dates[0][n+1]],prices[0][n+1],c='g',marker='_',s=1000,linewidths=4)
                        else:
                            peak2trough.append(dTime)
                            
                            if plotOpt:
                                plt.scatter(data.index.values[dates[0][n]],prices[0][n],c='g',marker='|',s=1000,linewidths=4)
                                plt.scatter(data.index.values[dates[1][n]],prices[1][n],c='r',marker='|',s=1000,linewidths=4) 
        
        # pattern = [peak2trough,trough2peaks]
        
        return [peakDates,troughDates]
    
    def volumeAtPrice(data,numBins=20,volumeType='all',integrated=True,ax=None,plotOpt=False):
        prices = np.mean(np.asarray((data['Open'],data['Close'])).transpose(),axis=1)
        volume = np.asarray(data['Volume'])
        
        pricesClose = np.asarray(data['Close'])
        
        rng = (min(prices),max(prices))
        binSize = (rng[1] - rng[0]) / numBins
        
        hist = np.zeros((numBins,3))
        pattern = []
        for ind,val in enumerate(pricesClose):
            if ind == 0:
                buySell = 0
            else:
                if val > pricesClose[ind-1]:    #buy
                    buySell = 0
                elif val < pricesClose[ind-1]:  #sell
                    buySell = 1
                else:                           #hold
                    buySell = 2
                   
            histBin = int(np.ceil((prices[ind] - rng[0]) / binSize) - 1)
            hist[histBin,buySell] += volume[ind]
            
            pattern.append((histBin,buySell,volume[ind]))
        
        if plotOpt:
            if ax == None:
                ax = plt.gca()
            
            x = np.linspace(rng[0],rng[1]-binSize,num=numBins)
            
            if integrated:
                dates = (data.index.values[0],data.index.values[-1])
                
                if volumeType.lower() == 'buy':
                    volumes = hist[:,0]
                    color = 'green'
                elif volumeType.lower() == 'sell':
                    volumes = hist[:,1]
                    color = 'red'
                elif volumeType.lower() == 'hold':
                    volumes = hist[:,2]
                    color = 'gray'
                elif volumeType.lower() == 'all':
                    volumes = np.sum(hist,axis=1)
                    color = 'blue'
                else:
                    raise Exception('ERROR: Invalid volumType specified')
                
                dateVolumes = (((volumes - min(volumes)) / (max(volumes) - min(volumes))) * (dates[1] - dates[0])) + dates[0]
                
                ax.barh(x,dateVolumes,binSize,color=color,alpha=0.3)
            else:
                ax.bar(x,hist[:,0],binSize,color='green')
                ax.bar(x,hist[:,1],binSize,bottom=hist[:,0],color='red')
                ax.bar(x,hist[:,2],binSize,bottom=np.sum(hist[:,0:2],axis=1),color='gray')
        
        return pattern
    
    def movingAverage(data,window=20,avgType='simple',steepness=3,ignoreStart=True,outputAll=False,colors=None,plotDelta=False,ax=None,plotOpt=False):
        prices = data['Close']
        
        pattern = []
        meanPrice, deltaPrice, avgSlope = [],[],[]
        for i in range(len(data)):
            if ignoreStart and i < window:
                meanPrice.append(np.nan)
                deltaPrice.append(np.nan)
                avgSlope.append(np.nan)
            else:
                if i < window:
                    vals = prices[:i+1]
                else:
                    vals = prices[i-window+1:i+1]
                    
                meanPrice.append(Utility.avg(vals,avgType=avgType,steepness=steepness))
                deltaPrice.append(prices[i] - meanPrice[i])
                if len(meanPrice) > 0:
                    avgSlope.append(meanPrice[i] - meanPrice[i-1])
                else:
                    avgSlope.append(np.nan)
                    
            if outputAll:
                pattern.append((meanPrice[-1],deltaPrice[-1],avgSlope[-1]))
            else:
                pattern.append(meanPrice[-1])
        
        deltaPrice = pd.Series(deltaPrice,index=data.index.values)
        
        if plotOpt:
            if isinstance(ax,(np.ndarray,list)):
                if len(ax) != 2:
                    raise Exception('ERROR: Invalid number of axes provided.')
                
                if colors == None or len(colors) != 2:
                    colors = ('tab:orange','tab:blue')
                    
                ax1 = ax[0]
                ax2 = ax[1]
                
                ax1.plot(data.index.values,meanPrice,color=colors[0],linewidth=1,label='MA_' + avgType.upper() + '_' + str(window))
                
                # ax2.bar(data.index.values,avgSlope,color=colors[1])
                # ax2.set_ylabel('Slope',color=colors[1]) 
                
                # ax2 = ax2.twinx()
                ax2.set_ylabel('Price Delta',color=colors[0])  
                
                ax2.plot(data.index.values,deltaPrice,color=colors[0],linewidth=1)
                Indicators.avgPrice(deltaPrice,colors=colors[0],ax=ax2,plotDev=True,plotOpt=True)
                # ax2.axhline(color='k',linewidth=0.5)
            else:
                if ax == None:
                    ax = plt.gca()
                
                if not plotDelta:
                    if colors == None:
                        colors = 'tab:orange'
                        
                    ax.plot(data.index.values,meanPrice,color=colors,linewidth=1,label='MA_' + avgType.upper() + '_' + str(window))
                else:
                    if colors == None or len(colors) != 2:
                        colors = ('tab:orange','tab:blue')
                        
                    ax.bar(data.index.values,avgSlope,color=colors[0])
                    ax.set_ylabel('Slope',color=colors[0]) 
                    
                    ax = ax.twinx()
                    ax.set_ylabel('Price Delta',color=colors[1])  
                    
                    ax.plot(data.index.values,deltaPrice,color=colors[1],linewidth=1)
                    Indicators.avgPrice(deltaPrice,colors=colors[0],ax=ax,plotDev=True,plotOpt=True)
                    # ax.axhline(color='k',linewidth=0.5)
                
        return pattern
    
    def rsi(data,window=14,avgType=None,ignoreStart=True):
        prices = data['Close']
        
        pattern = []
        rsi = []
        avgGain,avgLoss = [],[]
        for i in range(len(prices)):
            if ignoreStart and i < window:
                rsi.append(np.nan)
                avgRsi = rsi[-1]
            else:
                if len(pattern) <= window:
                    vals = prices[:i+1]
                        
                    gain,loss = [],[]
                    for n in range(len(vals)):
                        if n > 0:
                            change = vals[n]/vals[n-1] - 1
                            if change > 0:
                                gain.append(change)
                            else:
                                loss.append(change)
                
                    avgGain.append(np.sum(gain) / window)
                    avgLoss.append(abs(np.sum(loss)) / window)
                else:
                    change = prices[i]/prices[i-1] - 1
                    if change > 0:
                        gain = change
                        loss = 0
                    else:
                        loss = abs(change)
                        gain = 0
                                
                    avgGain.append(((avgGain[-1] * (window-1)) + gain) / window)
                    avgLoss.append(((avgLoss[-1] * (window-1)) + loss) / window)
                    
                rsi.append(100 - (100 / (1 + (avgGain[-1] / avgLoss[-1]))))
                
                if i < window:
                    vals = rsi[:i+1]
                else:
                    vals = rsi[i-window+1:i+1]
                
                if not avgType is None:
                    avgRsi = Utility.avg(vals,avgType=avgType)
                else:
                    avgRsi = rsi[-1]
                    
            pattern.append(avgRsi)
                
        return pattern
    
    def atr(data,window=14,avgType='simple',ignoreStart=True,ax=None,plotOpt=False):
        pricesHigh = data['High']
        pricesLow = data['Low']
        pricesClose = data['Close']
        
        atr = []
        for i in range(len(pricesClose)):
            hilo = pricesHigh[i] - pricesLow[i]
            hicl = abs(pricesHigh[i] - pricesClose[i-1])
            cllo = abs(pricesClose[i-1] - pricesLow[i])
            
            atr.append(max(hilo,hicl,cllo))
        
        pattern = []
        for i in range(len(atr)):
            if ignoreStart and i < window:
                meanPrice = np.nan
            else:
                if i < window:
                    vals = atr[:i+1]
                else:
                    vals = atr[i-window+1:i+1]
                    
                meanPrice = Utility.avg(vals,avgType=avgType)
                
            pattern.append(meanPrice)
        
        if plotOpt:
            if ax == None:
                ax = plt.gca()
                
            ax.plot(data.index.values,pattern)
        
        return pattern
    
    def bollingerBands(data,window=20,avgType='simple',ignoreStart=True,ax=None,plotOpt=False):
        prices = data['Close']

        pattern = []
        for i in range(len(prices)):
            if ignoreStart and i < window:
                meanPrice = np.nan
                stdPrice = np.nan
            else:
                if i <= window-1:
                    price = prices[:i+1]
                else:
                    price = prices[i-window+1:i+1]
                
                meanPrice = Utility.avg(price,avgType=avgType)
                stdPrice = np.std(price)
                
            pattern.append((meanPrice-stdPrice,meanPrice,meanPrice+stdPrice))
            
        if plotOpt:
            if ax == None:
                ax = plt.gca()
                
            bottom = [n[0] for n in pattern]
            top    = [n[1] for n in pattern]

            # ax.plot(data.index.values,bottom,color='red',linestyle='dashed',linewidth=1)
            ax.plot(data.index.values,top,color='green',linestyle='dashed',linewidth=1)
        
        return pattern
    
    def macd(data,fast=12,slow=26,sig=9,avgType='simple',ignoreStart=True,ax=None,plotOpt=False):
        def avgPrice(window):
            if ignoreStart and i < window:
                avg = [np.nan]
            else:
                if i <= fast-1:
                    avg = prices[:i+1]
                else:
                    avg = prices[i-window+1:i+1]
            
            return avg
                
        prices = data['Close']
        
        pattern = []
        macd = []
        for i in range(len(prices)):
            fastPrice = Utility.avg(avgPrice(fast),avgType=avgType)
            slowPrice = Utility.avg(avgPrice(slow),avgType=avgType)
            
            macd.append(fastPrice - slowPrice)
            if len(macd) < sig:
                signal = np.nan
            else:
                signal = Utility.avg(macd[i-sig+1:i+1],avgType=avgType)
                
            pattern.append((macd[-1],signal))
        
        if plotOpt:
            if ax == None:
                ax = plt.gca()
            
            ax.plot(data.index.values,list(zip(*pattern))[0],color='tab:blue',label='MACD')            
            ax.plot(data.index.values,list(zip(*pattern))[1],color='tab:orange',label='Signal')
            
            # Indicators.avgPrice(pd.Series(macd,index=data.index.values),colors='tab:blue',ax=ax,plotDev=True,plotOpt=True)
            
        return pattern
    
    def accDist(data):
        pricesHigh = data['High']
        pricesLow = data['Low']
        pricesClose = data['Close']
        volume = data['Volume']
        
        pattern = []
        ad = []
        for i in range(len(volume)):
            mfm = ((pricesClose[i] - pricesLow[i]) - (pricesHigh[i] - pricesClose[i])) / (pricesHigh[i] - pricesLow[i])
            moneyFlow = mfm * volume[i]
            
            if i == 0:
                ad.append(moneyFlow) 
            else:
                ad.append(moneyFlow + ad[-1]) 
            
            pattern.append(ad[-1])
        
        return pattern