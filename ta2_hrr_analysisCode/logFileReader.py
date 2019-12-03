import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ta2_hrr_analysisCode.gp_opt import BasicOptimiser 
from sklearn.gaussian_process.kernels import RBF

def readLogFile(logFilePath):
    if os.path.isfile(logFilePath):
        df = pd.read_csv(logFilePath,'\s+')
    else:
        df = None
        return

    GP_flag = 'fitness' in df.keys()
    params ={}
    for key in df.keys():
        vals = df[key].values
        if 'run' in key:
            continue
        elif 'burst' in key:
            bNum = vals
        elif key == 'fitness':
            fitness =vals
        elif key == 'fitness_error':
            fitness_err =vals
        else:
            params[key] = vals
    nDims = len(params.keys())
    return bNum, params, fitness, fitness_err, nDims
    
def GP_optimiser(nDims,sScale=None):
    if sScale is None:
        sScale = [200.0, 5000.0, 100000.0, 0.1, 100.0,0.2]
        sScale = sScale[:nDims]


    length_scale=[]
    length_scale_bounds=[]
    for n in range(nDims):
        length_scale.append(1)
        length_scale_bounds.append([0.1,5])

    kernel =2**2 * RBF(length_scale=length_scale,length_scale_bounds=length_scale_bounds)
    kernel.k1.constant_value_bounds = (0.001,100)

    BO = BasicOptimiser(nDims, kernel=kernel, sample_scale=sScale, maximise_effort=100,fit_white_noise=True)
    return BO


def xRange(x):
    return np.max(x)-np.min(x)

def getParamLims(xList):
    xLims = {}
    for key in xList[0].keys():
        xAll = []
        for x in xList:
            xAll.append(x[key].flatten())

        xAll = np.array(xAll).flatten()
        if np.min(xAll)==np.max(xAll):
            if np.mean(xAll)==0:
                xLims[key] = [-1,1]
            else:                
                xLims[key] = [np.mean(xAll)*(0.9),np.mean(xAll)*(1.1)]
        else:
            xLims[key] = [np.min(xAll),np.max(xAll)]     
    return xLims
        

def plotOptParams(xList,yList,xOptList):
    nRows = len(xList)
    nDims = len(xList[0].keys())
    nPlots = int(np.ceil(nDims/2))
    f, axs = plt.subplots(nRows,nPlots,figsize=(3*8.5/2.54, 3*8.5/2.54/1.6/3*nRows), dpi=80, facecolor='w', edgecolor='k')
    xLims = getParamLims(xList)
    m=0
    for x,y,xOpt in zip(xList,yList,xOptList):
        keys = list(x.keys())
        m = m+1
        for n in range(1,nPlots+1):
            fig = plt.subplot(nRows,nPlots,n+(nPlots*(m-1)))

            if (n*2)>nDims:
                k1 = keys[(n-1)*2]
                k2 = keys[0]
                xO = xOpt[(n-1)*2]
                yO = xOpt[0]
            else:
                k1 = keys[(n-1)*2]
                k2 = keys[(n-1)*2+1]
                xO = xOpt[(n-1)*2]
                yO = xOpt[(n-1)*2+1]
            a1=x[k1]
            a2=x[k2]

            iSel = np.argsort(y)

            plt.scatter(a1[iSel],a2[iSel],c=y[iSel],cmap='winter',s=4)

            
            if not (np.min(a1)==np.max(a1)):
                plt.plot([a1[0]]*2,xLims[k2],'k--')
                plt.plot([xO]*2,xLims[k2],'r--')
            if not (np.min(a2)==np.max(a2)):
                plt.plot(xLims[k1],[a2[0]]*2,'k--')
                plt.plot(xLims[k1],[yO]*2,'r--')
                
            plt.xlim(xLims[k1])
            plt.ylim(xLims[k2])
            plt.xlabel(k1)
            plt.ylabel(k2)

        plt.colorbar()
        plt.tight_layout()


