import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from ta2_hrr_analysisCode.gp_opt import BasicOptimiser 
from sklearn.gaussian_process.kernels import RBF, Matern

def readLogFile(logFilePath):
    if os.path.isfile(logFilePath):
        df = pd.read_csv(logFilePath,'\s+')
    else:
        df = None
        return

    fitness = None
    fitness_err = None
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

    BO = BasicOptimiser(nDims, kernel=kernel, sample_scale=1,scale=sScale, maximise_effort=100,fit_white_noise=True)
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
        

def plotOptParams(xList,yList,xOptList,plotDims=[16/2.54, 16/2.54/1.6/3],dpi=160):
    nRows = len(xList)
    nDims = len(xList[0].keys())
    nPlots = int(np.ceil(nDims/2))
    f, axs = plt.subplots(nRows,nPlots,figsize=(plotDims[0], plotDims[1]*nRows), dpi=dpi, facecolor='w', edgecolor='k')
    xLims = getParamLims(xList)
    m=0
    for x,y,xOpt in zip(xList,yList,xOptList):
        keys = list(x.keys())
        yKey = None
        for key in y.keys():
            if yKey is None:
                yKey = key
            else:
                print('extra y keys ignored')
        yVals = y[yKey]
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
            a1_factor1000 = np.trunc(np.log10(np.max(np.abs(xLims[k1])))/np.log10(1e3))
            a1_factor = 1e3**a1_factor1000
            a2_factor1000 = np.trunc(np.log10(np.max(np.abs(xLims[k2])))/np.log10(1e3))
            a2_factor = 1e3**a2_factor1000
            iSel = np.argsort(yVals)
            
            plt.scatter(a1[iSel]/a1_factor,a2[iSel]/a2_factor,c=yVals[iSel],cmap='winter',s=4)

            
            if not (np.min(a1)==np.max(a1)):
                plt.plot([a1[0]]*2/a1_factor,xLims[k2]/a2_factor,'k--')
                plt.plot([xO]*2/a1_factor,xLims[k2]/a2_factor,'r--')
            if not (np.min(a2)==np.max(a2)):
                plt.plot(xLims[k1]/a1_factor,[a2[0]]*2/a2_factor,'k--')
                plt.plot(xLims[k1]/a1_factor,[yO]*2/a2_factor,'r--')
                
            plt.xlim(xLims[k1]/a1_factor)
            plt.ylim(xLims[k2]/a2_factor)
            if a1_factor1000==0:
                plt.xlabel(k1)
            else:
                plt.xlabel(k1 + r'$ \times 10^{' + '%1u' % (a1_factor1000*3) + '}$')
            if a2_factor1000==0:
                plt.ylabel(k2)
            else:
                plt.ylabel(k1 + r'$ \times 10^{' + '%1u' % (a1_factor1000*3) + '}$')


        cbh=plt.colorbar()
        cbh.set_label(yKey)
    plt.tight_layout()
    return f, axs 

def plotGP_maximum(logFilePath,GP=None):
    bNum, params, fitness, fitness_err, nDims = readLogFile(logFilePath)
    
    if GP is None:
        nDims = len(params.keys())
        GP = GP_optimiser(nDims)
    GP_max = []
    for n in range(len(bNum)):
        x_test = []
        for key in params.keys():
            x_test.append(params[key][n])
        GP.tell(x_test,fitness[n],fitness_err[n])
        best_pos, best_val = GP.optimum()
        GP_max.append(best_val)
    return bNum,np.array(GP_max)
        


