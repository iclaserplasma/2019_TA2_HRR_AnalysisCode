#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Thu Sep  6 16:41:14 2018

@author: chrisunderwood
File:    /Users/chrisunderwood/Documents/Python/CUnderwood_Functions.py
    Import with following
import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions3 as func

"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

def displayCurvefitResults(popt, pcov):
    errors = np.sqrt(np.diag(pcov))
    print("Fit Results")
    for p, e in zip(popt, errors):
        print ( "{}\t{}\t{:2.3f}%".format(p, e,e/p))
        
def pcov_to_perr(pcov):
    errors = np.sqrt(np.diag(pcov))
    return errors
    

def legendToLeft(loc = 'center left', bbox_to_anchor=(1,.5), title = '', ncol = 1, ax = None):
    if ax is not None:
        ax.legend(title = title, loc = loc, bbox_to_anchor=bbox_to_anchor, 
               ncol = ncol)
    else:        
        plt.legend(title = title, loc = loc, bbox_to_anchor=bbox_to_anchor, 
               ncol = ncol)
        
def scientificAxis(axisName = 'y', ax = None):
    if ax == None:
        ax = plt.gca() 
    ax.ticklabel_format(style='sci', axis=axisName , scilimits=(0,0))

    

def check_and_makeFolder(savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))        


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))
    
    
def saveDictionary(fname, dict_to_save):
    import json
    with open(fname, 'w') as file:
         file.write(json.dumps(dict_to_save, default=default)) # use `json.loads` to do the reverse


def loadDictionary(fname):
    with open(fname) as f:
        data = json.load(f)
    return data

def printDictionary(dic):
    for item in list(dic):
        print (item, dic[item])
        print ()

def setup_figure_fontAttributes(size = 12, family = 'normal', weight = 'normal'):
    
    font = {'family' : family,
        'weight' : weight,
        'size'   : size}

    mpl.rc('font', **font)
    
def TwinAxis(ax, LHSLabel = '', LHSCol = 'red', RHSLabel= '',  RHSCol = 'blue'):
    ax.tick_params('y', colors=LHSCol)
    ax.set_ylabel(LHSLabel, color=LHSCol)
    
    ax2 = ax.twinx()
    ax2.tick_params('y', colors=RHSCol)
    ax2.set_ylabel(RHSLabel, color=RHSCol)
    return ax2

def tightLayout_with_suptitle():
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
def saveFigure(figName, DPI = 150, bbox='tight'):
    plt.savefig(figName, dpi = DPI, bbox_inches=bbox)
    

def SubFolders_in_Folder(rootFolder):
    d=rootFolder
    subFolders = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
    return subFolders

def FilesInFolder(DirectoryPath, fileType, starts = ""):
    files = os.listdir(DirectoryPath)
    shots = []
    for i in files:
#        print( i
        if not i.startswith('.') and i.endswith(fileType) and i.startswith(starts):
            shots.append(i)
    return shots

def sortArrSplice(arr, start, end):
    sortArr = []
    for a in arr:
        if a[start:end].isdigit():
            sortArr.append(float(a[start:end]))
        else:
            sortArr.append(a[start:end])
    return sortArr


def SpliceArr(arr, start, end):
    splicedArr = []
    for a in arr:
        splicedArr.append(a[start:end])
    return splicedArr

def SplitArr(arr, stringSplit, indexToReturn=0):
    splitArr = []
    for a in arr:
        item = a.split(stringSplit)[indexToReturn]
        if item.isdigit():
            splitArr.append(float(item))
        else:
            splitArr.append(item)
    return splitArr

def nearposn(array,value):
    if type(array) == "list":
        posn = (abs(array-value)).idxmin()
    else:
        posn = (abs(array-value)).argmin()
    return posn

def sortArrAbyB(A, B):
    B, A = zip(*sorted(zip(B, A)))
    return np.array(A), np.array(B)

#==============================================================================
# Returns the R^2 value of the fit which has been calculated using curve_fit
#==============================================================================
def rSquared_from_curveFit(function, xdata, ydata, popt):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    residuals = ydata- function(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def rSquared_twoArrays(y1, y2):
    residuals = y1 - y2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y1-np.mean(y1))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

    


#==============================================================================
# A function that replicates os.walk with a max depth level
#==============================================================================
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

#==============================================================================
# Creates a list of the folders of interest
#==============================================================================
def listFolders(mainDir):
    listSubFolders =  [x[0] for x in walklevel(mainDir)][1:]
    #Modify so useable path
    for i in range(len(listSubFolders)):
        listSubFolders[i] += '/'
    return listSubFolders

def normaliseArr(arr):
    arr = np.array(arr)
    arr = arr - arr.min()
    return arr / arr.max()
        
def arr_max_to_1(arr):
    arr = np.array(arr)
    return arr / arr.max()

def momentumToMeV(px):
    m_e = 9.11e-31
    q_e= 1.6e-19
    EnergyJ = px ** 2 / (2 * m_e)
    Energy_eV = EnergyJ / q_e
    Energy_MeV = Energy_eV  * 1e-6
    return Energy_MeV 

def E_Joules_to_MeV(EnergyJ):
    m_e = 9.11e-31
    q_e= 1.6e-19
    Energy_eV = EnergyJ / q_e
    Energy_MeV = Energy_eV  * 1e-6
    return Energy_MeV 

def Load3CurveVisitOutput(fileName, arrOut = 0):
    curve = open(fileName)
        
    ca = []
    cb = []
    cc = []
    CurveNo = 0
    for line in curve.readlines():
        if line.startswith('#'):
            CurveNo+=1
        else:               
            if int(CurveNo) == 1:
                ca.append([float(line.split(' ')[0]), float(line.split(' ')[1])])
            elif int(CurveNo) == 2:
                cb.append([float(line.split(' ')[0]), float(line.split(' ')[1])])
            elif int(CurveNo) == 3:
                cc.append([float(line.split(' ')[0]), float(line.split(' ')[1])])
            else:                     
                print( "Python error: ", type(CurveNo),  CurveNo, CurveNo == 1, CurveNo ==2, CurveNo==3)
    
    assert(int(CurveNo) == 3), "Unexpected number of curves!"
    
    if arrOut == 0:
        out = np.array(ca)
    elif arrOut ==1:
        out = np.array(cb)
    elif arrOut == 2:
        out = np.array(cc)
    return out

def errorbar(x, y, xerr=0, yerr=0, color = None, label = '',  
             marker='o', linestyle='None',  ax = None, lw = 1, 
                        capSize = 5, capWidth = 2, plot_line = True):
    ax = ax if ax is not None else plt.gca()
    
    if color is None:
        color = ax._get_lines.get_next_color()
        
    (_, caps, _) = ax.errorbar(x, y, 
             yerr=yerr, xerr=xerr,  marker = marker, linestyle=linestyle,
             label = label,  
             color = color, 
             capsize=capSize)
    if plot_line:
        ax.plot(x, y, lw = lw, color = color)
    for cap in caps:
        cap.set_markeredgewidth(capWidth)
        
def errorfill(x, y, yerr, color=None, label = '', nstd = 2,
              alpha_fill=0.6, lw = 1, ax=None, linestyle='-', marker = ""):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
        
    # There are 3 different ways of entering the error, one number, two numbers 
    # or matching array in size
    option_1_number = False
    if np.isscalar(yerr):
        option_1_number = True
    else:
        yerr = np.array(yerr)

    if not option_1_number:
        try:
            if len(yerr) == len(y):
                option_1_number = True
        except TypeError: #len() of unsized object
            pass
    if option_1_number:
        # ymin1 = y - yerr
        # ymax1 = y + yerr
        # ymin2 = y - yerr*2
        # ymax2 = y + yerr*2
        # ax.fill_between(x, ymax1, ymin1, color=color, alpha=alpha_fill)        
        # ax.fill_between(x, ymax2, ymin2, color=color, alpha=alpha_fill*0.5) 

        alpha_Values =  np.linspace(alpha_fill, alpha_fill * 0.5, num = nstd)
        for i in range(1, nstd + 1):            
            ymin1 = y - yerr * i
            ymax1 = y + yerr * i
    
            ax.fill_between(x, ymax1, ymin1, color=color, alpha=alpha_Values[i-1])           
        
    elif not option_1_number:
        if len(yerr) == 2:
            ymin, ymax = yerr
            ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)        
    ax.plot(x, y, color=color, lw = lw, linestyle = linestyle, marker = marker,
            label=label)

def imshow_with_lineouts(image, fitlerSize = 3, CropY=None):
    import matplotlib.gridspec as gridspec
    from scipy.signal import medfilt
    
        #Sum in each direction for lineouts
    sumX = []
    x = []
    for i, im in enumerate(image.T):
        sumX.append(sum(im))
        x.append(i)
    sumY = []
    y = []
    for i, im in enumerate(image[None:]):
        sumY.append(sum(im))
        y.append(i)
        
    fig = plt.figure(figsize=(14,8))
    # 3 Plots with one major one and two extra ones for each axes.
    gs = gridspec.GridSpec(4, 4, height_ratios=(1,1,1,1), width_ratios=(0.5,1,1,1))
    gs.update(wspace=0.025, hspace=0.025)
    
    #    Create all axis, including an additional one for the cbar
    ax1 = plt.subplot(gs[0:3, 1:-1])             # Image
    ax1.axis('off')
    ax2 = plt.subplot(gs[0:3, 0], sharey=ax1) # right hand side plot
    ax3 = plt.subplot(gs[-1, 1:-1], sharex=ax1 ) # below plot
    
    cax4 = fig.add_axes([0.7, 0.35, 0.05, 0.5])
    
    im = ax1.imshow(image)
    ax3.plot(x, medfilt(sumX, fitlerSize))
    ax2.plot(medfilt(sumY, fitlerSize), y)
    ax3.set_xlabel('')
    ax2.set_ylabel('')
    plt.colorbar(im, cax = cax4)
    
    
def pcolormesh_with_lineouts(X, Y, image, fitlerSize = 3, CropY=None, 
                             cmap = plt.cm.bwr, sciAxis = True, xlabel = "", ylabel = "", 
                             xlims = None, ylims = None, figsize=(14,8),
                             norm = None,
                             title = ""):
    import matplotlib.gridspec as gridspec
    from scipy.signal import medfilt
    
    image = np.array(image)
    setup_figure_fontAttributes(size = 14)
        #Sum in each direction for lineouts
    sumX = []
    for i, im in enumerate(image.T):
        sumX.append(np.average(im))
    sumY = []
    for i, im in enumerate(image[None:]):
        sumY.append(np.average(im))
        
    fig = plt.figure(figsize=figsize)
    # 3 Plots with one major one and two extra ones for each axes.
    gs = gridspec.GridSpec(4, 4, height_ratios=(1,1,1,1), width_ratios=(0.5,1,1,1))
    gs.update(wspace=0.025, hspace=0.025)
    
    #    Create all axis, including an additional one for the cbar
    ax1 = plt.subplot(gs[0:3, 1:-1])             # Image
    ax1.axis('off')
    ax2 = plt.subplot(gs[0:3, 0], sharey=ax1) # right hand side plot
    ax3 = plt.subplot(gs[-1, 1:-1], sharex=ax1 ) # below plot
    ax2.set_ylim([Y[0], Y[-1]])
    if xlims is not None:
        ax2.set_xlim([xlims[0], xlims[1]])
    
    if ylims is not None:
        ax3.set_ylim([ylims[0], ylims[1]])
    
    ax3.set_xlim([X[0], X[-1]])
    if sciAxis:
        ax2.ticklabel_format(style='sci', axis='y' , scilimits=(0,0))
        ax3.ticklabel_format(style='sci', axis='x' , scilimits=(0,0))        

    ax2.set_ylabel(ylabel)
    ax3.set_xlabel(xlabel)
    
    cax4 = fig.add_axes([0.7, 0.35, 0.05, 0.5])
    # cax4 = plt.subplot(gs[1:-1, -1])
    
    if norm is not None:
        im = ax1.pcolormesh(X, Y, image, cmap = cmap, norm = norm)
    else:
        im = ax1.pcolormesh(X, Y, image, cmap = cmap)
    ax3.plot(X, medfilt(sumX, fitlerSize))
    ax2.plot(medfilt(sumY, fitlerSize), Y)
    plt.colorbar(im, cax = cax4)    
    plt.suptitle(title)
    
    ax = [ax1, ax2, ax3, cax4]
    return ax, np.c_[X, sumX], np.c_[Y, sumY] 
            

def lin(x, *params):
    m = params[0]
    c = params[1]
    return m*x + c

def gaus(x, *params):
    #Gaussian function
    A = params[0]
    x0 = params[1]
    w = params[2]
    c = params[3]
    return A*np.exp(- 0.5 * ((x-x0)/w)**2) + c

def gaus_fwhm(popt):
    if type(popt) in [int, float, np.float, np.float32, np.float64, np.float128]:
        sigma = popt
    elif len(popt)==4:
        sigma = popt[2]
    else:
        print('Gausian fwhm input error', popt)
        return None
    
    return 2 * sigma * (2 * np.log(2))**0.5



def add_one_more_step_to_axis(arr):
    """ Extend a list by one step.
    This is useful for the pcolormesh plots
    """
    if len(arr) > 1:
        stepSize = arr[-1] - arr[-2]
        new_val = arr[-1] + stepSize
    else:
        new_val = arr[0] + 1
    if type(arr) is not list:
        	arr = arr.tolist()
    arr.append(new_val)
    return arr
