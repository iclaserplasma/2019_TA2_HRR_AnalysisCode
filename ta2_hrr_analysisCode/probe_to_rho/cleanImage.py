#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Jan  7 14:46:10 2020

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from loadData import loadInDataToNumpy

import CUnderwood_Functions3 as func

class filter_image(loadInDataToNumpy):        
    def blur(self, sigSize, plotting = True):
        """ Blur the image with a gaussain filter to create a bg to remove
        This removes large order deffects in the beam, but should leave the fringes.
        Make sure that sigSize is large enough
        """
        import scipy.ndimage as ndimage
        if hasattr(self ,'im_bgrm'):
            arr  =self.im_bgrm
        else:
            arr = self.im
        self.im_gblur = ndimage.gaussian_filter(arr,
                  sigma=(sigSize, sigSize), order=0)

        print( type(arr), type(self.im_gblur))

        if plotting:
            plt.imshow(self.im_gblur, 
                       # vmin = 33000
                       )
            plt.title(sigSize)
            plt.show()        
            
        self.im = np.array(arr - self.im_gblur, 
                           dtype = float)
        
    def sub_blurredIm(self, btlr, plotting = True):
        """ Subtract the blurred image from the raw image.
        If bot, top, left, right are given, also crops to a larger region of interest
        """
        imageRange = 1000
        [bot, top, left, right] = btlr
        if hasattr(self, 'im'):
            arr = self.im
        else:
            arr = self.data
        
        background = np.sum(arr[bot: top, left: right])
        print ('background', background)
        self.im_bgrm = arr - background
        if plotting:
            if not None in [bot, top, left, right]:
                plt.vlines(left, bot, top)
                plt.vlines(right, bot, top)
                plt.hlines(top, left, right)
                plt.hlines(bot, left, right)
            for y in top, bot:
                for x in left, right:
                    plt.plot([x], [y], 'x')
            
            self.plot_data(self.im_bgrm)
        
    def locate_regions(self, medianFilterKernel = 13, plotting = True):
        from scipy.signal import medfilt, find_peaks, find_peaks_cwt
        from scipy.ndimage import gaussian_filter1d

        # image_lineout = self.data.sum(axis=0)
        # grad = np.gradient(image_lineout)
        
        # plt.plot(func.normaliseArr(image_lineout))
        # plt.plot(func.normaliseArr(medfilt(abs(grad), medianFilterKernel)))
        dTrial = self.data
        delIm = dTrial.max()  - dTrial.min()
        maskedIm = np.array(dTrial > dTrial.min() + delIm * 0.15, dtype=int)
        if plotting:
            plt.imshow(maskedIm)
            plt.colorbar()
            plt.show()
        line = func.normaliseArr(maskedIm.sum(axis=0))
        grad = func.normaliseArr(np.gradient(line))
        plt.plot(line , lw = 2.5)
        plt.plot(grad )
        for i, point in enumerate(line[::-1]):
            if point > 0.4:
                break
        indexCylinder = len(line) - i
        print (indexCylinder)
        plt.imshow(dTrial)
        plt.vlines(indexCylinder, 0, 1200, 'b')
        plt.show()        
        
        
        newLine = func.normaliseArr(line[:indexCylinder])
        newGradient = gaussian_filter1d(func.normaliseArr(np.gradient(newLine)), 11)
        plt.plot(newLine)
        plt.plot(newGradient)
        plt.show()
        
        
        # plt.imshow(dTrial)
        # plt.vlines(indexCylinder, 0, 1200, 'b')
        # plt.vlines(p[-1], 0, 1200, 'r')        
        # plt.show()
        
        p1 = find_peaks_cwt(-newGradient, widths= np.arange(10, 60), min_snr = 1.5)
        p = find_peaks_cwt(-newLine, widths= np.arange(10, 60))
        index = []
        for gradPeak in p1:
            index.append(p[func.nearposn(p, gradPeak)])
            
        plt.plot(newLine)
        plt.plot(index, newLine[index], 's')
        plt.show()

        plt.imshow(dTrial)
        for i in index:
            plt.vlines(i, 0, 1200, 'b')        
        plt.vlines(indexCylinder, 0, 1200, 'r')
        plt.vlines(indexCylinder-p1[-1] + p1[-2], 0, 1200, 'r')            
        plt.show()        
        
if __name__ == "__main__":
    mpl.rcParams['figure.figsize'] =[6.0, 6.0]
    # mpl.rcParams['figure.figsize'] = [10,8]
    #Load original data
    LoadPath = "Trial_Images/"
    # fileName = "20190904r010s5.TIFF"
    # fileName= "20191008r008s3.TIFF"
    # fileName= "20191004r013s2.TIFF"
    fileName = "20190913r016b6s15.TIFF"
    
    
    # Reference Images
    fileName = "20190913r016b6s50.TIFF"
    
    
    LoadPath + fileName
        # Call the class
    plt.title("Raw input data")
    bc = filter_image()
    bc.loadData(LoadPath + fileName)
    bc.plot_data()
    # for i in range(10, 100, 10):
    #     bc.blur(i)
    bc.blur(35)
    
    # bc.sub_blurredIm([1000, 1200, 50, 400])
    bc.plot_data(bc.im, norm = func.MidpointNormalize(midpoint= 0), cmap = 'seismic')
    
    # mpl.rcParams['figure.figsize'] = [6,4]
    # for medF in np.arange(11, 102, 10):
    #     plt.title(medF)
    #     bc.locate_regions(medianFilterKernel = medF)
    #     plt.show()
    
    # bc.locate_regions(medianFilterKernel = 51)
    np.savetxt("Trial_Images/{}_clean.txt".format(fileName.split('.')[0]), bc.im)


    '''
    # Testing the frequency of the fringes
    # Take a lineout and check that the fringes in the cell match the fringes 
    # outside the cell
    b = bc.data
    for ind in [700, 730, 760 ]:
        l = b[ind]
        plt.title("Line index {}".format(ind))
        plt.plot(l[100:500])
        plt.plot(l[1100:1500] + 600)
        plt.show()
    plt.imshow(b)
    '''
    mpl.rcParams['figure.figsize'] = [10,8]
    
    from scipy.signal import medfilt
    conv = []
    for i in range(1200):
        l = bc.im[i]
        #plt.plot(l)
        gl = medfilt(np.gradient(medfilt(l, 5)), 11 )
        # plt.plot( gl )
        cl = np.convolve(abs(gl), np.ones(15))
        conv.append(cl)
    conv = np.array(conv)
    
    plt.imshow(conv, aspect='auto')
    l = func.normaliseArr(conv.sum(axis = 0)) 
    plt.plot(l * conv.shape[0] , 'r')
    plt.colorbar()
    plt.show()
    