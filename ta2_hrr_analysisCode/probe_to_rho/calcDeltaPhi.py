#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Jan 13 16:05:40 2020

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import CUnderwood_Functions3 as func

from loadData import loadInDataToNumpy
from imageCalculations import probe_image_analysis
from cleanImage import filter_image



class phaseShift(probe_image_analysis, filter_image):   
    def __init__(self):
        print ("Creating phaseShift class")
        
    def plot_raw_input(self):
        """ Plot the raw image with a colorbar
        """
        if self.refExists:
            f, ax = plt.subplots(ncols = 2)
            plt.suptitle("Input Data")
            self.plot_data(self.im, show=False, ax = ax[0])
            self.plot_data(self.ref, show=False, ax = ax[1])            
            # im1 = ax[0].pcolormesh(self.im , cmap = plt.cm.seismic)
            # im2 = ax[1].pcolormesh(self.ref, cmap = plt.cm.seismic)
            # plt.colorbar(im1, ax = ax[0])
            # plt.colorbar(im2, ax = ax[1])                
            ax[0].set_title("Im")
            ax[1].set_title("Ref")                
            plt.show()
        else:
            self.plot_data(self.im , cmap = plt.cm.seismic)
            
    def draw_outline(self, bot, top, left, right, ax = None):
        ''' Draw lines to show where the cropping is happening.
        '''
        if ax == None:
            ax = plt.gca()
        for y in [top, bot]:
            ax.hlines(y, left, right, color = "black")
        for x in [left, right]:
            ax.vlines(x, top, bot, color = "black" )              

    
    def load_arrIntoClass(self, im, ref = None, plotting = False):
        """ Load data into the class from arrays
        """
        self.im = im
        self.imShape = np.shape(self.im)
        if ref is not None:
            self.ref = ref
            self.refExists = True
        else:
            self.refExists = False

        if plotting:
            self.plot_raw_input()
    
    def load_data(self, loadPath, imFile, refFile = None, plotting = False):
        """ Load data into the class from file
        """        
        
        self.im = self.loadData(loadPath + imFile)
        self.imShape = np.shape(self.im)
        if refFile is not None:
            self.ref = self.loadData(loadPath + refFile)
            self.refExists = True
        else:
            self.refExists = False
        if plotting:
            self.plot_raw_input()        
            
    def zeroPadImages(self, padSize = 100):
        """ Pads an image with zeros
        """        
        self.padSize = padSize
        self.im_PlasmaChannel = np.pad(self.im_PlasmaChannel, padSize, 'constant')    
        if self.refExists:
            self.ref_PlasmaChannel = np.pad(self.ref_PlasmaChannel, padSize, 'constant')    
        
    
    def cropToPlasmaChannel(self, pc_crop_coors, plotting = False, 
                            paddingX = 30, paddingY = 10, 
                            padSize = 100, centreCoorsOfFringeRegion = [500, 200],
                            verbose = False):
        """ Crop to the plasma channel, using the four corner coors
        The crop is done with a window function to remove the fourier efffect of
        sharp edges
            # Would like to add choice of windows
        
        """     
        bot, top, left, right = self.check_btlr_coors_in_image(pc_crop_coors, self.imShape)

        # # Find the range and centres of the box marking the PC
        xr = (right-left)
        yr = (top-bot)
        self.pc_xsize = xr
        self.pc_ysize = yr
        
        self.paddingX = paddingX
        self.paddingY = paddingY    
        self.padSize = padSize           
        if plotting:
            self.draw_outline(bot, top, left, right)
            self.plot_data(self.im)
        power = 2*6
        gaus_cropping = self.createGaussianCroppingWindow(self.im, [bot, top , left, right], power)
        if False:
            plt.title("Gaussian cropping window")
            self.plot_data(gaus_cropping)
        self.im_PlasmaChannel = self.im * gaus_cropping
        if self.refExists:
            self.ref_PlasmaChannel = self.ref * gaus_cropping
        else:
            # Use a region where the fringes are unperturbed as the reference
            self.crop_reference_fringes(self.padSize, self.paddingX, self.paddingY, 
                                        centreCoorsOfFringeRegion)
        
        
        b_pc, t_pc, l_pc, r_pc = bot - paddingY, top + paddingY, left - paddingX, right + paddingX
        b_pc, t_pc, l_pc, r_pc = self.check_btlr_coors_in_image([b_pc, t_pc, l_pc, r_pc], self.imShape)
        
        # Crop to the new region of the image
        self.im_PlasmaChannel = self.im_PlasmaChannel[b_pc:t_pc, l_pc:r_pc]
        if self.refExists:
            self.ref_PlasmaChannel = self.ref_PlasmaChannel[b_pc:t_pc, l_pc:r_pc]           

        # Pad the image with zeros
        self.zeroPadImages(padSize)
        
        # The refence has now been created
        self.refExists = True        
        
        if plotting:
            plt.clf()
            print ("Plotting the raw images cropped with a window")
            plt.title("Raw image with Gauss window applied and padding")
            plt.pcolormesh(self.im_PlasmaChannel , cmap = plt.cm.seismic, norm = func.MidpointNormalize(midpoint=0))
            plt.axes().set_aspect('equal')
            plt.colorbar()            
            plt.show()       
            
    def fft_of_plasma(self, plotting = True):
        ''' Do 2D fft '''
        self.F_im_PC = np.fft.fft2(self.im_PlasmaChannel)
        self.F_imShape = np.shape(self.F_im_PC)
        self.F_im_PC = self.wrapImage(self.F_im_PC)
        if self.refExists:
            self.F_ref_PC = np.fft.fft2(self.ref_PlasmaChannel)
            self.F_ref_PC = self.wrapImage(self.F_ref_PC)
                    
        if plotting:    
            plt.pcolormesh( abs(self.F_im_PC), cmap = plt.cm.seismic, norm = mpl.colors.LogNorm())
            plt.colorbar()
            plt.title("Im PC FFT")
            plt.show()
            
    def plot_FT_space(self, bot, top, left, right, peakRelHeight):
        lenData = self.F_im_PC.shape[1]
        xData = range(lenData)
        yData = abs(self.F_im_PC).sum(axis = 0)


        from scipy.signal import find_peaks       
        l = abs(self.F_im_PC).sum(axis = 0)
        peaks = find_peaks( l , 
                      # width = 20
                      height = l.max() * peakRelHeight
                      )
        print ("Peaks in f space", peaks)

        f, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 6), sharex = True, 
                             gridspec_kw={'height_ratios': [3,1]})
        ax[0][0].imshow(abs(self.F_im_PC), cmap = plt.cm.seismic, aspect = 'auto'
                    , norm=mpl.colors.LogNorm()
                    )
        ax[1][0].plot(xData, yData)
        self.draw_outline(bot, top, left, right, ax = ax[0][0])
        ax[1][0].set_xlim([xData[0], xData[-1]])
        ax[0][1].plot(peaks[0], peaks[1]['peak_heights'], "x")
        ax[0][0].set_title("The crop region in F space")
        ax[0][1].set_title("The crop region for reference")

        ax[0][1].imshow(abs(self.F_ref_PC), cmap = plt.cm.seismic, aspect = 'auto'
                    , norm=mpl.colors.LogNorm()
                    )  
        ax[1][1].plot(abs(self.F_ref_PC).sum(axis = 0))
        
        func.tightLayout_with_suptitle()
        plt.show()
        
    def plot_cropWindow_and_cropped_Fpeak(self, bot, top, left, right, gaussCrop):
        f, ax = plt.subplots(nrows = 2, sharex = True, gridspec_kw={'height_ratios': [1, 3]} )
        ax[0].set_title("Cropping window")            
        ax[1].set_title("Cropped peak in Fourier space")
        im_c = ax[0].pcolormesh(gaussCrop)
        self.draw_outline(bot, top, left, right, ax = ax[0])     
        self.draw_outline(bot, top, left, right,  ax = ax[1])

        im = ax[1].pcolormesh( abs(self.F_cropped_im)  , cmap = plt.cm.viridis
                , norm=mpl.colors.LogNorm(), vmin  = 1
                ) 
        plt.colorbar(im_c, ax = ax[0])                    
        plt.colorbar(im, ax = ax[1])        
        plt.show()
            
    def crop_to_FFT_peak(self, crop_coors, GausPower = 6,  plot_crop_window_and_peak = True,
                         plot_fft_space = True, peakRelHeight = 0.3):
        
        F_shape = np.shape(self.F_im_PC)
        bot, top, left, right = self.check_btlr_coors_in_image(crop_coors, F_shape)
        self.F_cropped_im = np.zeros_like(self.F_im_PC)
        
        gaussCrop = self.createGaussianCroppingWindow(self.F_im_PC, [bot, top, left, right],
                                                      GausPower)
        self.F_cropped_im = gaussCrop * self.F_im_PC
        if self.refExists:
            self.F_cropped_ref = gaussCrop * self.F_ref_PC        

        if plot_fft_space:
            self.plot_FT_space(bot, top, left, right, peakRelHeight)
            
        if plot_crop_window_and_peak:
            self.plot_cropWindow_and_cropped_Fpeak(bot, top, left, right, gaussCrop)
            
            
    def find_peaks_in_lineouts(self, peakRelHeight):
        """ Create a lineout of the image and find the peaks
        """
        from scipy.signal import find_peaks           
        x_range = np.arange(self.F_im_PC.shape[0])
        xData = abs(self.F_im_PC).sum(axis = 0)
        
        y_range = np.arange(self.F_im_PC.shape[1])
        yData = abs(self.F_im_PC).sum(axis = 1)
        
        # Locate the peaks
        xpeaks = find_peaks(xData , height = xData.max() * peakRelHeight)         
        ypeaks = find_peaks(yData , height = yData.max() * peakRelHeight)  
        # Take the first and last peak to be the peaks of interest. Tune the variable peakRelHeight
        # so this is the case
        if len(xpeaks[0]) > 2:
            xpeaks = [ [xpeaks[0][0], xpeaks[0][-1]], 
                      {'peak_heights': [xpeaks[1]['peak_heights'][0], xpeaks[1]['peak_heights'][-1]]} ]

        return x_range, xData, y_range, yData, xpeaks, ypeaks
        
            
    def auto_select_FT_peak(self, GausPower = 6, peakRelHeight = 0.6, yPercRange = 0.3, xPercRange = 0.3, 
                            plot_fft_space = True, plotting_cropping = True, plot_found_peaks = True):
        """ Auto select the FT crop region from the peak locations
        """
        F_shape = np.shape(self.F_im_PC)
        self.F_cropped_im = np.zeros_like(self.F_im_PC)

        # Find the peaks in the lineouts
        x_range, xData, y_range, yData, xpeaks, ypeaks = self.find_peaks_in_lineouts(peakRelHeight)        
        if plot_found_peaks:        
            plt.figure(figsize = (6, 4))
            plt.title("Finding the peaks in the FFT of the image")
            plt.plot(x_range, xData, label = "x")
            plt.plot(y_range, yData, label = "y")
            plt.plot(xpeaks[0], xpeaks[1]['peak_heights'], "x")
            plt.plot(ypeaks[0], ypeaks[1]['peak_heights'], "x")
            plt.legend()
            plt.show()        

        xrange = xpeaks[0][1] - xpeaks[0][0]
        yrange = F_shape[0]
        
        left = xpeaks[0][1] - xPercRange * xrange
        right = xpeaks[0][1] + xPercRange * xrange                
        bot = ypeaks[0][0] - yrange * yPercRange
        top = ypeaks[0][0] + yrange * yPercRange
        bot, top, left, right = self.check_btlr_coors_in_image([bot, top, left, right], F_shape)

        print ("Found crop coors", bot, top, left, right)
        self.crop_to_FFT_peak([bot, top, left, right])
        
        return bot, top, left, right 
            
    
    def createPhase_inverseFT(self, plotting = False):
        self.result_im = np.fft.ifft2(self.F_cropped_im)
        
        if self.refExists:
            self.result_ref = np.fft.ifft2(self.F_cropped_ref)
            self.phaseShift = np.angle(self.result_im / self.result_ref)
        else:
            self.phaseShift = np.angle(self.result_im)
        
        if type(self.padSize) ==  list:
            padSizeX = self.padSize[1][0]
            padSizeY = self.padSize[0][0]
        else:
            padSizeX = padSizeY = self.padSize
        
        self.phaseShift = self.phaseShift[padSizeY:-padSizeY,
                                          padSizeX:-padSizeX ]
        self.phaseShift = self.phaseShift[self.paddingY:-self.paddingY,
                                          self.paddingX:-self.paddingX]
        if plotting:
            plt.title("Phase Shift Due to Plasma")
            plt.pcolormesh(self.phaseShift , cmap = plt.cm.seismic, vmin = -np.pi, vmax = np.pi)
            cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
            cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
            plt.show()

    def move_CroppedRegion_To_Centre(self, plotting = True):
        F_lineout = np.average(abs(self.F_cropped_im), axis = 0)
        maxInd = func.nearposn(F_lineout, F_lineout.max())
        
        if plotting:
            f, ax = plt.subplots(nrows = 2, sharex = True, figsize = (6, 3), gridspec_kw={'height_ratios': [1, 3]})
            ax[0].set_title("Peak in fourier space before moving")
            ax[0].plot(F_lineout)
            ax[1].pcolormesh(abs(self.F_cropped_im), norm = mpl.colors.LogNorm(), vmin = 1)
            plt.show()        
            
        self.F_cropped_im = self.moveCenterOfImage(self.F_cropped_im, 0, self.F_imShape[1]//2 - maxInd)
        if self.refExists:
            self.F_cropped_ref = self.moveCenterOfImage(self.F_cropped_ref, 0, self.F_imShape[1]//2 - maxInd)            

        if plotting:
            F_lineout = np.average(abs(self.F_cropped_im), axis = 0)
            f, ax = plt.subplots(nrows = 2, sharex = True, figsize = (6, 3), gridspec_kw={'height_ratios': [1, 3]})
            ax[0].set_title("Peak in fourier space after moving")            
            ax[0].plot(F_lineout)
            ax[1].pcolormesh(abs(self.F_cropped_im), norm = mpl.colors.LogNorm() , vmin = 1)
            plt.show()       
    
        self.F_cropped_im = self.wrapImage(self.F_cropped_im)
        if self.refExists:
            self.F_cropped_ref = self.wrapImage(self.F_cropped_ref)
            
    def unwrap_phase(self, plotting = False):
        
        self.phase = unwrap_phase(self.phaseShift) 
        if plotting:
            self.plot_data(self.phase)

# =============================================================================
# From here adding to get the creation of fringes
# =============================================================================
        
    def crop_reference_fringes(self, channelPadding , xpad , ypad, safeRegion_Centre, plotting = False):
        ''' Create a reference from the unperturbed region in the interferogram
        This needs to be same size as the actual image, so the fourier space
        treatment of it works.
            A safe region is chose, which at the moment is in the top left of 
        the image where nothing should be blocking the fringes.
        '''
        xc, yc = safeRegion_Centre
        left = xc - self.pc_xsize // 2
        right = xc + self.pc_xsize // 2
        bot = yc - self.pc_ysize // 2
        top = yc + self.pc_ysize // 2
        
        bot, top, left, right = self.check_btlr_coors_in_image([bot, top, left, right],
                                     self.imShape)
        if plotting: 
            plt.title("Reference region")
            self.draw_outline(bot, top, left, right)
            self.plot_data(self.im)
            
        # # Find the range and centres of the box marking the PC
        xr = self.pc_xsize
        yr = self.pc_ysize
        
        self.refData = self.im * 1.0
        sgCrop = self.crop_to_ref([bot, top, left, right]
                , xr, yr, channelPadding, xpad , ypad)
            
        if plotting and False: 
            plt.figure(figsize=(5,3.5))
            plt.title('The cropped and padded referce')
            self.plot_data(self.ref_PlasmaChannel, show = False, cmap = 'twilight',
                           norm = func.MidpointNormalize(midpoint=0))
            plt.show()        
                
        print ('Reference Created')
        return sgCrop        
    
            
    def shift_centerofCrop_to_Centre(self, arr, bot, top, left, right):
        '''Shift the image in the crop region to the center of the arr.   
        '''
        shape = abs(arr).shape # abs to stop it being complex
        xr = (right-left)
        yr = (top-bot)
        xshift = int(left + 0.5 * xr ) - shape[1] // 2
        yshift = int(bot + 0.5 * yr) - shape[0] // 2     
        arr = self.moveCenterOfImage(arr, -yshift, -xshift)      
        return arr    

    def crop_to_ref(self, btlr, xr, yr, channelPadding, xpad , ypad):
        ''' Crop the reference image in the same way as the main image.
        '''
        bot, top, left, right = self.check_btlr_coors_in_image(btlr, self.imShape)
        centeredReference = self.shift_centerofCrop_to_Centre(self.refData, 
                                                        bot, top, left, right)   
        # Shifting the crop region to match the shifting of the center of the image    
        bot, top, left, right = self.centerCropRegionOnImage(xr, yr)        
        self.rawReference = centeredReference[bot:top, left:right]
   
        sgCrop = self.createGaussianCroppingWindow(centeredReference, [bot, top, left, right])
        
        # croppedImage = centeredPC * (1- sgCrop) # Debugging option, invert the crop
        croppedImage = centeredReference * sgCrop
        self.ref_PlasmaChannel = self.crop_with_padding(croppedImage, 
                            bot, top, left, right, channelPadding, xpad , ypad)  
        return sgCrop

    def centerCropRegionOnImage(self, xr, yr):
        '''Shifting the crop region to match the shifting of the center of the image    
        '''
        btlr = [self.imShape[0] // 2 - yr //2, self.imShape[0] // 2 + yr //2, 
                self.imShape[1] // 2 - xr //2, self.imShape[1] // 2 + xr //2]
        return btlr
    
    def crop_with_padding(self, croppedImage, bot, top, left, right, channelPadding, xpad , ypad):
        ''' New crop with padding even around all sides!
        '''
        bot, top, left, right = self.check_btlr_coors_in_image(
            [bot-channelPadding, top+channelPadding, left-channelPadding,right+channelPadding], 
                                   self.imShape)
        # The cropped and padded image.
        croppedImage = croppedImage[bot: top, left: right]
        # Additional padding, can be different in each direction
        croppedImage = np.pad(croppedImage, [(ypad, ), (xpad, )], mode='constant')   
        return croppedImage    


if __name__ == "__main__":
    loadPath = "/Users/chrisunderwood/Documents/Experimental_Tools/Probe_Interfer_Analysis/Trial_Images/"
    refFile = '20190913r016b6s50_clean.txt'
    imFile = '20190913r016b6s15_clean.txt'
    # imFile = '20190913r016b6s15.TIFF'
    refFile = None
    
    pc_crop_coors = [300, 500, 1100, 1210] # bot, top, left, right
     #
    
    ps = phaseShift()
    # try:
    #     ps.im = im * 1.0
    #     # ps.ref = ref * 1.0
    #     # ps.refExists = True
    #     ps.refExists = False
    #     ps.imShape = ps.im.shape
    # except NameError:
    if True:
        ps.load_data(loadPath, imFile, refFile,  plotting = True)
        im = ps.im * 1.0
        if imFile.endswith('TIFF'):
            ps.blur(35)
        
        # ref = ps.ref * 1.0
        # ps.plot_data(ps.im)
        
        
    # Crop to the plasma channel
    ps.cropToPlasmaChannel( pc_crop_coors, plotting=True,
                            paddingX=20, paddingY=20, # Padding on the crop
                            # Extra pad  y      x
                            # padSize = [(10, ), (10, )]
                            padSize = 10,
                            
                           )    
    ps.fft_of_plasma(plotting = True)

    if True:    
        # FT_crop_coors = [50, 140, 250, 330] # bot, top, left, right
        # FT_crop_coors = [64, 268, 128, 135] # bot, top, left, right
        FT_crop_coors = [49, 205, 87, 92]
    
        ps.crop_to_FFT_peak(FT_crop_coors)
    else:
        ps.auto_select_FT_peak(yPercRange = 0.3, xPercRange = 0.25, plot_fft_space = True, plotting_cropping = False, plot_found_peaks = False)
        #### ps.move_CroppedRegion_To_Centre(plotting = False) 
    
    ps.createPhase_inverseFT(plotting = True)
    ps.unwrap_phase()   
    
    np.savetxt(loadPath + imFile.split(".")[0] + '_phi.txt', ps.phase)
    
    
