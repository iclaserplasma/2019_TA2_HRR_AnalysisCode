#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
""" 
       _
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
Modified on 25/02/2020, 13:10:52
@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

# Load my module of functions
import CUnderwood_Functions3 as func

# Module to import
from skimage.restoration import unwrap_phase
import abel
import cv2
from scipy.signal import find_peaks       
from scipy.optimize import curve_fit


class loadInDataToNumpy():
    ''' Class to load data into python.
    Deals with any format that is required, and makes a np array.
    '''
    def loadData(self, filePath):
        """ Loads the data
        This looks at the file extension and loads it appropiately
        """
        if filePath.endswith(".png"):
            print ("png files not yet supported")
            return None
        elif filePath.endswith(".txt"):
            # print ("Loading .txt")           
            data = np.loadtxt(filePath)
        elif filePath.endswith(".tiff") or filePath.endswith(".TIFF"):
            # print ("Loading .tiff/TIFF")
            from skimage import io
            data = io.imread(filePath)
            data = data.astype(float)
            # print (type(data)) 
        elif filePath.endswith(".tif"):
            # print ("Loading .tif")            
            from skimage import io
            data = io.imread(filePath)
            data = data.astype(float)
            # print (type(data))            
        else:
            print ("The type is not an expected file")
            print ("File: ", filePath)
            print ("Please Edit: {}\n\tTo accept this file type".format("loadDataToNumpy_class.py"))

        assert type(data) == np.ndarray
        return data
    
    def plot_data(self, arr = None, show = True, ax = None, **kwargs):
        try:
            if arr == None:
                arr = self.data
        except ValueError:
            pass
        
        if ax == None:
            ax = plt.gca()
        im = ax.imshow(arr, aspect = 'auto', **kwargs)
        plt.colorbar(im, ax = ax)
        if show:
            plt.show()



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

        # print( type(arr), type(self.im_gblur))

        if plotting:
            plt.imshow(self.im_gblur, 
                       # vmin = 33000
                       )
            plt.title(sigSize)
            plt.show()        
            
        self.im = np.array(arr - self.im_gblur, 
                           dtype = float)
        # Filter the reference file as well,
        # Only do this if the reference is the same size as the image,
        # as this means it is also just been directly loaded from file.
        if hasattr(self, 'ref'):
            if hasattr(self, 'refSameSize'):
                if self.refSameSize:
                    ref_gblur = ndimage.gaussian_filter(self.ref,
                          sigma=(sigSize, sigSize), order=0)
                    self.ref = np.array(arr - self.im_gblur, 
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
        


class probe_image_analysis():
    
    def check_btlr_coors_in_image(self,btlr, shape):
        """ Check the four indexes that they are within the image
        Corrects them to be within if too large/small
        """
        assert len(shape) == 2
        assert len(btlr) == 4
        bot, top, left, right = btlr
        bot, top, left, right = [int(bot), int(top), int(left), int(right)]
        # print (bot, top, left, right, shape)
        if bot < 0:
            bot = 0
        if left < 0:
            left = 0
        if top > shape[0]-1:
            top = shape[0]-1
        if right > shape[1]-1:
            right = shape[1]-1   
        return int(bot), int(top), int(left), int(right)    
    
    def wrapImage(self, image):
        wrappedImage = np.zeros_like(image)
        shape = np.shape(image)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Wrapping image
                i_w = (i + shape[0]//2)%shape[0]
                j_w = (j + shape[1]//2)%shape[1]
                # print 
                wrappedImage[i_w][j_w] = image[i][j]
        return wrappedImage    
    
    def moveCenterOfImage(self, image, moveX, moveY):
        wrappedImage = np.zeros_like(image)
        shape = np.shape(image)
        for i in range(shape[0]):
            for j in range(shape[1]):
                # Wrapping image
                i_w = (i + moveX)%shape[0]
                j_w = (j + moveY)%shape[1]
                # print 
                wrappedImage[i_w][j_w] = image[i][j]
        return wrappedImage    
    
    def superGaussian(self, x, x0, w, power):
        """ Super Gaussian function
        """
        return np.exp(- (2 * (x- x0)/(w))**power)    
    
    def createGaussianCroppingWindow(self, image, btlr, 
                                     power = 8):
        """ Create a window to crop in fourier space with
        """
        # print ("Creating Gaussian To Crop. Image shape", np.shape(image))
        bot, top, left, right = btlr
        cropGaussian = np.zeros_like(image, dtype = float)

        s = np.shape(image)   
        sgY = self.superGaussian(np.arange(s[1]), (left + right)*0.5, abs(right-left), power)
        sgX = self.superGaussian(np.arange(s[0]), (top + bot)*0.5, abs(top - bot), power)    

        yMask = sgY > 0.005
        xMask = sgX > 0.005
        xIndexs = []
        yIndexs = []
        for i, x in enumerate(xMask):
            if x:
                xIndexs.append(i)
        for i, y in enumerate(yMask):
            if y:
                yIndexs.append(i)       
        # Just use region of gaussians where the peak is large enough
        for i in xIndexs:
            for j in yIndexs:
                cropGaussian[i][j] = sgX[i] * sgY[j]         
        # print ("\t\tOutput shape of crop:" ,np.shape(cropGaussian))
    
        return cropGaussian        
    
    def rotate_array(self, arr, angleDeg):

        M = cv2.getRotationMatrix2D((arr.shape[0]/2, arr.shape[1]/2), 
                                    angleDeg, # Angle of rotation in Degrees
                                    1   #Scale
                                    )
        arr_Rot = cv2.warpAffine(arr ,M,(arr.shape[1] ,arr.shape[0]))
        return arr_Rot

class phaseShift(probe_image_analysis, filter_image):   
    def __init__(self, power = 6):
        # print ("Creating phaseShift class")
        self.gausPower = 2*power

        
    def plot_raw_input(self):
        """ Plot the raw image with a colorbar
        """
        if self.refExists:
            f, ax = plt.subplots(ncols = 2,sharex = True, sharey = True)
            plt.suptitle("Input Data")
            self.plot_data(self.im, show=False, ax = ax[0])
            self.plot_data(self.ref, show=False, ax = ax[1])   
            ax[0].set_xlim([0, self.im.shape[1]] )
            ax[0].set_ylim([ self.im.shape[0], 0] )
            # im1 = ax[0].pcolormesh(self.im , cmap = plt.cm.seismic)
            # im2 = ax[1].pcolormesh(self.ref, cmap = plt.cm.seismic)
            # plt.colorbar(im1, ax = ax[0])
            # plt.colorbar(im2, ax = ax[1])                
            ax[0].set_title("Im")
            ax[1].set_title("Ref")                
            plt.show()
        else:
            self.plot_data(self.im , cmap = plt.cm.seismic)
            
    def draw_outline(self, bot, top, left, right, ax = None, color = "black"):
        ''' Draw lines to show where the cropping is happening.
        '''
        if ax == None:
            ax = plt.gca()
        for y in [top, bot]:
            ax.hlines(y, left, right, color = color)
        for x in [left, right]:
            ax.vlines(x, top, bot, color = color )              

    
    def load_arrIntoClass(self, im, ref = None, plotting = False):
        """ Load data into the class from arrays
        """
        self.im = im
        self.imShape = np.shape(self.im)
        if ref is not None:
            self.ref = ref
            self.refExists = True
            # Check the size of the inputted reference
            if self.ref.shape == self.im.shape:
                print ("Reference and Image same size")
                self.refSameSize = True
            else:
                print ("reference is different to image, assume that the fringes are centered in this image")
                self.refSameSize = False            
        else:
            self.refExists = False

        if plotting:
            self.plot_raw_input()
    
    def load_data(self, loadPath, imFile, refFile = None, blurSize = 35, plotting = False,
        plottingBlurring = False):
        """ Load data into the class from file
        """        
        
        self.im = self.loadData(loadPath + imFile)
        self.imShape = np.shape(self.im)
        if refFile is not None:
            self.refExists = True
            if type(refFile) == str:
                print ("Loading reference filePath ", refFile)
                self.ref = self.loadData(loadPath + refFile)
                print ("Assuming that the reference is an image of the same size")
                self.refSameSize = True
                
            elif type(refFile) == np.ndarray:
                print ("reference is array, ")                        
                self.ref = refFile
                
                # Check the size of the inputted reference
                if self.ref.shape == self.im.shape:
                    print ("Reference and Image same size")
                    self.refSameSize = True
                else:
                    print ("reference is different to image, assume that the fringes are centered in this image")
                    self.refSameSize = False
            else:
                assert False, "ERROR reading referecence, neither an array or filepath"
        else:
            self.refExists = False
        if plotting:
            self.plot_raw_input()        
        if imFile.split('.')[-1] in ['tiff', 'TIFF']:
            # Removing the background by subtracting a blurred version of the image
            self.blur(blurSize, plottingBlurring)                             
            
    def zeroPadImages(self, padSize = 100):
        """ Pads an image with zeros
        """        
        self.padSize = padSize
        self.im_PlasmaChannel = np.pad(self.im_PlasmaChannel, padSize, 'constant')    
        if self.refExists:
            self.ref_PlasmaChannel = np.pad(self.ref_PlasmaChannel, padSize, 'constant')    
        
    def cropCentreOfRef_DiffShape(self, channelPadding, xpad , ypad):
        print("Cropping to the centre of the refenece image, with the same size as the crop")
        bot, top , left, right = self.centerCropRegionOnImage(self.pc_xsize, self.pc_ysize, self.ref.shape )
        print (bot, top , left, right)

        sgCrop = self.createGaussianCroppingWindow(self.ref, [bot, top , left, right], self.gausPower)
        self.ref_PlasmaChannel = self.ref * sgCrop
        b_pc, t_pc, l_pc, r_pc = bot - self.paddingY, top + self.paddingY, left - self.paddingX, right + self.paddingX
        old = [b_pc, t_pc, l_pc, r_pc]
        b_pc, t_pc, l_pc, r_pc = self.check_btlr_coors_in_image([b_pc, t_pc, l_pc, r_pc], self.imShape)
        assert [b_pc, t_pc, l_pc, r_pc] == old, 'New image is outside of crop'
        self.ref_PlasmaChannel = self.ref_PlasmaChannel[b_pc:t_pc, l_pc:r_pc]

        
        
    
    def cropToPlasmaChannel(self, pc_crop_coors, plotting = False, plot_crop_result = False,
                            paddingX = 30, paddingY = 10, 
                            padSize = 100, centreCoorsOfFringeRegion = [500, 200],
                            verbose = False):
        """ Crop to the plasma channel, using the four corner coors
        The crop is done with a window function to remove the fourier efffect of
        sharp edges
            # Would like to add choice of windows
        
        """     
        bot, top, left, right = self.check_btlr_coors_in_image(pc_crop_coors, self.imShape)
        self.PlasmaStartCoors = int(left)
        # # Find the range and centres of the box marking the PC
        self.pc_xsize = (right-left)
        self.pc_ysize = (top-bot)
        
        self.paddingX = paddingX
        self.paddingY = paddingY    
        self.padSize = padSize           
        
        if plotting and self.refExists:
            # Plot the cropping box on the image
            self.draw_outline(bot, top, left, right)
            self.plot_data(self.im)

        gaus_cropping = self.createGaussianCroppingWindow(self.im, [bot, top , left, right], self.gausPower)
        if False:
            plt.title("Gaussian cropping window")
            self.plot_data(gaus_cropping)
        self.im_PlasmaChannel = self.im * gaus_cropping
        
        if self.refExists:
            if self.refSameSize:
                if verbose: print ("Ref is same size as crop region")
                self.ref_PlasmaChannel = self.ref * gaus_cropping
            else:
                if verbose: print ("Ref is a different size to the crop region")
                self.cropCentreOfRef_DiffShape(self.padSize, self.paddingX, self.paddingY)
                if plotting:
                    plt.title("Smaller reference cropped")
                    self.plot_data(self.ref_PlasmaChannel)
                    if verbose: print (self.ref_PlasmaChannel.shape)
        else:
            # Use a region where the fringes are unperturbed as the reference
            _, referenceOutline = self.crop_reference_fringes(self.padSize, self.paddingX, self.paddingY, 
                                        centreCoorsOfFringeRegion, plotting = True)
            if plotting:
                self.draw_outline(bot, top, left, right, color = 'black')
                b, t, l, t = referenceOutline
                self.draw_outline(b, t, l, t, color = 'red')
                self.plot_data(self.im)
                
        # f, ax = plt.subplots(ncols = 2)
        # plt.title("Dif between im and ref pc")
        # ax[0].imshow(self.im_PlasmaChannel)
        # ax[1].imshow(self.ref_PlasmaChannel)        
        # plt.show()
        b_pc, t_pc, l_pc, r_pc = bot - self.paddingY, top + self.paddingY, left - self.paddingX, right + self.paddingX
        b_pc, t_pc, l_pc, r_pc = self.check_btlr_coors_in_image([b_pc, t_pc, l_pc, r_pc], self.imShape)
        
        # Crop to the new region of the image
        self.im_PlasmaChannel = self.im_PlasmaChannel[b_pc:t_pc, l_pc:r_pc]
        if self.refExists:
            if self.refSameSize:
                self.ref_PlasmaChannel = self.ref_PlasmaChannel[b_pc:t_pc, l_pc:r_pc]           

        # Pad the image with zeros
        self.zeroPadImages(padSize)

        if verbose: print (self.im_PlasmaChannel.shape, self.ref_PlasmaChannel.shape)
        
        # The refence has now been created
        self.refExists = True        
        
        if plot_crop_result:
            plt.clf()
            print ("Plotting the raw images cropped with a window")
            plt.title("Raw image with Gauss window applied and padding")
            plt.pcolormesh(self.im_PlasmaChannel , cmap = plt.cm.seismic, norm = func.MidpointNormalize(midpoint=0))
            plt.axes().set_aspect('equal')
            plt.colorbar()            
            plt.show()       

        assert self.im_PlasmaChannel.shape == self.ref_PlasmaChannel.shape, 'The cropping to size has failed, {} = {}'.format(self.im_PlasmaChannel.shape, self.ref_PlasmaChannel.shape)
        
            
    def fft_of_plasma(self, plotting = True):
        ''' Do 2D fft '''
        self.F_im_PC = np.fft.fft2(self.im_PlasmaChannel)
        self.F_imShape = np.shape(self.F_im_PC)
        self.F_im_PC = self.wrapImage(self.F_im_PC)
        if self.refExists:
            self.F_ref_PC = np.fft.fft2(self.ref_PlasmaChannel)
            self.F_ref_PC = self.wrapImage(self.F_ref_PC)
                    
        if plotting:    
            f, ax = plt.subplots(nrows = 2, sharex = True)
            im = ax[0].pcolormesh( abs(self.F_im_PC), cmap = plt.cm.seismic, norm = mpl.colors.LogNorm())
            plt.colorbar(im, ax = ax[0])
            ax[0].set_title("Im PC FFT")
            ax[1].plot(abs(self.F_im_PC).sum(axis = 0))
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
        x_range = np.arange(self.F_im_PC.shape[1])
        xData = abs(self.F_im_PC).sum(axis = 0)
        
        y_range = np.arange(self.F_im_PC.shape[0])
        yData = abs(self.F_im_PC).sum(axis = 1)
        
        search = True
        searchCount = 0
        while search:
            # Locate the peaks
            xpeaks = find_peaks(xData , height = xData.max() * peakRelHeight)         
            ypeaks = find_peaks(yData , height = yData.max() * peakRelHeight)  
            # Take the first and last peak to be the peaks of interest. Tune the variable peakRelHeight
            # so this is the case
            if len(xpeaks[0]) > 2:
                xpeaks = [ [xpeaks[0][0], xpeaks[0][-1]], 
                          {'peak_heights': [xpeaks[1]['peak_heights'][0], xpeaks[1]['peak_heights'][-1]]} ]
            if len(xpeaks[0]) == 2:
                print( " the number of peaks found is 2")
                search = False
            else:
                searchCount += 1
                peakRelHeight -= 0.005
                print (peakRelHeight)
                
            if searchCount > 1e3 or peakRelHeight < 0:
                break
            
        print (xpeaks)
        assert len(xpeaks[0]) == 2, 'The peaks have not been found {}'.format(xpeaks)
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
            # plt.plot(xData)
            
            plt.plot(y_range, yData, label = "y")
            # plt.plot(yData)
            
            plt.plot(xpeaks[0], xpeaks[1]['peak_heights'], "x")
            plt.plot(ypeaks[0], ypeaks[1]['peak_heights'], "x")
            plt.legend()
            plt.show()        

        print (xpeaks)
        
        xrange = xpeaks[0][1] - xpeaks[0][0]
        yrange = F_shape[0]
        
        left = xpeaks[0][1] - xPercRange * xrange
        right = xpeaks[0][1] + xPercRange * xrange                
        bot = ypeaks[0][0] - yrange * yPercRange
        top = ypeaks[0][0] + yrange * yPercRange
        bot, top, left, right = self.check_btlr_coors_in_image([bot, top, left, right], F_shape)

        print ("Found crop coors", bot, top, left, right)
        self.crop_to_FFT_peak([bot, top, left, right], 
                        plot_crop_window_and_peak = plotting_cropping,
                        plot_fft_space = plot_fft_space
                        )
        
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
        # ## Make the phase positive
        # self.phase -= self.phase.min()
        if plotting:
            self.plot_data(self.phase, cmap = 'jet')

# =============================================================================
# From here adding to get the creation of fringes
# =============================================================================
        
    def crop_reference_fringes(self, channelPadding , xpad , ypad, safeRegion_Centre, plotting = False,
        verbose = False):
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
        referenceOutline = np.array([bot, top, left, right]) * 1.0
        if plotting and self.refExists: 
            plt.title("Reference region")
            if verbose: print (bot, top, left, right, referenceOutline)
            self.draw_outline(bot, top, left, right)
            self.plot_data(self.im)
            
        # # Find the range and centres of the box marking the PC
        xr = self.pc_xsize
        yr = self.pc_ysize
        
        # self.refData = self.im * 1.0
        sgCrop = self.crop_to_ref([bot, top, left, right]
                , xr, yr, channelPadding, xpad , ypad)
            
        if plotting and False: 
            plt.figure(figsize=(5,3.5))
            plt.title('The cropped and padded referce')
            self.plot_data(self.ref_PlasmaChannel, show = False, cmap = 'twilight',
                           norm = func.MidpointNormalize(midpoint=0))
            plt.show()        
                
        if verbose: 
            print ('Reference Created')
            print (bot, top, left, right, referenceOutline)
        return sgCrop , referenceOutline    
    
            
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
        # print ("crop_to_ref:: ", channelPadding, xpad , ypad)
        bot, top, left, right = self.check_btlr_coors_in_image(btlr, self.imShape)
        centeredReference = self.shift_centerofCrop_to_Centre(self.im, 
                                                        bot, top, left, right)   
        # Shifting the crop region to match the shifting of the center of the image    
        bot, top, left, right = self.centerCropRegionOnImage(xr, yr, self.imShape)        
        self.rawReference = centeredReference[bot:top, left:right]
   
        sgCrop = self.createGaussianCroppingWindow(centeredReference, [bot, top, left, right], self.gausPower)
        
        # croppedImage = centeredPC * (1- sgCrop) # Debugging option, invert the crop
        croppedImage = centeredReference * sgCrop
        self.ref_PlasmaChannel = self.crop_with_padding(croppedImage, 
                            bot, top, left, right, channelPadding, xpad , ypad)  
        return sgCrop

    def centerCropRegionOnImage(self, xr, yr, imShape):
        '''Shifting the crop region to match the shifting of the center of the image    
        '''
        btlr = [imShape[0] // 2 - yr //2, imShape[0] // 2 + yr //2, 
                imShape[1] // 2 - xr //2, imShape[1] // 2 + xr //2]
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




class fourier_filter_for_Phasemask(probe_image_analysis):
    def TakeImage(self, phi):
        self.phase = phi
        
    def correctPhaseShiftSign(self, plotting = True):
        
        from peakdetect import peakdetect
        l = np.average(self.phase,axis=1)
        l -= np.average(l)  # attempt to set around zero
        max_peaks, min_peaks = np.array(peakdetect(l, lookahead= len(l) // 6))
        peaks = [max_peaks, min_peaks]
        peakDict = {}
        for p, n in zip( [max_peaks, min_peaks], ["max_peaks", "min_peaks"]) :
            if len(p) > 0:
                x, y = zip(*p)
                if len(x) > 1:
                    y = np.array(y)
                    ind = func.nearposn(abs(y), max(abs(y)) )
                    x = (x[ind],)
                    y = (y[ind],)
                
                peakDict[n] = [x,y]
            else:
                peakDict[n] = False
                
        # for n in peakDict:
        #     print (n, peakDict[n])
            
        decisionMade = False
        if peakDict['max_peaks'] == False and peakDict['min_peaks'] != False:
            print ("Need to invert")
            invert = -1
            decisionMade = True
        if peakDict['min_peaks'] == False and peakDict['max_peaks'] != False :
            print ("Need to remain the same invert")            
            invert = 1
            decisionMade = True
            
        if decisionMade:
            self.phase = self.phase * invert       
            print ("The phase in the correct orientation")
            if plotting:
                plt.title("Correct Sign Phase shift")
                plt.imshow(self.phase)
                plt.colorbar()
                plt.show()
        else:
            
            if plotting:
                plt.figure(figsize=(6,4))
                plt.title("y lineout of phase, with found peaks")
                plt.plot(l)
            peaks = []
            for n in peakDict:
                peaks.append(peakDict[n])
            print (peaks)
            # print (np.shape(peaks) )
            # if len(np.shape(peaks)) == 3:
            #     s = np.shape(peaks)
            #     peaks = np.array(peaks).reshape( s[0], s[2])
            peaks = np.array(peaks)
            
            if plotting:
                plt.plot(peaks[:,0], peaks[:,1], 's')
                plt.show()
            ind = np.where(abs(peaks[:,1]) == max(abs(peaks[:,1])))[0][0]
            if peaks[ind][1] < 0:
                print ("Inverting the phase, multiple peaks found.")
                self.phase = self.phase * -1
                if plotting:
                    plt.title("Correct Sign Phase shift")
                    plt.imshow(self.phase)
                    plt.colorbar()
                    plt.show()
            
        
        
    
    def create_mask(self, x = 0.9, y = 0.9, mask_percentage = 0.5, plotPhaseShift = False, 
                    plotMasks = False):
        self.correctPhaseShiftSign(plotPhaseShift)
        self.F_image= np.fft.fft(self.phase)
        self.crop_in_fourier_space(x, y)
        self.filtered_image = np.fft.ifft(self.F_image)
        self.norm_filter = func.normaliseArr(np.real(self.filtered_image))
        self.mask_bgTrue = self.norm_filter < mask_percentage
        self.mask_pcTrue = self.norm_filter > mask_percentage

        if plotMasks:
            self.showMasks()        
        self.bg = self.mask_bgTrue * self.phase
    
    def crop_in_fourier_space(self, x = 0.5, y = 0.5):
        y = int(self.phase.shape[0]//2 * y)
        x = int(self.phase.shape[1]//2 * x)
        print (x, y)
    
        for i in range(self.phase.shape[0]//2-y, self.phase.shape[0]//2+y):
            for j in range(self.phase.shape[1]//2-x, self.phase.shape[1]//2+x):
                self.F_image[i][j] = 0
                
                
''' 
If running script by itself, the bottom version needs to be commented out,
if running as part of densityExtraction the top version needs to be commented out.
    This is due to the loading order of inherited classes. 
    I have not yet figured this out 
    CIDU
'''        
# class phi_to_rho(loadInDataToNumpy, fourier_filter_for_Phasemask):
class phi_to_rho(fourier_filter_for_Phasemask):
    def __init__(self):
        # print ("Creating phi_to_rho class")
        pass
    
    def fit_background(self, plotting = False):
        """ Fit the background.
        Fit a 2D plane and subtract
        """
        # self.bg = np.ma.array(self.phase, mask = self.mask_bgTrue)
   
        m = self.phase.shape[0]
        n = self.phase.shape[1]
        X1, X2 = np.mgrid[:m, :n]    
        
        #Regression
        X = np.hstack(   ( np.reshape(X1, (m*n, 1)) , np.reshape(X2, (m*n, 1)) ) )
        X = np.hstack(   ( np.ones((m*n, 1)) , X ))
        YY = np.reshape(self.bg, (m*n, 1))
        
        theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
        
        self.bg_plane = np.reshape(np.dot(X, theta), (m, n))
        if plotting:
            plt.title("Removing background plane")
            plt.imshow(self.bg_plane, vmin = self.phase.min(), vmax = self.phase.max(), 
                       norm = func.MidpointNormalize(midpoint = 0),
                       cmap = plt.cm.seismic)    
            plt.colorbar()
            plt.show()
        
            plt.imshow(self.phase - self.bg_plane, norm = func.MidpointNormalize(midpoint = 0),
                       cmap = plt.cm.seismic)    
            # plt.plot(self.pc_centres[:,0], self.pc_centres[:,1], "o-g", lw = 2) # DOESN'T EXIST
            plt.colorbar()
            plt.show()
        self.phase_with_bg = self.phase
        self.phase = self.phase - self.bg_plane

        if plotting:        
            f, ax = plt.subplots(ncols = 2, sharex = True, sharey = True)
            cRange = [self.bg.min(), self.bg.max()]
            ax[0].imshow(self.bg, vmin = cRange[0], vmax = cRange[1])
            ax[1].imshow(self.bg_plane, vmin = cRange[0], vmax = cRange[1])
            ax[0].set_title("Background")
            ax[1].set_title("Background plane fitted")            
            plt.show()
            
    def showMasks(self):
        f, ax = plt.subplots(ncols = 2, sharex = True, sharey = True)
        ax[0].imshow(self.mask_bgTrue)
        ax[1].imshow(self.mask_pcTrue)
        ax[0].set_title("mask, background")
        ax[1].set_title("mask, PC")        
        plt.show()
        
        
    def find_PC_angle(self, plotting = False):
        ''' Rotated the plasma channel so it is level.
        '''        
        phaseShift = self.phase * self.mask_pcTrue
        
    
        indexes = []
        for i, col in enumerate(phaseShift.T):
            if False:
                # option 1
                ind = find_peaks(col, height=phaseShift.max() * 0.8, distance= 15)
                # print (ind)
                if len(ind) == 2:
                    if len(ind[0]) == 1:
                        indexes.append( [ i, ind[0][0]] )
            else:
                # # option 2
                import peakutils
                ind = peakutils.indexes(col, thres=0.02/max(col), min_dist=30)
                if len(ind) == 1:
                    indexes.append([i, ind[0]])

        indexes = np.array( indexes )    
        print (len(indexes))
        if len(indexes) < 3:
            plt.imshow(phaseShift)
            plt.colorbar()
            plt.show()
            assert False
        popt, pcov = curve_fit(func.lin, indexes[:,0], indexes[:,1], p0 = [0, np.average(indexes[:,1])] )
        if plotting:        
            plt.plot(indexes[:,0], indexes[:,1])
            plt.imshow(phaseShift)
            plt.colorbar()
            plt.plot(indexes[:,0], func.lin(indexes[:,0], *popt), 'o-')    
            plt.show()
        
        self.centerOfPlasmaChannel = indexes
        self.angle  = np.rad2deg(np.arctan(popt[0]))
        print ("Angle from horizontal", self.angle)
        # return self.angle
            
    def plasmaChannel_Horz(self, angle = None,  plotting = False, cropToImage = False):
        if angle is not None:
            self.angle = angle
        else:
            self.find_PC_angle(plotting)
        pInit = self.phase
        self.phase = self.rotate_array(self.phase, self.angle)
        self.phase_rot_mask = self.rotate_array(np.ones_like(self.phase), self.angle)
        self.phase_rot_mask_mask = self.rotate_array(np.array(self.mask_pcTrue, dtype = float), self.angle)
        
        # Crop to the plasma channel where the rotation is not effecting the result
        lineout = self.phase_rot_mask_mask.sum(axis = 0)
        lineMask = lineout>  lineout.max() * 0.8
        for ind, p in enumerate(lineMask):
            if p:
                ind = ind + 1
                break
        if plotting and cropToImage:
            plt.plot(lineout)
            l, h = plt.ylim()
            plt.vlines(ind, l, h)
            plt.show()
        if cropToImage:
            self.phase = self.phase[:, ind:]

        if plotting:
            f, ax = plt.subplots(ncols = 2)
            ax[0].imshow(pInit)
            if hasattr(self, "centerOfPlasmaChannel"):
                ax[0].plot(self.centerOfPlasmaChannel[:,0], self.centerOfPlasmaChannel[:,1],
                        'o-r')
            ax[1].imshow(self.phase)
            ax[0].set_title("Initial")
            ax[1].set_title("Rotated to PC horz")
            plt.show()        
        return self.angle
        
    def constants(self, mPerPix, laserwavelength_m = 800e-9):
        self.lambda_l = laserwavelength_m
        self.mu_0 =  12.57e-7
        self.m_e = 9.11e-31
        self.q_e = 1.6e-19
        self.e_0 = 8.85e-12
        self.c = 3e8
        self.sizePerPixel = mPerPix
        
        self.w_0 = (2 * np.pi * self.c ) / self.lambda_l
        self.n_c = (self.w_0**2 * self.m_e * self.e_0) / self.q_e**2   

        self.r_e = 2.81794e-15           
        

    def inverse_abel_transform(self, plotting = False, method = "hansenlaw"):
        # Flip the image, so the abel transform work
        image = self.phase.T
        
        # Using the inbuilt gaussian method to find the axis
        print ("Inverse abel, outputs from PyAbel")
        self.inverse_abel = abel.Transform(image,
                                      #center = (50, 200),
                                      method =  method, 
                                      center = "gaussian",
                                      center_options = {'axes' : 1, "verbose":True},
                                      direction='inverse', verbose = True).transform.T        
            
        if plotting:
            if hasattr(self, "PlasmaStartCoors"):
                offset = self.PlasmaStartCoors
            else: 
                offset = 0
            lineout_ave = np.average(self.inverse_abel[ 10:-10, :], axis = 0)
            xAxis = np.arange(offset, offset + len(lineout_ave)) * self.sizePerPixel * 1e3            

            f, ax = plt.subplots(nrows = 2, figsize = (6,6), sharex = True)            
            im1 = ax[0].pcolormesh( 
                    xAxis,
                    np.arange(self.phase.shape[0]) * self.sizePerPixel *1e3 - self.phase.shape[0] * 0.5 * self.sizePerPixel *1e3,
                    self.inverse_abel, 
                    cmap = plt.cm.seismic, 
                    norm = func.MidpointNormalize(midpoint = 0),
                   ) 
            cax = f.add_axes([0.95, 0.25, 0.05, 0.5])
            cbar = plt.colorbar(im1, cax = cax)
            cbar.set_label("Raw Abel Transform")
            ax[0].set_title("Inverse Abel Transform")
            ax[0].set_xlabel("Pixels")
            ax[0].set_ylabel("Pixels")        
            

            ax[1].plot(xAxis, lineout_ave)
            ax[1].set_xlabel("Distance (mm)")
            plt.show()

    def convert_Inverse_Abel_to_Ne(self, plotting = True, pixelsAroundPlasmaChannel = 10, perCm3 = True,
                            verbose = False):
        
        if perCm3: self.inverse_abel *= 1e-6
        
        # Using N_e = 1/(r_e * l) * pyabelInverse, from chat with Rob Shaloo
    
        self.n_e = self.inverse_abel / (self.r_e *  self.lambda_l * self.sizePerPixel)
        
        # Take an average cropping to the center, this could be cleverer
        
        if verbose: print ("Taking average lineout in region of size {}mm around axis".format(2 * pixelsAroundPlasmaChannel * self.sizePerPixel *1e3))
        lineout_ave = np.average(self.n_e[self.phase.shape[0]//2 - pixelsAroundPlasmaChannel:
                                          self.phase.shape[0]//2 + pixelsAroundPlasmaChannel,
                                          :], 
                                 axis = 0)
        if hasattr(self, "PlasmaStartCoors"):
            offset = self.PlasmaStartCoors
        else: 
            offset = 0        
        xAxis = np.arange(offset, offset + self.phase.shape[1]) * self.sizePerPixel *1e3
        yAxis = np.arange(self.phase.shape[0]) * self.sizePerPixel *1e3 - self.phase.shape[0] * 0.5 * self.sizePerPixel *1e3            
        if plotting:
            f, ax = plt.subplots(nrows=2, sharex = True, figsize = (8,6))
            ax[0].set_title("Number Density")
            
            im1 = ax[0].pcolormesh( xAxis, yAxis, self.n_e, cmap = plt.cm.seismic,
                    norm = func.MidpointNormalize(midpoint = 0) )
            
            for height in  [-pixelsAroundPlasmaChannel,  pixelsAroundPlasmaChannel]:
                # print (self.phase.shape, height)
                ax[0].hlines(height * self.sizePerPixel *1e3, xAxis[0], xAxis[-1])
            
            cax = f.add_axes([0.95, 0.25, 0.05, 0.5])
            plt.colorbar(im1, cax = cax)
            ax[1].plot( xAxis , lineout_ave)
            ax[1].set_xlabel("Distance (mm)")
            # ax[1].set_ylim([0, None])
            if perCm3:
                # print ("per cm3")
                ax[1].set_ylabel("Plasma Density ($cm^{-3}$)")
            else:
                # print ("per m3")
                ax[1].set_ylabel("Plasma Density ($m^{-3}$)")
            plt.show()        
            
        # Returns two arrays
        # -- 2D Data: the x and y axis, the electron density, the phase shift
        # -- 1D Data: the lineout of n_e through the cell/nozzle
        return (xAxis, yAxis, self.n_e, self.phase), np.c_[np.arange(len(lineout_ave)) * self.sizePerPixel *1e3, lineout_ave]
        
        
        



class rhoExtraction(phi_to_rho, phaseShift):
    def __init__(self, gpower = 6,  **kwargs):
        phi_to_rho.__init__(self,**kwargs)
        phaseShift.__init__(self, **kwargs)
        self.gausPower = gpower
        


    def extractDensityFromImages(self, loadPath, imFile, 
                   pc_crop, FT_crop_coors, referenceFile, angle, referenceROIcentre, mPerPix, mask_percentage, paddingX, paddingY, padSize , blurSize,
                   verbose , visualise ):
        if verbose: print ("Loading in the data")
        self.load_data(loadPath, imFile, referenceFile,  blurSize = blurSize, plotting = False)

        if verbose: print ("Cropping to the plasma channel")
        # Crop to the plasma channel
        _ = self.cropToPlasmaChannel( pc_crop, plotting=visualise,
                                paddingX=paddingX, paddingY=paddingY, # Padding on the crop
                                padSize = padSize,
                                centreCoorsOfFringeRegion = referenceROIcentre
                               )    
        self.fft_of_plasma(plotting = False)        
        
        if verbose: 
            print ("Cropping to the peak in F Space")
            method =  len(FT_crop_coors) == 4
            if method:
                print ("Using crop coors in F space")
            else:
                print ("Searching for F Space crop")
        # There are two methods, one requires the peak to be properly located,
        # and the other uses the shape in F space to work out where to crop.
        if len(FT_crop_coors) == 4:    
            self.crop_to_FFT_peak(FT_crop_coors, plot_crop_window_and_peak=False,
                                 plot_fft_space = False)
        else:
            self.auto_select_FT_peak(yPercRange = 0.3, xPercRange = 0.25, plot_fft_space = False,
                                    plotting_cropping = False, plot_found_peaks = False)

        if verbose: print ("Creating Phase")
        self.createPhase_inverseFT(plotting = False)
        if verbose: print ("Unwrapping phase")
        self.unwrap_phase(plotting = visualise)       

        if verbose: print ("Creating a mask of the Plasma channel")
        self.create_mask(mask_percentage = mask_percentage, 
                    plotPhaseShift = visualise, plotMasks = visualise)    
        if verbose: print ("Fitting the phase background")        
        self.fit_background(plotting=False)

        if verbose:
            print ("Rotating the channel to Horz")
            print ("Input Angle ", angle)
        self.plasmaChannel_Horz(angle, plotting = False)
        self.constants(mPerPix)

        if verbose: print ("Abel Inversion")
        self.inverse_abel_transform(plotting=False)
        output = self.convert_Inverse_Abel_to_Ne(plotting=visualise)
        if verbose:
            print ("The outputs")
            print (type(output), len(output))
        return output

    def display_pc_cropping(self, pc_crop_coors, centreCoorsOfFringeRegion):
        b,t,l,r = pc_crop_coors
        xc, yc = centreCoorsOfFringeRegion
        pc_xsize = r - l
        pc_ysize = t - b 
        left = xc - pc_xsize // 2
        right = xc + pc_xsize // 2
        bot = yc - pc_ysize // 2
        top = yc + pc_ysize // 2        

        self.draw_outline(b,t,l,r, color = 'red') # PLASMA CHANNEL
        self.draw_outline(bot,top,left,right, color = 'blue')
        self.plot_data( self.im)


def extract_plasma_density(data_file_s, calibrationData, analysisSavePath, 
            verbose = False, visualise = False):
    if verbose:
        print ("files", data_file_s)
        print ("len Files", len(data_file_s))
        print ("len of cal data", len(calibrationData))

    refFile, cellChanging, cellOn, cellPC, cell_FCrop, nozzleOn, nozzlePC, nozzle_FCrop, angle, centreCoorsOfFringeRegion, mPerPix, mask_percentage, paddingX, paddingY, padSize, blurSize = calibrationData

    assert type(nozzleOn) == bool
    assert type(cellOn) == bool            
    assert type(cellChanging) == bool
    assert len(calibrationData) == 16
    if verbose: print ("Input data recieved correctly")

    loadPath = ''
    burstData = []
    for imFile in data_file_s:
        print ("\n\nAnalysis on {}\n".format(imFile))
        rho = rhoExtraction()
        if cellOn:
            try:
                # print ("Extracting Cell")
                cellDensity = rho.extractDensityFromImages(loadPath, imFile, cellPC, cell_FCrop, refFile, angle, 
                                centreCoorsOfFringeRegion, mPerPix, mask_percentage, paddingX, paddingY, padSize, blurSize, verbose, visualise) 
            except:
                print ("\tFAILED: Cell Extraction")
                cellDensity = ([], [])    
        else:
            cellDensity = ([], [])

        if nozzleOn:
            try:
                # print ("Extracting Nozzle")            
                nozzleDensity = rho.extractDensityFromImages(loadPath, imFile, nozzlePC, nozzle_FCrop, refFile, angle, 
                                centreCoorsOfFringeRegion, mPerPix, mask_percentage, paddingX, paddingY, padSize, blurSize, verbose, visualise) 
            except:
                print ("\tFAILED: Nozzle Extraction")
                nozzleDensity = ([], [])
        else:
            nozzleDensity = ([], [])    
        
        output = (nozzleDensity, cellDensity)
        burstData.append(output)
        
        folderpath = analysisSavePath.split("Probe_Interferometry_Analysis")[0]
        print ("Save the data: ", folderpath)
        func.check_and_makeFolder(folderpath)
        np.save(analysisSavePath.split(".")[0]  +  imFile.split(".")[0].split('Shot')[1] + '.npy', output)

    return burstData




