#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Jan 14 11:46:28 2020

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# # Load my module of functions
# import sys
# sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions3 as func
from imageCalculations import probe_image_analysis
from loadData import loadInDataToNumpy
import abel


# =============================================================================
# Flattern background phase
# =============================================================================

class fourier_filter_for_Phasemask(probe_image_analysis):
    def TakeImage(self, phi):
        self.phase = phi
    
    def create_mask(self, x = 0.9, y = 0.9, mask_percentage = 0.5):
        
        self.F_image= np.fft.fft(self.phase)
        self.crop_in_fourier_space(x, y)
        self.filtered_image = np.fft.ifft(self.F_image)
        self.norm_filter = func.normaliseArr(np.real(self.filtered_image))
        self.mask = self.norm_filter < mask_percentage
        self.antimask = self.norm_filter > mask_percentage
        
        self.bg = self.mask * self.phase
    
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
        print ("Creating phi_to_rho class")
    
    def fit_background(self, plotting = False):
        """ Fit the background.
        Fit a 2D plane and subtract
        """
        # self.bg = np.ma.array(self.phase, mask = self.mask)
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
            plt.plot(self.pc_centres[:,0], self.pc_centres[:,1], "o-g", lw = 2)
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
            
    def find_PC_angle(self, plotting = False):
        ''' Rotated the plasma channel so it is level.
        '''        
        phaseShift = self.phase * self.antimask
    
        indexes = []
        for i, col in enumerate(phaseShift.T):
            ind = find_peaks(col, height=phaseShift.max() * 0.8, distance= 15)
            if len(ind) == 2:
                if len(ind[0]) == 1:
                    indexes.append( [ i, ind[0][0]] )
        indexes = np.array( indexes )    
        popt, pcov = curve_fit(func.lin, indexes[:,0], indexes[:,1], p0 = [0, np.average(indexes[:,1])] )
        if plotting:        
            plt.plot(indexes[:,0], indexes[:,1])
            plt.imshow(phaseShift)
            plt.colorbar()
            plt.plot(indexes[:,0], func.lin(indexes[:,0], *popt))    
            plt.show()
        
        self.angle  = np.rad2deg(np.arctan(popt[0]))
        print ("Angle from horozontal", self.angle)
        # return self.angle
            
    def plasmaChannel_Horz(self, plotting = False):
        self.find_PC_angle()
        pInit = self.phase
        self.phase = self.rotate_array(self.phase, self.angle)
        self.phase_rot_mask = self.rotate_array(np.ones_like(self.phase), self.angle)
        self.phase_rot_mask_mask = self.rotate_array(np.array(self.antimask, dtype = float), self.angle)
        
        # Crop to the plasma channel where the rotation is not effecting the result
        lineout = self.phase_rot_mask_mask.sum(axis = 0)
        lineMask = lineout>  lineout.max() * 0.8
        for ind, p in enumerate(lineMask):
            if p:
                ind = ind + 1
                break
        if plotting and False:
            plt.plot(lineout)
            l, h = plt.ylim()
            plt.vlines(ind, l, h)
            plt.show()
        self.phase = self.phase[:, ind:]

        if plotting:
            f, ax = plt.subplots(ncols = 2)
            ax[0].imshow(pInit)
            ax[1].imshow(self.phase)
            ax[0].set_title("Initial")
            ax[1].set_title("Rotated to PC horz")
            plt.show()        
        
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
        self.inverse_abel = abel.Transform(image,
                                      #center = (50, 200),
                                      method =  method, 
                                      center = "gaussian",
                                      center_options = {'axes' : 1, "verbose":True},
                                      direction='inverse', verbose = True).transform.T        
            
        if plotting:
            f, ax = plt.subplots(nrows = 2, figsize = (6,6), sharex = True)            
            im1 = ax[0].pcolormesh( 
                    np.arange(self.phase.shape[1]) * self.sizePerPixel *1e3,
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
            
            lineout_ave = np.average(self.inverse_abel[ 10:-10, :], axis = 0)
            
            ax[1].plot(np.arange(len(lineout_ave)) * self.sizePerPixel * 1e3, lineout_ave)
            ax[1].set_xlabel("Distance (mm)")
            plt.show()

    def convert_Inverse_Abel_to_Ne(self, plotting = True, pixelsAroundPlasmaChannel = 10, perCm3 = True):
        
        if perCm3: self.inverse_abel *= 1e-6
        
        # Using N_e = 1/(r_e * l) * pyabelInverse, from chat with Rob Shaloo
    
        self.n_e = self.inverse_abel / (self.r_e *  self.lambda_l * self.sizePerPixel)
        
        # Take an average cropping to the center, this could be cleverer
        
        print ("Taking average lineout in region of size {}mm around axis".format(2 * pixelsAroundPlasmaChannel * self.sizePerPixel *1e3))
        lineout_ave = np.average(self.n_e[self.phase.shape[0]//2 - pixelsAroundPlasmaChannel:
                                          self.phase.shape[0]//2 + pixelsAroundPlasmaChannel,
                                          :], 
                                 axis = 0)
        if plotting:
            f, ax = plt.subplots(nrows=2, sharex = True, figsize = (8,6))
            ax[0].set_title("Number Density")
            im1 = ax[0].pcolormesh( 
                    np.arange(self.phase.shape[1]) * self.sizePerPixel *1e3,
                    np.arange(self.phase.shape[0]) * self.sizePerPixel *1e3 - self.phase.shape[0] * 0.5 * self.sizePerPixel *1e3,
                    self.n_e, cmap = plt.cm.seismic,
                    norm = func.MidpointNormalize(midpoint = 0) )
            
            for height in  [-pixelsAroundPlasmaChannel,  pixelsAroundPlasmaChannel]:
                # print (self.phase.shape, height)
                ax[0].hlines(height * self.sizePerPixel *1e3, 0, self.phase.shape[1]* self.sizePerPixel *1e3)
            
            cax = f.add_axes([0.95, 0.25, 0.05, 0.5])
            plt.colorbar(im1, cax = cax)
            ax[1].plot(np.arange(len(lineout_ave)) * self.sizePerPixel *1e3, lineout_ave)
            ax[1].set_xlabel("Distance (mm)")
            # ax[1].set_ylim([0, None])
            if perCm3:
                # print ("per cm3")
                ax[1].set_ylabel("Plasma Density ($cm^{-3}$)")
            else:
                # print ("per m3")
                ax[1].set_ylabel("Plasma Density ($m^{-3}$)")
            plt.show()        
            
        return self.n_e, np.c_[np.arange(len(lineout_ave)) * self.sizePerPixel *1e3, lineout_ave]
        
        
        

if __name__ == "__main__":
    loadPath = "/Users/chrisunderwood/Documents/Experimental_Tools/Probe_Interfer_Analysis/Trial_Images/"
    fileName = '20190913r016b6s15_clean_phi.txt'
    # phi = np.loadtxt(loadPath + fileName)    
        

    phiCoor = phi_to_rho()
    phase = phiCoor.loadData(loadPath + fileName)
    phiCoor.phase = phase
    phiCoor.create_mask(mask_percentage = 0.35)    
    phiCoor.fit_background()

    phiCoor.plasmaChannel_Horz()
    phiCoor.constants(mPerPix = 4.98107e-06)
    phiCoor.inverse_abel_transform(plotting=True)
    phiCoor.convert_Inverse_Abel_to_Ne()
