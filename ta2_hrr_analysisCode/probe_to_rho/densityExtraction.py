#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Jan 15 15:24:11 2020

@author: chrisunderwood
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

# Load my module of functions
import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions3 as func

from phase_correction import phi_to_rho
from calcDeltaPhi import phaseShift

# =============================================================================
# Density extraction
# =============================================================================

class rhoExtraction(phi_to_rho, phaseShift):
    def __init__(self, **kwargs):
        # phi_to_rho.__init__(**kwargs)
        pass

        
if __name__ == "__main__":
    
    loadPath = "/Users/chrisunderwood/Documents/Experimental_Tools/Probe_Interfer_Analysis/Trial_Images/"
    refFile = None
    imageNos = 2
    
    if imageNos == 1:
        # imFile = '20190913r016b6s15_clean.txt'
        imFile = '20190913r016b6s15.TIFF'        
        pc_crop_coors = [300, 500, 1100, 1210] # bot, top, left, right
        FT_crop_coors = [49, 205, 87, 92]        
    elif imageNos == 2:
        imFile = '20190904r010s5.TIFF'
        pc_crop_coors = [500, 650, 930, 1170]     #"20190904r010s5_clean.txt"
        FT_crop_coors = [ 36, 162, 153, 159]        
    
    
    rho = rhoExtraction()
    rho.load_data(loadPath, imFile, refFile,  plotting = False)
    if imFile.endswith('TIFF'):
        # Removing the background by subtracting a blurred version of the image
        rho.blur(35,plotting = False)    
        
    # Crop to the plasma channel
    _ = rho.cropToPlasmaChannel( pc_crop_coors, plotting=True,
                            paddingX=20, paddingY=20, # Padding on the crop
                            # Extra pad  y      x
                            # padSize = [(10, ), (10, )]
                            padSize = 10,
                            centreCoorsOfFringeRegion = [500, 200]
                           )    
    rho.fft_of_plasma(plotting = False)        
    
    # There are two methods, one requires the peak to be properly located,
    # and the other uses the shape in F space to work out where to crop.
    if False:    
        rho.crop_to_FFT_peak(FT_crop_coors, plot_crop_window_and_peak=False,
                             plot_fft_space = False)
    else:
        rho.auto_select_FT_peak(yPercRange = 0.3, xPercRange = 0.25, plot_fft_space = False,
                                plotting_cropping = False, plot_found_peaks = False)
    
    rho.createPhase_inverseFT(plotting = False)
    rho.unwrap_phase(plotting = False)       

    # rho.phase = phase
    rho.create_mask(mask_percentage = 0.35)    
    rho.fit_background(plotting=False)

    rho.plasmaChannel_Horz(plotting = True)
    rho.constants(mPerPix = 4.98107e-06)
    rho.inverse_abel_transform(plotting=True)
    rho.convert_Inverse_Abel_to_Ne()
    


