#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Jan 22 10:45:11 2020

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

sys.path.insert(0, '/Users/chrisunderwood/Documents/Experimental_Tools/Probe_Interfer_Analysis/')

from densityExtraction import rhoExtraction

desiredShift = 1
loadPath = ''
imFile = 'synthetic_image_{}.txt'.format(desiredShift)
refFile = 'reference_{}.txt'.format(desiredShift)
rho = rhoExtraction()
rho.load_data(loadPath, imFile, refFile,  plotting = False)
rho.plot_raw_input()

pc_crop_coors = [50, 250, 100, 400 ]

_ = rho.cropToPlasmaChannel( pc_crop_coors, plotting=True,
                        paddingX=20, paddingY=20, # Padding on the crop
                        # Extra pad  y      x
                        # padSize = [(10, ), (10, )]
                        padSize = 10,
                       )  
rho.fft_of_plasma(plotting = True)        
rho.auto_select_FT_peak(yPercRange = 0.3, xPercRange = 0.25, plot_fft_space = False,
                                plotting_cropping = False, plot_found_peaks = False)

rho.createPhase_inverseFT(plotting = True)
rho.unwrap_phase(plotting = False)      
# The final output is rho.phase

plt.title("Unwrapped extracted phase")
plt.imshow(rho.phase, cmap = 'jet', aspect = 'auto',
           vmax = round(rho.phase.max() / np.pi) * np.pi)
cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi,  4 *np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
plt.show()


maxShift = rho.phase.max() / (2 * np.pi)
print ("MaxPhase Shift = ", maxShift, '\nExpected Shift = ', desiredShift)
assert (maxShift/desiredShift) > 0.99, 'Result is not within 1% of expected'
assert (maxShift/desiredShift) < 1.01, 'Result is not within 1% of expected'

