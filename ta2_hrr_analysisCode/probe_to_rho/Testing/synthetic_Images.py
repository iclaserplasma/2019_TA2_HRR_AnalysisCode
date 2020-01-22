#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Jan 22 10:07:27 2020

@author: chrisunderwood

    Make a fringe shift pattern with a known spacing

"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0,6.0]
import matplotlib.pyplot as plt

# Load my module of functions
import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions3 as func

def shift_row_in_image(im, shift_Func):
    assert shift_Func.shape[0] == im.shape[0]
    plt.plot(shift_Func)
    plt.show()
    
    
from scipy.signal import find_peaks


x = np.linspace(0, 30 * np.pi, 501)
shifts = x % (2* np.pi)
p = find_peaks(shifts)    
plt.plot(x % (2* np.pi), '.-')
plt.plot(p[0], 0*np.ones_like(p[0]), 'x')
plt.show()
spacing = abs(np.median( p[0][:-1] - p[0][1:] ))

desiredShift = 2 
lambda_peaks = spacing * desiredShift


sin = np.sin(x)
im = np.tile(sin, (300, 1))
plt.imshow(im)
plt.title("Reference")
plt.show()

np.savetxt("reference_{}.txt".format(desiredShift), im)

shift_Func = func.gaus( np.arange(im.shape[0]), *[lambda_peaks, im.shape[0] / 2,  10, 0])
plt.figure(figsize = (4,4))
plt.plot(shift_Func)
plt.show()

imNew = []
for i, s in enumerate(shift_Func):
    s = int(round(s))
    # print (i, s)
    
    l = im[i]
    # if s is not 0:
    #     print (s)
    l = np.concatenate( (l[s:], l[:s]) )
    imNew.append(l)
imNew = np.array(imNew)
plt.title("Phase Shift")
plt.imshow(imNew)
    
plt.show()

np.savetxt("synthetic_image_{}.txt".format(desiredShift), imNew)    
    
