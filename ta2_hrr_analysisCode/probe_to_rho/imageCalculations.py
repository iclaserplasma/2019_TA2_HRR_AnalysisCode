#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Jan  7 15:40:19 2020

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

import cv2

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
        print ("Creating Gaussian To Crop. Image shape", np.shape(image))
        bot, top, left, right = btlr
        cropGaussian = np.zeros_like(image, dtype = float)

        s = np.shape(image)   
        sgY = self.superGaussian(np.arange(s[1]), (left + right)*0.5, abs(right-left), power)
        sgX = self.superGaussian(np.arange(s[0]), (top + bot)*0.5, abs(top - bot), power)    

        for i, x in enumerate(sgX):
            for j, y in enumerate(sgY):
                cropGaussian[i][j] = x * y
    
        cropGaussian = np.real(cropGaussian)
        return cropGaussian        
    
    def rotate_array(self, arr, angleDeg):

        M = cv2.getRotationMatrix2D((arr.shape[0]/2, arr.shape[1]/2), 
                                    angleDeg, # Angle of rotation in Degrees
                                    1   #Scale
                                    )
        arr_Rot = cv2.warpAffine(arr ,M,(arr.shape[1] ,arr.shape[0]))
        return arr_Rot