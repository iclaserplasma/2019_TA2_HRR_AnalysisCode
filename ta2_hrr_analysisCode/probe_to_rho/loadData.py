#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Wed Jun 19 09:32:01 2019

@author: chrisunderwood

    Load data class
    turns the data into a numpy array
"""
import numpy as np
import matplotlib.pyplot as plt


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
            print ("Loading .txt")           
            data = np.loadtxt(filePath)
        elif filePath.endswith(".tiff") or filePath.endswith(".TIFF"):
            print ("Loading .tiff/TIFF")
            from skimage import io
            data = io.imread(filePath)
            data = data.astype(float)
            # print (type(data)) 
        elif filePath.endswith(".tif"):
            print ("Loading .tif")            
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

