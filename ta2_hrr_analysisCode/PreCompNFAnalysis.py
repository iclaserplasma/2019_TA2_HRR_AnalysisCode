# PreCompNf Scripts

import numpy as np
import matplotlib.pyplot as plt

def analyseNFImages(files,calib):
    ''' Given a file path or list of file paths to NF images
    And a calibration (relevant to the correct date), produce
    a NF (calibrarted for energy per unit area) and give the enegry
    of the beam
    '''

    # unpack the calibration data
    (meanRefCnts, pix2J) = calib

    # NOTE meanRefCnts is NOT THE MEAN COUNTS PER PIXLE
    # BUT RATHER THE MEAN IMAGE COUNTS (I.E SUM)

    # Now perform the analysis
    if len(dataFile[0]) > 1:
		# This means we have an array of filenames
		numFiles = len(dataFile)
		nextFile = dataFile[0]
	else:
		numFiles = 1
		nextFile = dataFile
	

	for i in range(numFiles):
        img = plt.imread(fullRefPath).astype(float)
        imgCnts = np.sum(img) - meanRefCnts
        energy = imgCnts*pix2J

        
        NFImg = img - 


        
