# PreCompNf Scripts

import numpy as np
import matplotlib.pyplot as plt

def analyseNFImage(filePath,calib):
    ''' Given a file path to a NF image
    And a calibration (relevant to the correct date), produce
    a NF (calibrarted for energy per unit area) and give the enegry
    of the beam
    '''

    # unpack the calibration data
    (meanRefCnts, pix2J) = calib

    # NOTE meanRefCnts is NOT THE MEAN COUNTS PER PIXLE
    # BUT RATHER THE MEAN IMAGE COUNTS (I.E SUM)

    img = plt.imread(filePath).astype(float)
    rows,cols = np.shape(img)

    imgCnts = np.sum(img) - meanRefCnts
    energy = imgCnts*pix2J

    avgPixVal = meanRefCnts/(rows*cols)
    img = (img - avgPixVal)*pix2J


    return (energy,img)




        
