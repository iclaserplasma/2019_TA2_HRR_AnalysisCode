# HASOAnalysis Scripts

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import os


def splitHASOFile(filename):
    # Open file to read
    f = open(filename, "r")
    # Open first file to output to
    g = open('Wavefront.xml',"w")

    # Start reading file line by line
    for i, line in enumerate(f):
    	# If the line matches the line at which the files should be separated
    	# then close Wavefront.xml and open zernike.xml
    	if re.match(r'\<phase_reconstruction_mode', line):    
            g.close()
            g = open('Pupil.xml',"w")
    	g.write(line)



def extractWavefrontInfo(dataFile):
	
	# First split the HASO file into a wavefront
	# and a pupil info file
	splitHASOFile(dataFile)

	# Read in the XML file
	tree = ET.parse('Wavefront.xml')
	root = tree.getroot()

	# pull out the individual elements
	n = int(root[0][1][0][0].text)
	m = int(root[0][1][0][1].text)
	step = float(root[0][1][1][0].text)
	xSlopesAsText = root[0][1][2][0].text
	ySlopesAsText = root[0][1][3][0].text
	intensityAsText = root[0][1][4][0].text
	pupilAsText = root[0][1][5][0].text

	X = np.linspace(-n*step/2,n*step/2-1,n)
	Y = np.linspace(-m*step/2,m*step/2-1,m)

	# Convert the text arrays to numpy arrays of floats
	xSlopes = convertAsTextToArray(xSlopesAsText)
	ySlopes = convertAsTextToArray(ySlopesAsText)
	intensity = convertAsTextToArray(intensityAsText)
	pupil = convertAsTextToArray(pupilAsText)



	# Fially, convert xSlopes and ySlopes to phase information
	phase = convertSlopesToPhase(xSlopes,ySlopes)

	# And finally finally, we must delete the temporary files
	# that were created by splitHASOFiles
	os.remove('Wavefront.xml')
	os.remove('Pupil.xml')

	return (X,Y,phase,intensity,pupil,xSlopes,ySlopes)

def convertAsTextToArray(asTextArr):
    # Takes the raw input from the haso file
    # and converts it to a numpy array
    asTextArr = asTextArr
    lines = asTextArr.split('\n')
    lines = lines[1:-1]
    rows = len(lines)
    cols = len(lines[0].split('\t')) -1 
    returnArr = np.zeros((rows,cols))
    i = 0
    for line in lines:    
        elems = line.split('\t')
        elems = elems[0:-1]
        j = 0
        for e in elems:
            returnArr[i,j] = float(e)
            j = j+1
        i = i+1
    return returnArr

def convertSlopesToPhase(xSlopes,ySlopes):
	return 0
