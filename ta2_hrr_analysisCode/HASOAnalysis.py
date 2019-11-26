# HASOAnalysis Scripts

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import os
from scipy.integrate import trapz



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

	# And finally, finally, get Zernike Coefficients
	zernikeCoeffs = getZernikeCoefficients(X,Y,pupil,phase)

	# And finally finally finally, we must delete the temporary files
	# that were created by splitHASOFiles
	os.remove('Wavefront.xml')
	os.remove('Pupil.xml')

	return (X,Y,phase,intensity,pupil,xSlopes,ySlopes,zernikeCoeffs)

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


def getPupilCoords(X,Y,pupil):
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    x,y = np.meshgrid(X,Y)
    (m,n) = x.shape
    cgx = np.sum(x*pupil)/np.sum(pupil)
    cgy = np.sum(y*pupil)/np.sum(pupil)
    r = np.sqrt(np.sum(pupil)*dx*dy/np.pi)
    return (cgx,cgy,r)


def zernike(X,Y,pupilCoords,j):
    # Provides the required Zernike Polynomial
    # Defined on the pupil.
    # The zernike poynomial is indexed by j, with the following:
    # j = {0:Piston, 1:YTilt, 2:XTilt, 3:ObliqueAstig, 4:Focus, 5:VerticalAstig,...
    # 6:VerticalTrefoil, 7:VerticalComa, 8:HorizontalComa, 9:ObliqueTrefoil}
    # Currently, we're only using the first 10
    
    # X and Y are 1D arrays of the x and y direction
    # Pupil coords is the result of getPupilCoords
    # j is the index of the Zernike
    
    (cgx,cgy,r) = pupilCoords
    x,y = np.meshgrid(X,Y)
    R = np.sqrt((x-cgx)**2 + (y-cgy)**2)
    theta = np.arctan2(y-cgy, x-cgx)
    
    pupil = np.ones((len(Y),len(X)))
    pupil[np.where(R > r) ] = np.nan

    if j == 0:
        # Piston
        zMode = np.ones((len(Y),len(X)))
        
    elif j ==1:
        # Y Tilt
        zMode = 2*R*np.sin(theta)
        
    elif j ==2:
        # X Tilt
        zMode = 2*R*np.cos(theta)
        
    elif j ==3:
        # Oblique Astigmatism
        zMode = np.sqrt(6) * (R**2) * np.sin(2*theta)
        
    elif j ==4:
        # Focus
        zMode = np.sqrt(3) * (2*(R**2) -1)
        
    elif j ==5:
        # Vertical Astigmastism
        zMode = np.sqrt(6) * (R**2) * np.cos(2*theta)
        
    elif j ==6:
        # Vertical Trefoil
        zMode = np.sqrt(8) * (R**3) * np.sin(3*theta)
        
    elif j ==7:
        # Vertical Coma
        zMode = np.sqrt(8) * (3*(R**3) - 2*R) * np.sin(theta)
        
    elif j ==8:
        # Horizontal Coma 
        zMode = np.sqrt(8) * (3*(R**3) - 2*R) * np.cos(theta)    
        
    elif j ==9:
        # Oblique Trefoil
        zMode = np.sqrt(8) * (R**3) * np.cos(3*theta)
    else:
        print('j out of bounds')
        
        
    return zMode*pupil

def getZernikeCoefficients(X,Y,pupil,phase):
    # Find the zernike coefficients for the phase profile
    
    pupilCoords = getPupilCoords(X,Y,pupil)
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    
    zList = np.zeros(10,)
    for i in range(10):
        Zj = zernike(X,Y,pupilCoords,i)
        Zj[np.isnan(Zj)]=0
        phase[np.isnan(phase)]=0
        integral = trapz(trapz(Zj*phase,dx=dx,axis=1),dx=dy,axis=0)
        zList[i] = integral/np.pi
    return zList