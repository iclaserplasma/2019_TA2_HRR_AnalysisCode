# HASOAnalysis Scripts

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import re
import os
from scipy.integrate import trapz
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import lsqr,lsmr
from numpy import matlib as mb
from scipy.interpolate import RegularGridInterpolator,griddata
from math import factorial


# 202-02-06 Edit: 	corrected error in creation of X and Y arrays in extractWavefrontInfo
#				  	It would have incorrectly calculated dx by 2.5 %


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



def extractWavefrontInfo(dataFile,verbose=False):
	
	# First split the HASO file into a wavefront
	# and a pupil info file
	if len(dataFile[0]) > 1:
		# This means we have an array of filenames
		# in this case pull in each file and average it
		numFiles = len(dataFile)
		nextFile = dataFile[0]
	else:
		numFiles = 1
		nextFile = dataFile
	
	for i in range(numFiles):
		if i > 0:
			nextFile = dataFile[i]
		
		splitHASOFile(nextFile)
		if verbose:
			print('Reading In File' + nextFile)
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

		X = np.linspace(-n*step/2,n*step/2-step,n)
		Y = np.linspace(-m*step/2,m*step/2-step,m)

		if i == 0:
			# Convert the text arrays to numpy arrays of floats
			xSlopes = convertAsTextToArray(xSlopesAsText)
			ySlopes = convertAsTextToArray(ySlopesAsText)
			intensity = convertAsTextToArray(intensityAsText)
			pupil = convertAsTextToArray(pupilAsText)
		else:
			xSlopes = xSlopes + convertAsTextToArray(xSlopesAsText)
			ySlopes = ySlopes + convertAsTextToArray(ySlopesAsText)
			intensity = intensity + convertAsTextToArray(intensityAsText)
			pupil = pupil + convertAsTextToArray(pupilAsText)
		
		
		# And finally finally finally, we must delete the temporary files
		# that were created by splitHASOFiles
		os.remove('Wavefront.xml')
		try:
			os.remove('Pupil.xml')
		except:
			print('No pupil.xml File Found to Delete')

	xSlopes = xSlopes/numFiles
	ySlopes = ySlopes/numFiles
	intensity = intensity/numFiles
	pupil = pupil/numFiles
	pupil[np.where(pupil > 1/(numFiles+1))] = 1


	# Fially, convert xSlopes and ySlopes to phase information
	(X,Y,phase,intensity,pupil) = convertSlopesToPhase(X,Y,xSlopes,ySlopes,intensity,pupil)

	# And finally, finally, get Zernike Coefficients
	zernikeCoeffs,pupilCoords = getZernikeCoefficients(X,Y,pupil,phase)

	# Sub finally, lets remove the piston term just because, it does nothing really.
	phase = phase - zernikeCoeffs[0]*zernike(X,Y,pupilCoords,0)
	zernikeCoeffs[0] = 0

	
	return (X,Y,phase,intensity,pupil,pupilCoords,zernikeCoeffs)

def extractCalibratedWavefrontInfo(dataFile,zernikeOffsets,verbose=False):
	# Given a data File and a calibration File, this function spits out a list of
	# Zernike polynomials which can be used to correctly identify the wavefront
	# at the interaction point from measurements on the leakage diagnostic table.

	# The zernikeOffsets are simply the difference in zernike polynomial between the chamber
	# and the leakage table.

	# First extract the wavefront
	(X,Y,phase,intensity,pupil,pupilCoords,zernikeCoeffs) = extractWavefrontInfo(dataFile,verbose=verbose)

	zernikes = zernikeCoeffs + zernikeOffsets
	return zernikes

def createCalibrationFile(inChamberHASDir,leakageHASDir,saveDir):
	# Given a set of haso measurements from inside the chamber and from the leakage table
	# This script will calculate the zernike offset between the two.
	# Zernikes are used because they provide a reasonable way to output the data
	# independant of fluctuations in beam size and in a format readily available for
	# input into other codes. 


	# Lets pull in the in-chamber files and get wavefront
	files_C = os.listdir(inChamberHASDir)
	for i in range(len(files_C)):
		files_C[i] = os.path.join(inChamberHASDir,files_C[i])
	(X_C,Y_C,phase_C,intensity_C,pupil_C,pupilCoords_C,zernikeCoeffs_C) = extractWavefrontInfo(files_C)

	# Now the same for the files taken on the leakage diagnostic
	files_L = os.listdir(leakageHASDir)
	for i in range(len(files_L)):
		files_L[i] = os.path.join(leakageHASDir,files_L[i])
	(X_L,Y_L,phase_L,intensity_L,pupil_L,pupilCoords_L,zernikeCoeffs_L) = extractWavefrontInfo(files_L)

	zernikeOffsets = zernikeCoeffs_C - zernikeCoeffs_L
	savePath = os.path.join(saveDir,'HASOCalibration')
	np.save(savePath,zernikeOffsets)

	calibDetails = [inChamberHASDir,leakageHASDir]
	savePath = os.path.join(saveDir,'HASOCalibrationPathsUSed')
	np.save(savePath,calibDetails)

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



def getPupilCoords(X,Y,pupil):
    dx = X[1]-X[0]
    dy = Y[1]-Y[0]
    x,y = np.meshgrid(X,Y)
    (m,n) = x.shape
    cgx = np.sum(x*pupil)/np.sum(pupil)
    cgy = np.sum(y*pupil)/np.sum(pupil)
    r = np.sqrt(np.sum(pupil)*dx*dy/np.pi)
    return (cgx,cgy,r)



def intgrad2(fx,fy,X,Y,f00):
	''' Based on Matlab code by John D'Errico
	Painully converted to Python by Rob Shalloo
	 intgrad2: generates a 2-d surface, integrating gradient information.
	 arguments: (input)
	  fx,fy - (ny by nx) arrays, as gradient would have produced. fx and
	          fy must both be the same size. Note that x is assumed to
	          be the column dimension of f, in the meshgrid convention.
	          nx and ny must both be at least 2.
	          fx and fy will be assumed to contain consistent gradient
	          information. If they are inconsistent, then the generated
	          gradient will be solved for in a least squares sense.
	          Central differences will be used where possible.
	     X - x array of image
	     Y - y array of image
	 arguments: (output)
	   fhat - (nx by ny) array containing the integrated gradient
	'''

	(ny,nx) = fx.shape

	# Convert X and Y arrays to difference arrays
	dx = np.diff(X)
	dy = np.diff(Y)

	
	# build gradient design matrix, sparsely. Use a central difference
	# in the body of the array, and forward/backward differences along
	# the edges.

	# A will be the final design matrix. it will be sparse.
	# The unrolling of F will be with row index running most rapidly.
	rhs = np.zeros((2*nx*ny,))
	Af = np.zeros((2*nx*ny,6))
	L = 0

	indx = 0
	indy = np.linspace(0,ny-1,ny).astype(int)
	ind = indy + (indx)*ny
	rind = np.transpose(mb.repmat(np.linspace(L,L+ny-1,ny).astype(int),2,1))
	cind = np.transpose(np.asarray([ind,ind+ny]))
	dfdx = mb.repmat(np.linspace(-1,1,2)/dx[0],ny,1)
	Af[np.linspace(L,L+ny-1,ny).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	rhs[np.linspace(L,L+ny-1,ny).astype(int)] = fx[:,1]
	L = L + ny

	# interior partials in x, central difference
	indx,indy = np.meshgrid(np.linspace(1,nx-2,nx-2).astype(int),np.linspace(0,ny-1,ny).astype(int))
	indx = np.reshape(np.transpose(indx),((nx-2)*ny,))
	indy = np.reshape(np.transpose(indy),((nx-2)*ny,))
	ind = indy + (indx)*ny
	m = ny*(nx-2)
	rind = np.transpose(mb.repmat(np.linspace(L,L+m-1,m).astype(int),2,1))
	cind = np.transpose(np.asarray([ind-ny,ind+ny]))
	dfdx = np.transpose(np.asarray([- 1/(dx[indx-1] + dx[indx]), 1/(dx[indx-1] + dx[indx])]))
	Af[np.linspace(L,L+m-1,m).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	fxtmp = np.reshape(np.transpose(fx),nx*ny,)
	rhs[np.linspace(L,L+m-1,m).astype(int)] = fxtmp[ind]
	L = L+m

	# do the trailing edge in x, backward difference
	indx = nx-1
	indy = np.linspace(0,ny-1,ny).astype(int)
	ind = indy + (indx)*ny
	rind = np.transpose(mb.repmat(np.linspace(L,L+ny-1,ny).astype(int),2,1))
	cind = np.transpose(np.asarray([ind-ny,ind]))
	dfdx = mb.repmat(np.linspace(-1,1,2)/dx[-1],ny,1)
	Af[np.linspace(L,L+ny-1,ny).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	rhs[np.linspace(L,L+ny-1,ny).astype(int)] = fx[:,-1]
	L = L+ny


	# do the leading edge in y, forward difference
	indx = np.linspace(0,nx-1,nx).astype(int)
	indy = 0
	ind = indy + (indx)*ny
	rind = np.transpose(mb.repmat(np.linspace(L,L+nx-1,nx).astype(int),2,1))
	rind.shape
	cind = np.transpose(np.asarray([ind,ind+1]))
	cind.shape
	dfdx = mb.repmat(np.linspace(-1,1,2)/dy[0],nx,1)
	Af[np.linspace(L,L+nx-1,nx).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	#rhs[np.linspace(L,L+nx-1,nx).astype(int)] = np.transpose((fy[1,:],))
	rhs[np.linspace(L,L+nx-1,nx).astype(int)] = fy[1,:]
	L = L + nx

	# interior partials in y, central difference
	indx,indy = np.meshgrid(np.linspace(0,nx-1,nx).astype(int),np.linspace(1,ny-2,ny-2).astype(int))
	indx = np.reshape(np.transpose(indx),((ny-2)*nx,))
	indy = np.reshape(np.transpose(indy),((ny-2)*nx,))
	ind = indy + (indx)*ny
	m = nx*(ny-2)
	rind = np.transpose(mb.repmat(np.linspace(L,L+m-1,m).astype(int),2,1))
	cind = np.transpose(np.asarray([ind-1,ind+1]))
	dfdx = np.transpose(np.asarray([- 1/(dy[indy-1] + dy[indy]), 1/(dy[indy-1] + dy[indy])]))
	Af[np.linspace(L,L+m-1,m).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	fytmp = np.reshape(np.transpose(fy),nx*ny,)
	rhs[np.linspace(L,L+m-1,m).astype(int)] = fytmp[ind]
	L = L+m

	# do the trailing edge in y, backward difference
	indx = np.linspace(0,nx-1,nx).astype(int)
	indy = ny-1
	ind = indy + (indx)*ny 
	rind = np.transpose(mb.repmat(np.linspace(L,L+nx-1,nx).astype(int),2,1))
	rind.shape
	cind = np.transpose(np.asarray([ind-1,ind]))
	cind.shape
	dfdx = mb.repmat(np.linspace(-1,1,2)/dy[-1],nx,1)
	Af[np.linspace(L,L+nx-1,nx).astype(int),:] = np.concatenate((rind,cind,dfdx),axis=1)
	rhs[np.linspace(L,L+nx-1,nx).astype(int)] = fy[-1,:]

	A1 = csr_matrix((Af[:,4] , (Af[:,0].astype(int) ,Af[:,2].astype(int))),shape=(2*nx*ny,nx*ny))
	A2 = csr_matrix((Af[:,5] , (Af[:,1].astype(int) ,(Af[:,3]).astype(int))),shape=(2*nx*ny,nx*ny))
	A = A1+A2

	Tmp = np.zeros(len(rhs),)
	for i in range(len(rhs)):
	    Tmp[i] = A[i,0]
	    
	rhs = rhs - Tmp*f00

	X = lsqr(A, rhs)
	X = np.transpose(np.reshape(X[0],(nx,ny)))

	return(X)



def convertSlopesToPhase(X,Y,xSlopes,ySlopes,intensity,pupil):
	# Use intgrad2 method to convert from slopes to phase
	# Prior to using this we will do the following:
	# 	regrid the slopes data to add a larger box and to more finely sample it
	# 	Set the perimeter of the box to the average value of the slopes
	# 	Interpolate the slopes data out to the edge of the box to remove Nans
	# 	Use intgrad2 to find the phase
	# 	
	# 	Finally we will look for how well we reconstructed the phase
	# NOTE: THE FINAL PHASE THAT IS RETURNED HAS A SIZE DIFFERENT TO THAT OF THE OTHER
	# # ARRAYS. THIS WILL NEED TO BE TAKEN CARE OF AFTER THE FUNCTION. 

	LAMBDA = 0.8 # wavelength of light in microns

	(m,n) = np.shape(xSlopes)
	Nx  = 8*n
	Ny = 8*m

	# Save this for regridding later
	oldXRes = (X[-1]-X[0])/(n-1)
	oldYRes = (Y[-1]-Y[0])/(m-1)

	xRes = (X[-1]-X[0])/(n-1)/4
	yRes = (Y[-1]-Y[0])/(m-1)/4
	Xtmp = X
	Ytmp = Y
	(X,Y, xSlopes) = reGridData(Xtmp,Ytmp,xSlopes,xRes,yRes,Nx,Ny,verbose=False)
	(X,Y, ySlopes) = reGridData(Xtmp,Ytmp,ySlopes,xRes,yRes,Nx,Ny,verbose=False)

	# ALSO REGRID PUPIL AND INTENSITY ONTO SAME GRID 
	(X,Y, pupil) = reGridData(Xtmp,Ytmp,pupil,xRes,yRes,Nx,Ny,verbose=False)
	(X,Y, intensity) = reGridData(Xtmp,Ytmp,intensity,xRes,yRes,Nx,Ny,verbose=False)

	intensity[np.isnan(intensity)]=0
	pupil[np.isnan(pupil)]=0
	pupil[np.where(pupil >=0.1)] = 1
	pupil[np.where(pupil <0.1)] = 0

	# get the pupil coordinates for cropping the image afterwards
	(cgx,cgy,r) = getPupilCoords(X,Y,pupil)
	x,y = np.meshgrid(X,Y)
	RR = np.sqrt((x-cgx)**2 +(y-cgy)**2)
	newPupil = np.ones((len(Y),len(X)))
	newPupil[np.where(RR > r)] = np.nan


	# Stick a border around the edge of the image, we will interpolate to this!
	xM = np.nanmean(xSlopes)
	xSlopes[:,0] = xM
	xSlopes[:,-1] = xM
	xSlopes[0,:] = xM
	xSlopes[-1,:] = xM

	yM = np.nanmean(ySlopes)
	ySlopes[:,0] = yM
	ySlopes[:,-1] = yM
	ySlopes[0,:] = yM
	ySlopes[-1,:] = yM

	# Add a mask to the array and then interpolate the data using griddata
	xSlopes = np.ma.masked_invalid(xSlopes)
	x1 = x[~xSlopes.mask]
	y1 = y[~xSlopes.mask]
	newarr = xSlopes[~xSlopes.mask]

	xSlopesI = griddata((x1, y1), newarr.ravel(),
	                          (x, y),
	                             method='cubic')

	ySlopes = np.ma.masked_invalid(ySlopes)
	x1 = x[~ySlopes.mask]
	y1 = y[~ySlopes.mask]
	newarr = ySlopes[~ySlopes.mask]

	ySlopesI = griddata((x1, y1), newarr.ravel(),
	                          (x, y),
	                             method='cubic')


	XnanVals = np.isnan(xSlopes)
	YnanVals = np.isnan(ySlopes)


	# WE FINISH HERE BY INTEGRATION AND SOME UNIT CONVERSIONS 
	# First we integrate the phase slopes and then divide by 1000.
	# THE factor of 1000 is becaue the phase slopes are in units of um/mm
	# When we integrated we integrated using units of microns rather than mm
	# thus the factor of 1000 converts us to um.
	phase = intgrad2(xSlopesI,ySlopesI,X,Y,0)
	phase = phase*newPupil/1000

	# Now convert from microns to radians
	phase = phase*2*np.pi/LAMBDA

	# Remove the mean level( almost piston )
	phase = phase - np.nanmean(phase)

	return (X,Y,phase,intensity,pupil)


def reGridData(x,y,dat,resX,resY,Nx,Ny,verbose=False):
    ''' Re Gridding data dat from a grid defined by x and y
    on to a new grid Nx by Ny which has grid spacing resX and resY
    The input beam should be centered on (0,0) which should be the middle of the grid'''
    
    # Create the function for interpolating the data
    interFn = RegularGridInterpolator((y,x), dat,bounds_error=False,fill_value=np.nan)
    
    # Target Grid
    xNew = np.linspace(-resX*Nx/2,resX*Nx/2 - resX ,Nx)
    yNew = np.linspace(-resY*Ny/2,resY*Ny/2 - resY ,Ny)
    regriddedData = np.zeros((Ny,Nx),dtype=float)

    for i in range(Nx):
    	for j in range(Ny):
    		regriddedData[j][i] = interFn([yNew[j],xNew[i]])

    return (xNew,yNew, regriddedData)


def zernike(X,Y,pupilCoords,j):
	'''	
	Calculate the Zernike Polynomials to arbitrary order.
	Makes use of consructor formula on https://en.wikipedia.org/wiki/Zernike_polynomials
	So far the first 100 have been tested for ortogonality and normalization
	
	X and Y are 1D arrays of the x and y direction
	Pupil coords is the result of getPupilCoords
	j is the index of the Zernike
	'''

	# Setup
	(cgx,cgy,r) = pupilCoords
	x,y = np.meshgrid(X,Y)
	rho = np.sqrt((x-cgx)**2 + (y-cgy)**2)/r
	theta = np.arctan2(y-cgy, x-cgx)

	pupil = np.ones((len(Y),len(X)))
	pupil[np.where(rho > 1) ] = np.nan

	n,m = getZernikeNM(j)

	# next get the radial part
	R = RmnGenerator(abs(m),n,rho)

	# Now multiply by the azimuthal part
	if m < 0:
		Z = R*np.sin(-m*theta)
	else:
		Z = R*np.cos(m*theta)

	# Normalization
	if n == 0:
		scaling = 1
	else:
		if m == 0:
			scaling = np.sqrt((n+1))
		else:
			scaling = np.sqrt(2*(n+1))
	Z = Z*scaling


	return Z*pupil

def RmnGenerator(m,n,rho):
	'''Generate the Radial part of the Zernike Polynomials
	m,n integers non negative
	rho a 2D array of values (positive)
	'''
	if n == 0:
		try:
			r, = rho.shape
			Rmn = np.ones(r,)
		except:
			r,c = rho.shape
			Rmn = np.ones((r,c))  
	elif (n-m)%2 == 0:
		# Even, Rmn is not 0
		k = np.linspace(0,int((n-m)/2),int((n-m)/2) + 1).astype(int)
		try:
			r, = rho.shape
			Rmn = np.zeros(r,)
		except:
			r,c = rho.shape
			Rmn = np.zeros((r,c))  
		for i in k:
			Rmn = Rmn +  ((-1)**i * factorial(n-i)) / ( factorial(i)*factorial((n+m)/2-i)*factorial((n-m)/2-i) )  * rho**(n-2*i) 
			
	else:
		try:
			r, = rho.shape
			Rmn = np.zeros(r,)
		except:
			r,c = rho.shape
			Rmn = np.zeros((r,c))  
	
	Rmn[np.where(rho > 1)] = 0
	return Rmn


def getZernikeNM(zernikeIndx):
	# numZernikes = 1/2*n*(n+1)+n+1
	# First get the zernike order n
	vals = np.roots((1/2,3/2,-zernikeIndx))
	vals = np.amax(vals)
	n = int(np.ceil(np.around(vals,decimals=5)))

	# Now find the possible m values for this order
	mVals = np.linspace(-n,n,n+1)

	# now single out the correct m value
	mInd = zernikeIndx - (1/2*(n-1)*(n)+n)  # here we're subtracting the zernikes from the previous orders and then subtracting one to create an index
	m = int(mVals[int(mInd)])

	return n,m

def zernikeOld(X,Y,pupilCoords,j):
	# Provides the required Zernike Polynomial
	# Defined on the pupil.
	# The zernike poynomial is indexed by j, with the following:
	# j = {0:Piston, 1:YTilt, 2:XTilt, 3:ObliqueAstig, 4:Focus, 5:VerticalAstig,...
	# 6:VerticalTrefoil, 7:VerticalComa, 8:HorizontalComa, 9:ObliqueTrefoil}
	# Currently, we're only using the first 36

	# X and Y are 1D arrays of the x and y direction
	# Pupil coords is the result of getPupilCoords
	# j is the index of the Zernike


	(cgx,cgy,r) = pupilCoords
	x,y = np.meshgrid(X,Y)
	R = np.sqrt((x-cgx)**2 + (y-cgy)**2)/r
	theta = np.arctan2(y-cgy, x-cgx)

	pupil = np.ones((len(Y),len(X)))
	pupil[np.where(R > 1) ] = np.nan

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

	elif j ==10:
		# Oblique Quadrofail
		zMode = np.sqrt(10) * (R**4) * np.sin(4*theta)

	elif j ==11:
		# Oblique secondary astigmatism
		zMode = np.sqrt(10) * (4*(R**4) - 3*(R**2)) * np.sin(2*theta)

	elif j ==12:
		# Oblique secondary astigmatism
		zMode = np.sqrt(5) * (6*(R**4) - 6*(R**2) + 1) 

	elif j ==13:
		# Primary Spherical
		zMode = np.sqrt(10) * (4*(R**4) - 3*(R**2)) * np.cos(2*theta)

	elif j ==14:
		# vertical Quadrofail
		zMode = np.sqrt(10) * (R**4) * np.cos(4*theta)

	elif j == 15:
		zMode = np.sqrt(12) * (R**5) * np.sin(5*theta)
	
	elif j == 16:
		zMode = np.sqrt(12) * (5*(R**5) - 4*(R**3)) * np.sin(3*theta)

	elif j == 17:
		zMode = np.sqrt(12) * (10*(R**5) - 12*(R**3) + 3*R) * np.sin(theta)

	elif j == 18:
		zMode = np.sqrt(12) * (10*(R**5) - 12*(R**3) + 3*R) * np.cos(theta)

	elif j == 19:
		zMode = np.sqrt(12) * (5*(R**5) - 4*(R**3)) * np.cos(3*theta)
	
	elif j == 20:
		zMode = np.sqrt(12) * (R**5) * np.cos(5*theta)
	
	elif j == 21:
		zMode = np.sqrt(14) * (R**6) * np.sin(6*theta)

	elif j == 22:
		zMode = np.sqrt(14) * (6*(R**6) - 5*(R**4)) * np.sin(4*theta)

	elif j == 23:
		zMode = np.sqrt(14) * (15*(R**6) - 20*(R**4) + 6*(R**2)) * np.sin(2*theta)

	elif j == 24:
		zMode = np.sqrt(7) * (20*(R**6) - 30*(R**4) + 12*(R**2) -1) 

	elif j == 25:
		zMode = np.sqrt(14) * (15*(R**6) - 20*(R**4) + 6*(R**2)) * np.cos(2*theta)

	elif j == 26:
		zMode = np.sqrt(14) * (6*(R**6) - 5*(R**4)) * np.cos(4*theta)

	elif j == 27:
		zMode = np.sqrt(14) * (R**6) * np.cos(6*theta)

	elif j == 28:
		zMode = 4 * (R**7) * np.sin(7*theta)

	elif j == 29:
		zMode = 4 * (7*(R**7) - 6*(R**5)) * np.sin(5*theta)

	elif j == 30:
		zMode = 4 * (21*(R**7) - 30*(R**5) + 10*(R**3)) * np.sin(3*theta)

	elif j == 31:
		zMode = 4 * (35*(R**7) - 60*(R**5) + 30*(R**3) - 4*R) * np.sin(theta)

	elif j == 32:
		zMode = 4 * (35*(R**7) - 60*(R**5) + 30*(R**3) - 4*R) * np.cos(theta)

	elif j == 33:
		zMode = 4 * (21*(R**7) - 30*(R**5) + 10*(R**3)) * np.cos(3*theta)

	elif j == 34:
		zMode = 4 * (7*(R**7) - 6*(R**5)) * np.cos(5*theta)

	elif j == 35:
		zMode = 4 * (R**7) * np.cos(7*theta)

	else:
		print('j out of bounds')
		
		
	return zMode*pupil

def getZernikeCoefficients(X,Y,pupil,phase,numZernikes=36):
	''' Find the zernike coefficients for the phase profile'''
    
	pupilCoords = getPupilCoords(X,Y,pupil)
	(cgx,cgy,r) = pupilCoords

	# Scale everything to a unit radius pupil
	X = X/r
	Y = Y/r
	pupilCoordsScaled = (cgx/r,cgy/r,1) 

	dx = (X[1]-X[0])
	dy = (Y[1]-Y[0])
	#dx = (X[1]-X[0])/r
	#dy = (Y[1]-Y[0])/r

	zList = np.zeros(numZernikes,)
	for i in range(numZernikes):
		Zj = zernike(X,Y,pupilCoordsScaled,i)
		Zj[np.isnan(Zj)]=0
		phase[np.isnan(phase)]=0
		integral = trapz(trapz(Zj*phase,dx=dx,axis=1),dx=dy,axis=0)
		zList[i] = integral/np.pi	
	return zList,pupilCoords

def removeTiltFocus(X,Y,phase,zList,pupilCoords):
	# Scale everything to a unit radius pupil
	(cgx,cgy,r) = pupilCoords
	X = X/r
	Y = Y/r
	pupilCoords = (cgx/r,cgy/r,1) 

	phasePTFR = phase-zList[0]*zernike(X,Y,pupilCoords,0) - zList[1]*zernike(X,Y,pupilCoords,1)-zList[2]*zernike(X,Y,pupilCoords,2)-zList[4]*zernike(X,Y,pupilCoords,4)
	return phasePTFR

def removeAstigmatism(X,Y,phase,zList,pupilCoords):
	# Scale everything to a unit radius pupil
	(cgx,cgy,r) = pupilCoords
	X = X/r
	Y = Y/r
	pupilCoords = (cgx/r,cgy/r,1) 

	phaseNoAstig = phase-zList[3]*zernike(X,Y,pupilCoords,3) - zList[5]*zernike(X,Y,pupilCoords,5)
	return phaseNoAstig


def createWavefront(X,Y,zList,pupilCoords):
	'''Given a set of zernike coefficents, create a wavefront'''
	
	# Scale everything to a unit radius pupil
	(cgx,cgy,r) = pupilCoords
	X = X/r
	Y = Y/r
	pupilCoords = (cgx/r,cgy/r,1) 

	phase = np.zeros((len(Y),len(X)))
	
	for i in range(len(zList)):
		phase = phase + zList[i]*zernike(X,Y,pupilCoords,i)
	
	return phase


def createReference(inChamberHas,leakageHas):
	''' The wavefront outside the chamber was referenced to that inside
	the chamber by placing the haso at both positions and measuring the beam
	in quick succession. This gives us an idea of the aberrations introduced 
	by the leakage line with respect to the interaction point.
	Here we combine the two wavefronts to find the reference.
	
	We need to be careful here, because there is no guarantee they are the same beam size
	To make things simpler we could simply add the zernike modes...rather than regridding the
	wavefront
	
	In creating this reference, we need to perform some transformations on the In Chamber HASO to
	correctly align it with the out of chamber HASO.

	By Robs calculation: from just before the leakage splitter to the in chamber HASO there are
	- 7 in plane reflections: Flips beam horizontally
	- 1 turning periscope: Rotates beam counter clockwise by 90 degrees (as viewed head on)
	- 1 keplerian telescope: Center transform of beam.

	And from just before the leakage split to the leakage HASO there are:
	- 7 in plane reflections: Flips beam horizontally
	- 1 keplerian telescope: Center transform of beam.

	Thus the beams are simply offset by 90 degrees. As viewed on the camera, this 90 degree offset
	introduced by the turning periscope should be clockwise as seen on the CCD. Thus a 90 degree anti
	clockwise transform is to be performed on the in chamber HASO phase to match it up
	'''

	(XIC,YIC,phaseIC,intensityIC,pupilIC,pupilCoordsIC,zernikeCoeffsIC) = extractWavefrontInfo(inChamberHas)
	
	# need to rotate the phase nd pupil by 90 degreees clockwise
	phaseIC_rot = np.rot90(phaseIC,k=3)
	pupilIC_rot = np.rot90(pupilIC,k=3)

	# Now get zernike list. Note we've swaped X and Y axis as the image has been rotated by 90 degrees
	zernikeCoeffsIC_rot,pupilCoordIC_rot = getZernikeCoefficients(YIC,XIC,pupilIC_rot,phaseIC_rot)


	# Now import the 
	(XLK,YLK,phaseLK,intensityLK,pupilLK,pupilCoordsLK,zernikeCoeffsLK) = extractWavefrontInfo(leakageHas)

	return zernikeCoeffsIC_rot-zernikeCoeffsLK

