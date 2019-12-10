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

		X = np.linspace(-n*step/2,n*step/2-1,n)
		Y = np.linspace(-m*step/2,m*step/2-1,m)

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
		os.remove('Pupil.xml')

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
	''' Converted to python from Matlab code by John D'Errico
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
	oldXRes = (X[-1]-X[0])/n
	oldYRes = (Y[-1]-Y[0])/m

	xRes = (X[-1]-X[0])/n/4
	yRes = (Y[-1]-Y[0])/m/4
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

def getZernikeCoefficients(X,Y,pupil,phase):
	''' Find the zernike coefficients for the phase profile'''
    
	pupilCoords = getPupilCoords(X,Y,pupil)
	(cgx,cgy,r) = pupilCoords
	dx = (X[1]-X[0])/r
	dy = (Y[1]-Y[0])/r

	zList = np.zeros(36,)
	for i in range(36):
		Zj = zernike(X,Y,pupilCoords,i)
		Zj[np.isnan(Zj)]=0
		phase[np.isnan(phase)]=0
		integral = trapz(trapz(Zj*phase,dx=dx,axis=1),dx=dy,axis=0)
		zList[i] = integral/np.pi	
	return zList,pupilCoords

def removeTiltFocus(X,Y,phase,zList,pupilCoords):
	phasePTFR = phase-zList[0]*zernike(X,Y,pupilCoords,0) - zList[1]*zernike(X,Y,pupilCoords,1)-zList[2]*zernike(X,Y,pupilCoords,2)-zList[4]*zernike(X,Y,pupilCoords,4)
	return phasePTFR

def createWavefront(X,Y,zList,pupilCoords):
	'''Given a set of zernike coefficents, create a wavefront'''

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
	wavefront'''

	(XIC,YIC,phaseIC,intensityIC,pupilIC,pupilCoordsIC,zernikeCoeffsIC) = extractWavefrontInfo(inChamberHas)
	(XLK,YLK,phaseLK,intensityLK,pupilLK,pupilCoordsLK,zernikeCoeffsLK) = extractWavefrontInfo(leakageHas)

	return zernikeCoeffsIC-zernikeCoeffsLK