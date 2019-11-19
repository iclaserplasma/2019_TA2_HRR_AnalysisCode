import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import medfilt
from sqlDatabase import *


SQLDATABASE = 'TA2SQLConnection.txt'


class dataRun:

	# ADMIN

	def __init__(self, baseDataFolder, baseAnalysisFolder, runDate, runName,refRunDate,refRunName,verbose=0):
		# baseDataFolder is the location of the Mirage Data folder
		# runDate is a string
		# runName is a string
		
		# Options	
		self.verbose = verbose

		self.baseDataFolder 	= baseDataFolder
		self.baseAnalysisFolder	= baseAnalysisFolder
		
		# Data Information
		self.runDate 			= runDate
		self.runName 			= runName
		self.diagList 			= self.findAvailableDiagnostics(runDate,runName)
		datStyle, datShots, diagShotDict = self.runsOrBursts(runDate,runName,self.diagList)
		self.datStyle 			= datStyle
		self.datShots			= datShots
		self.diagShotDict 		= diagShotDict
		self.analysisPath 		= self.createAnalysisFolder()

		# Reference Information
		self.refRunDate			= refRunDate
		self.refRunName			= refRunName
		self.refDiagList 		= self.findAvailableDiagnostics(refRunDate,refRunName)
		refStyle, refShots, refShotDict	= self.runsOrBursts(refRunDate,refRunName,self.refDiagList)
		self.refStyle 			= refStyle
		self.refShots			= refShots
		self.refShotDict 		= refShotDict
		


		# Collect SQL Data
		self.collectSQLData()

	# ELECTRON SPECTRUM ANALYSIS CALLS
	def initESpecAnalysis(self,overwriteData):
		# Load in reference
		# Analyse each shot in run and save key parameters
		# update log

	def getESpecCharge(self,overwriteData):
		# Load the espec charge for the run
		# if it exists get it, if not, run initESpecAnalysis and update log
		return sho, charge

	# HASO DIAGOSTIC

	def findAvailableDiagnostics(self,runDate,runName):
		# By following through the folder system, this will tell us what the diagnostics were for the run
		baseDataFolder 	= self.baseDataFolder
		
		diagList = next(os.walk(baseDataFolder))[1] # diagnostics are folders in the baseDataFolder
		unavailableDiags = []

		for diag in diagList:
		    diagAvailable = True
		    diagAvailable = os.path.exists(os.path.join(baseDataFolder, diag, runDate, runName))
		    if diagAvailable is not True:
		        unavailableDiags.append(diag)
		for diag in unavailableDiags:
		    diagList.remove(diag)
		
		if self.verbose:
			print('\n\nList of Available Diagnostics for ' + os.path.join(runDate,runName) + '\n')
			for element in diagList:
				print(element)

		return(diagList)

	def runsOrBursts(self,runDate,runName,diagList):
		# function to determine if the run was single shots or bursts
		# if the data is singleShot form, shots is the number of shots taken
		# if the data is in burst form then shots is a tuple (numBursts,numShots)
		baseDataFolder 	= self.baseDataFolder
		diagShotDict = {} # create a dictionary in which we will store the shot information

		testDiag = diagList[0]
		diagPath = os.path.join(baseDataFolder, testDiag, runDate, runName)
		lsDir = os.listdir(diagPath)
		if os.path.isdir(os.path.join(diagPath,lsDir[0])):
			dataStyle = 'burst'
			# In this case we want to know how many bursts and how many shots per burst
			numBursts,numShots = (0,0)
			for diag in diagList:
				try:
					diagPath = os.path.join(baseDataFolder, diag, runDate, runName)
					diagLsDir = os.listdir(diagPath)
					tmpNumBursts = len(diagLsDir)
					for burst in diagLsDir:
						tmpNumShots = len(os.listdir(os.path.join(diagPath,burst)))
						if tmpNumShots > numShots:
							numShots = tmpNumShots
					if tmpNumBursts > numBursts:
						numBursts = tmpNumBursts
					if self.verbose:
						print('Diag: '+ diag + '(numBursts,numShots) = (%i,%i)' %(tmpNumBursts,tmpNumShots))
					diagShotDict[diag] = (tmpNumBursts,tmpNumShots)
				except:
					print('somethings gone wrong in runsOrBursts function')		
			shots = (numBursts,numShots)

		else:
			dataStyle = 'singleShot'
			# in this case we just want to know the number of shots
			for diag in diagList:
				try:
					numShots = len(lsDir)
					diagPath = os.path.join(baseDataFolder, diag, runDate, runName)
					diagLsDir = os.listdir(diagPath)
					tmpNumShots = len(diagLsDir)
					if tmpNumShots > numShots:
						numShots = tmpNumShots
					diagShotDict[diag] = tmpNumShots
					if self.verbose:
						print('Diag: '+ diag + 'numShots = %i' %(tmpNumShots))
					
				except:
					print('somethings gone wrong in runsOrBursts function')	
					
			shots = numShots
			


		return dataStyle, shots, diagShotDict

	def createAnalysisFolder(self):
		# Checks if an analysisFolder already exists.
		# If not, one will be created
		baseAnalysisFolder 	= self.baseAnalysisFolder
		runDate 		   	= self.runDate
		runName				= self.runName
		analysisPath = os.path.join(baseAnalysisFolder, runDate, runName)
		if os.path.exists(analysisPath) is not True:
			os.makedirs(analysisPath)
			if self.verbose:
				print('\n\nAnalysis folder has been created\n')
		else:
			if self.verbose:
				print('\n\nAnalysis folder already exists\n')
		return(analysisPath)


	def collectSQLData(self):
		db = connectToSQL(SQLDATABASE,True)
		runName = self.runName
		runDate = self.runDate

		# Get all data pertaining to the run
		cursor = db.cursor()
		command = "SELECT shot_or_burst,GasCellPressure,GasCellLength FROM shot_data WHERE run='"+runDate+"/"+runName+"'"
		
		cursor.execute(command)
		SQLData = cursor.fetchall() ## it returns list of tables present in the database
		## showing all the tables one by one
		gasCellPressure = {}
		gasCellLength = {}

		for datum in SQLData:
		    shotOrBurst,Pressure,Length = datum
		    gasCellPressure[shotOrBurst] = Pressure
		    gasCellLength[shotOrBurst] = Length


		self.saveData('GasCellPressure',gasCellPressure)
		self.saveData('gasCellLength',gasCellLength)


	# MAIN ANALYSIS ROUTINES


	def analyseCounts(self,diag):
		# get the averaged XRay counts
		bgImg = self.averageReferenceImages(diag)
		#apply a median filter to prevent hot pixels fucking with stuff
		bgImg = medfilt(bgImg,5)

		# Now run through each data image, subtract the 
		baseDataFolder = self.baseDataFolder
		runName = self.runName
		runDate = self.runDate
		diagShotDict = self.diagShotDict
		datShots = diagShotDict[diag] 

		datPath = os.path.join(baseDataFolder,diag,runDate,runName)
		counts = np.zeros(datShots)

		if type(datShots) is tuple:
			
			# For each burst, go through, pull each shot and subtract the bg
			# calculate the counts and store in a big array
			bursts = os.listdir(datPath)
			bCntr = 0
			burstNums =[]
			for burst in bursts:
				shots = os.listdir(os.path.join(datPath,burst))
				sCntr = 0
				for shot in shots:
					try:
						img = self.getImage(os.path.join(datPath,burst,shot))
						counts[bCntr,sCntr] = self.getCounts(img-bgImg)
					except:
						print('Index Out of Bounds: (%i,%i)' %(bCntr,sCntr))
					sCntr = sCntr + 1
				bCntr = bCntr + 1
				if self.verbose:
					print('Analysed: ' + os.path.join(datPath,burst))
				burstNums.append( int(burst[5:]))
		# Save data to be analysed later on
		diagCountsName = diag + 'Counts'
		diagBNumName = diag + 'Counts' + 'BNums'
		self.saveData(diagCountsName,counts)
		self.saveData(diagBNumName,burstNums)

		return(counts)	



	def saveAveragedImgsFromBurst(self,diag,meanImgCountThreshold,bgRegion=None):
		# get the averaged images for each burst in a run
		# provided the image counts are above meanImgCountThreshold

		baseDataFolder = self.baseDataFolder
		runName = self.runName
		runDate = self.runDate
		diagShotDict = self.diagShotDict
		datShots = diagShotDict[diag] 

		datPath = os.path.join(baseDataFolder,diag,runDate,runName)
		bursts = os.listdir(datPath)
		bCntr = 0

		# Set up background stuff
		if bgRegion == None: 
			bgImg = self.averageReferenceImages(diag)

			#apply a median filter to prevent hot pixels fucking with stuff
			bgImg = medfilt(bgImg,5)
			(m,n) = bgImg.shape

		else:
			# If no bg image is to be used, we nee to find the size of an image
			shots = os.listdir(os.path.join(datPath,bursts[0]))
			img = self.getImage(os.path.join(datPath,bursts[0],shots[0]))
			(m,n) = img.shape

		# For each burst, go through, pull each shot and subtract the bg
		# calculate the counts and store in a big array

		for burst in bursts:
			burstImg = np.zeros((m,n))
			shots = os.listdir(os.path.join(datPath,burst))
			sCntr = 0
			for shot in shots:
				try:
					img = self.getImage(os.path.join(datPath,burst,shot))
					if bgRegion is not None:
						img = self.subtractRemainingBg(img,bgRegion)
						img[np.where(img < 0)] = 0

					else:
						img = img - bgImg
						img[np.where(img < 0)] = 0

					counts = self.getCounts(img)
					meanCounts = float(counts)/float((m*n))

					if meanCounts > meanImgCountThreshold:
						burstImg = burstImg + img
						sCntr = sCntr + 1

				except:
					print('Index Out of Bounds: (%i,%i)' %(bCntr,sCntr))
						
			bCntr = bCntr + 1
			if sCntr > 0:
				# Average the images
				burstImg = burstImg/float(sCntr)
				# Now save the data
				filename = 'burst' + str(bCntr) + 'averagedImg'
				dataName = os.path.join(diag,filename)
				self.saveData(dataName,burstImg)

			if self.verbose:
				print('Image Saved for : ' + os.path.join(datPath,burst))

	


	# SUBROUTINES
	def getImage(self,filePath):
		# We pass the full file path to here
		img = plt.imread(filePath).astype(float)
		return img

	def averageImagesInFolder(self,folderPath):
		imgFiles = os.listdir(folderPath)
		tmpImg = self.getImage(os.path.join(folderPath,imgFiles[0]))
		(rows, cols) = tmpImg.shape
		img = np.zeros((rows,cols))

		cntr = 0
		for file in imgFiles:
			try:
				tmpImg = self.getImage(os.path.join(folderPath,file))
				img = img + tmpImg
				cntr = cntr + 1
			except:
				print('Could not add ' + os.path.join(folderPath,file) +' to averaged image')
			img = img/cntr
		return(img)

	def averageReferenceImages(self,diag):
		# Average the images of a particular diagnostic
		baseDataFolder = self.baseDataFolder
		
		# First lets get the reference counts
		refRunDate = self.refRunDate
		refRunName = self.refRunName

		refPath = os.path.join(baseDataFolder,diag,refRunDate,refRunName)

		if type(self.refShots) is tuple:
			# In this case go through burst by burst
			bursts = os.listdir(refPath)
			bgImg = None
			cntr = 0
			for burst in bursts:
				tmpImg = self.averageImagesInFolder(os.path.join(refPath,burst))
				if bgImg is None:
					bgImg = tmpImg
					cntr = 1
				else:
					bgImg = bgImg + tmpImg
					cntr = cntr+1
			bgImg = bgImg/cntr

		else:
			# go through image by image and average them
			bgImg = self.averageImagesInFolder(refPath)

		return bgImg

	def getCounts(self,img):
		# Simple routine to get the total counts of an image
		return(np.sum(img))


	def subtractRemainingBg(self,img,bgRegion):
		# Sometimes the background subtraction fails, so lets subtract
		# the remaining using bgRegion as a signal free region
		# Should only be used if necessary

		# img is the image
		# bgRegion is a tupe of (left right bottom top)

		bgLevel = np.mean(img[bgRegion[2]:bgRegion[3], bgRegion[0]:bgRegion[1] ])
		return (img - bgLevel)


	def saveData(self,dataName,data):
		# Saves analysed Data 
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		# Check if the folder exists, bearing in mind that we might have an extra
		# path in the dataName
		fullFilePath = os.path.join(baseAnalysisFolder,runDate,runName,dataName)
		# check if it exists
		if os.path.exists(os.path.dirname(fullFilePath)) is not True:
			os.mkdir(os.path.dirname(fullFilePath))

		np.save(fullFilePath, data)


	def saveRunObject(self):
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName
		np.save(os.path.join(baseAnalysisFolder,runDate,runName,'runObject'), self)

	def createDataBuckets(arr):
	    # Bin the data using too many (although still a reasonable amount of) bins.
	    numElements = round(len(arr)/2)
	    bins = np.linspace(np.amin(arr),np.amax(arr),numElements)
	    vals,edges=np.histogram(arr,bins)

	    # Fourier transform this
	    BFT = np.fft.fft(vals)

	    # Find the first peak of the FFT that isn't the DC peak
	    peaks,properties = find_peaks(BFT)
	    sortedIndexs = np.argsort(BFT[peaks]) # sorts in ascending order
	    sortedIndexs[-2:]
	    peaks = peaks[sortedIndexs[-2:]]
	    peak = np.amin(peaks)

	    # This tells us the number of bins (minus 1) that should be used in the binning
	    numElements = peak + 1
	    #bins = np.linspace(np.amin(arr),np.amax(arr),numElements)
	    #vals,edges=np.histogram(arr,bins)

	    # find the average value in the first bucket and the last bucket
	    datRange = np.amax(arr) - np.amin(arr)
	    binWidth = datRange/numElements
	    tmp = np.where(arr < np.amin(arr)+ binWidth)
	    bin1 = np.mean([arr[i] for i in tmp[0]])

	    tmp2 = np.where(arr > np.amax(arr)- binWidth)
	    bin2 = np.mean([arr[i] for i in tmp2[0]])

	    binCenters = np.linspace(bin1,bin2,numElements)
	    binWidth = (bin2-bin1)/numElements
	    
	    return binCenters,binWidth

	







