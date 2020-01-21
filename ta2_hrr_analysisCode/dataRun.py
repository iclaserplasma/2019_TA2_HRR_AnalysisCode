import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import medfilt
from sqlDatabase import connectToSQL
import logging
import csv 
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# IMPORT DIAGNOSTIC CODES
import SPIDERAnalysis
import HASOAnalysis
import ESpecAnalysis


# HELPER FUNCTIONS - COULD BE PLACED ELSEWHERE?
def getSortedFolderItems(itemPath,key):
	itemList = os.listdir(itemPath)
	iList = []
	iNum = []
	for s in itemList:
		if key in s:
			iList.append(s)
			iNum.append(int(''.join(filter(str.isdigit, s))))
	return [x for _,x in sorted(zip(iNum,iList))]

def getShotFilePathDict(runPath):
	burstList = getSortedFolderItems(runPath,'Burst')
	filePathDict = {}
	for b in burstList:
		bPath = os.path.join(runPath,b)
		fileList = getSortedFolderItems(bPath,'Shot')
		filePathDict[b] = [os.path.join(bPath, f) for f in fileList]
	return filePathDict

def loadRunObject(path):
	'''Load a pickled Run object'''
	f = open(path, 'rb')
	run = pickle.load(f)
	return run

class dataRun:



	# -------------------------------------------------------------------------------------------------------------------
	# -----										WHATS IN HERE????		 											-----
	# -------------------------------------------------------------------------------------------------------------------

	# Housekeeping and admin
	#	__init__
	#	findAvailableDiagnostics
	#	runsOrBursts
	#	createAnalysisFolder
	#	collectSQLData
	#
	# Diagnostic Function Calls
	# 	Espec
	#		performESpecAnalysis
	#		getESpecCharge
	#	SPIDER
	#		getSpectralPhaseOrders
	#		getTemporalProfile
	#	
	# A Selection of Generic Functions
	#	getDiagDataPath	
	#	getDiagAnalysisPath
	#	getImage
	#	averageImagesInFolder
	#	averageReferenceImages
	#	getCounts
	#	subtractRemainingBg
	#	createDataBuckets
	#
	# Saving and Loading Analysed Data
	#	saveData
	#	saveRunObject





	# -------------------------------------------------------------------------------------------------------------------
	# -----										HOUSEKEEPING AND ADMIN	 											-----
	# -------------------------------------------------------------------------------------------------------------------


	def __init__(self, baseDataFolder, baseAnalysisFolder, calibrationFolder, runDate, runName,refRunDate,refRunName,verbose=0):
		# baseDataFolder is the location of the Mirage Data folder
		# runDate is a string
		# runName is a string
		
		# Options	
		self.verbose = verbose

		self.baseDataFolder 	= baseDataFolder
		self.baseAnalysisFolder	= baseAnalysisFolder
		self.calibrationFolder  = calibrationFolder

		# Data Information
		self.runDate 			= runDate
		self.runName 			= runName
		loggingAnalysisFolder 	= self.createAnalysisFolder()
		loggerFile 				= os.path.join(loggingAnalysisFolder,'analysisInfo.log')
		logging.basicConfig(filename=loggerFile, filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S',level=logging.INFO)
		self.logger 			= logging.getLogger('logFile')
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
		
		self.logger.info('\n\nList of Available Diagnostics for ' + os.path.join(runDate,runName) + '\n')
		for element in diagList:
				self.logger.info(element)
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
		# This folder is a general non diagnostic specific folder
		# If not, one will be created
		baseAnalysisFolder 	= self.baseAnalysisFolder
		diagName 			= 'General'
		runDate 		   	= self.runDate
		runName				= self.runName
		analysisPath = os.path.join(baseAnalysisFolder, diagName , runDate, runName)
		if os.path.exists(analysisPath) is not True:
			os.makedirs(analysisPath)
			if self.verbose:
				print('\n\nGeneral Analysis folder has been created\n')
		else:
			if self.verbose:
				print('\n\nGeneral Analysis folder already exists\n')
		return(analysisPath)


	def collectSQLData(self):
		db = connectToSQL(True)
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

		analysisPath,isItReal = self.getDiagAnalysisPath('General')
		self.saveData(os.path.join(analysisPath,'GasCellPressure'),gasCellPressure)
		self.saveData(os.path.join(analysisPath,'gasCellLength'),gasCellLength)





	# -------------------------------------------------------------------------------------------------------------------
	# -----										DIAGNOSTIC FUNCTION CALLS 											-----
	# -------------------------------------------------------------------------------------------------------------------



	# ELECTRON SPECTRUM ANALYSIS CALLS -- THIS IS CURRENTLY JUST AN EXAMPLE
	def performESpecAnalysis(self,overwriteData=False):
		# Check if analysed data already exists
		# if it does and overWriteData == False, then exit function
		# otherwise, perform analysis
		# 	Load in reference eSpec Data
		# 	Analyse each shot in run and save key parameters to file using the standard save funcitons below
		# 	update log

		return 0

	def getESpecCharge(self,useCalibration=True,overwriteData=False):
		# Load the espec charge for the run from the analysed data
		# if it exists get it, if not, run initESpecAnalysis and update log
		diag = 'HighESpec'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		
		if useCalibration:
			eSpecCalib = self.loadCalibrationData(diag)
		# NEED TO PUT ALTERNATIVE CALIBRATION TUPLE HERE TO HELP NICS CODE RUN

		for burstStr in filePathDict.keys():		
			if useCalibration:
				analysedData = ESpecAnalysis.extractCharge(filePathDict[burstStr],eSpecCalib)
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecCharge')
				self.saveData(analysisSavePath,analysedData)

			else:
				analysedData = ESpecAnalysis.extractCharge(filePathDict[burstStr])
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecCharge_NoCalibration')
				self.saveData(analysisSavePath,analysedData)
			
			print('Analysed ESpec Charge for Burst '+ burstStr)



	# SPIDER ANALYSIS CALLS -- THIS IS CURRENTLY JUST AN EXAMPLE
	def getSpectralPhaseOrders(self):
		diag = 'SPIDER'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		for burstStr in filePathDict.keys():		
			analysedData = SPIDERAnalysis.polyOrders(filePathDict[burstStr])
			# Save the data
			analysisSavePath = os.path.join(analysisPath,burstStr,'analysis')
			self.saveData(analysisSavePath,analysedData)

		# Print some shit to the log here. Someone to write function.

		return 0

	# HASO Analysis
	def performHASOAnalysis(self,useChamberCalibration=True):
		diag = 'HASO'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		
		if useChamberCalibration:
			zernikeOffsets = self.loadCalibrationData(diag)
		
		for burstStr in filePathDict.keys():		
			if useChamberCalibration:
				analysedData = HASOAnalysis.extractCalibratedWavefrontInfo(filePathDict[burstStr],zernikeOffsets)
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'calibratedWavefront')
				self.saveData(analysisSavePath,analysedData)

			else:
				analysedData = HASOAnalysis.extractWavefrontInfo(filePathDict[burstStr])
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'waveFrontOnLeakage')
				self.saveData(analysisSavePath,analysedData)
			
			print('Analysed HASO '+ burstStr)


		

	# -------------------------------------------------------------------------------------------------------------------
	# -----								A SELECTION OF GENERIC FUNCTIONS											-----
	# -------------------------------------------------------------------------------------------------------------------


	# SUBROUTINES

	def getDiagDataPath(self,diag):
		# Gives you back the relative paths to retrieve data from
		# DOESN'T WORK FOR ANALYSIS OF DARKFIELDS OR REFERNECES
		baseDataFolder 	= self.baseDataFolder
		runDate 		   	= self.runDate
		runName				= self.runName
		dataPath = os.path.join(baseDataFolder, diag, runDate, runName)

		return dataPath


	def getDiagAnalysisPath(self,diag):
		# Gives you back the relative analysis path to save analysis data or retrieve analysis data from
		# DOESN'T WORK FOR ANALYSIS OF DARKFIELDS OR REFERNECES
		baseAnalysisFolder 	= self.baseAnalysisFolder
		runDate 		   	= self.runDate
		runName				= self.runName
		analysisPath = os.path.join(baseAnalysisFolder, diag, runDate, runName)

		# check if it exists
		isItReal = os.path.exists(analysisPath)

		return analysisPath, isItReal

	def createRunPathLists(self,diag):
		# Get RelevantPaths
		# analysisPath, analysisPathExists = self.getDiagAnalysisPath(diag)
		dataPath = self.getDiagDataPath(diag)
		filePathDict = getShotFilePathDict(dataPath)
		return filePathDict

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




	# -------------------------------------------------------------------------------------------------------------------
	# -----								SAVING AND LOADING ANALYSED DATA 											-----
	# -------------------------------------------------------------------------------------------------------------------

	#def saveData(self,dataName,data):
	#	# Saves analysed Data 
	#	baseAnalysisFolder = self.baseAnalysisFolder
	#	runDate = self.runDate
	#	runName = self.runName

	#	# Check if the folder exists, bearing in mind that we might have an extra
	#	# path in the dataName
	#	fullFilePath = os.path.join(baseAnalysisFolder,runDate,runName,dataName)
	#	# check if it exists
	#	if os.path.exists(os.path.dirname(fullFilePath)) is not True:
	#		os.mkdir(os.path.dirname(fullFilePath))

	#	np.save(fullFilePath, data)

	def saveData(self,path,data):
		# Check if the folder exists, bearing in mind that we might have an extra
		dirName = os.path.dirname(path)
		if os.path.exists(dirName) is not True:
			os.makedirs(dirName)
		np.save(path, data)

	def saveRunObject(self):
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName
		outputFile = open(os.path.join(baseAnalysisFolder,'General',runDate,runName,'runObject.pkl'), 'wb')
		pickle.dump(self,outputFile,protocol=0)
		outputFile.close()

	
	def loadCalibrationData(self, diag):
		''' Find the calibration data for a particular diagnostic
		This function will open the calibration look up csv file and find
		the relevant folder containing the calibrations for the chosen data run
		It will then load the data and spit it out as a return
		'''

		# Get data run and construct date/run string
		runDate = self.runDate
		runName = self.runName

		dateRunString = runDate + '\\' + runName

		# Now open the csv file and find the row in which the first column entry matches the dateRunString
		calibrationFolder = self.calibrationFolder
		calibrationPath = os.path.join(calibrationFolder,'CalibrationPaths.csv')
		csv_file = csv.reader(open(calibrationPath, "r"), delimiter=",")

		cntr = 0
		for row in csv_file:
			if cntr == 0:
				# This row contains the list of diagnostics
				diagList = row
			cntr = 1
			dateRunStringTest = row[0]
			if dateRunStringTest == dateRunString:
				break

		cntr = 0
		for tmpDiag in diagList:
			if diag == tmpDiag:
				break
			cntr = cntr + 1
			
		calibrationFilePath = row[cntr]

		# Now Load in the data 
		try:
			calibData = np.load(os.path.join(calibrationFolder,calibrationFilePath),allow_pickle=True)
		except:
			print('Info Recieved')
			print(os.path.join(calibrationFolder,calibrationFilePath))
			raise Exception('COULD NOT LOAD CALIBRATION FILE, PLEASE CHECK IT EXISTS AND THE PATH IN calibrationPaths.csv IS CORRECT')


		return calibData








