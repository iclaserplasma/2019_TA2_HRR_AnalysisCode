import os
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import medfilt
from sqlDatabase import connectToSQL
import logging
import csv 
# Works with file paths on all operating systems
from pathlib import Path

try:
	import cPickle as pickle
except ModuleNotFoundError:
	import pickle
from datetime import datetime

# IMPORT DIAGNOSTIC CODES
import SPIDERAnalysis
import HASOAnalysis
import ESpecAnalysis
import PreCompNFAnalysis
import XRayAnalysis 


try:
	import probe_density_extraction 
except:
	print('Cannot import probe_density_extraction')

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
	# -----										HOUSEKEEPING AND ADMIN	 											-----
	# -------------------------------------------------------------------------------------------------------------------


	def __init__(self, baseDataFolder, baseAnalysisFolder, calibrationFolder, runDate, runName,verbose=0,overwrite=False):
		# baseDataFolder is the location of the Mirage Data folder
		# runDate is a string
		# runName is a string
		
		# First Check if the run Object already exists. If it does, then just load it in.

		runObjectPath = os.path.join(baseAnalysisFolder , 'General' , runDate , runName , 'runObject.pkl' )
		doesRunObjectExist = os.path.isfile(runObjectPath)

		if doesRunObjectExist and not overwrite:
			oldRun = pickle.load( open( runObjectPath, "rb" ) )
			print('Run Object already Exists. Loading it in. If you would prefer to start from scratch, use the argument overwrite==True')
			self.verbose = verbose

			self.baseDataFolder 	= oldRun.baseDataFolder
			self.baseAnalysisFolder	= oldRun.baseAnalysisFolder
			self.calibrationFolder  = oldRun.calibrationFolder

			# Data Information
			self.runDate 			= oldRun.runDate
			self.runName 			= oldRun.runName
			loggingAnalysisFolder 	= self.createAnalysisFolder()
			self.loggerFile 		= oldRun.loggerFile
			self.logThatShit('Analysis Continued\n')
			

			self.diagList 			= oldRun.diagList
			self.datStyle 			= oldRun.datStyle
			self.datShots			= oldRun.datShots
			self.diagShotDict 		= oldRun.diagShotDict
			self.analysisPath 		= self.createAnalysisFolder()
		else:

			# Options	
			self.verbose = verbose

			self.baseDataFolder 	= baseDataFolder
			self.baseAnalysisFolder	= baseAnalysisFolder
			self.calibrationFolder  = calibrationFolder

			# Data Information
			self.runDate 			= runDate
			self.runName 			= runName

			loggingAnalysisFolder 	= self.createAnalysisFolder()
			self.loggerFile 		= os.path.join(loggingAnalysisFolder,'analysisInfo.log')
			
			# Open the first log and add to it
			loggerFile = self.loggerFile
			f = open(loggerFile, "w")
			now = datetime. now()
			current_time = now. strftime('%d/%m/%Y %H:%M:%S')	
			f.write('\n' + current_time + ' - ' + 'Run Analysis Initiated\n')
			f.close()
			
			self.diagList 			= self.findAvailableDiagnostics(runDate,runName)
			datStyle, datShots, diagShotDict = self.runsOrBursts(runDate,runName,self.diagList)
			self.datStyle 			= datStyle
			self.datShots			= datShots
			self.diagShotDict 		= diagShotDict
			self.analysisPath 		= self.createAnalysisFolder()


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
		
		self.logThatShit('\n\nList of Available Diagnostics for ' + os.path.join(runDate,runName) + '\n')
		for element in diagList:
				self.logThatShit(element)
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
				print('\n\nGeneral Analysis folder has been created\n{}\n'.format(analysisPath))
		else:
			if self.verbose:
				print('\n\nGeneral Analysis folder already exists\n{}\n'.format(analysisPath))
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

	# PROBE ANALYSIS
	def performProbeDensityAnalysis(self, ):
		diag = 'Probe_Interferometry'
		print ("In ", diag)

		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		print (analysisPath, pathExists)
		
		probeCalib = self.loadCalibrationData(diag)
		for burstStr in filePathDict.keys():		
			analysisSavePath = os.path.join(analysisPath,burstStr,'{}_Analysis'.format(diag))
			analysedData = probe_density_extraction.extract_plasma_density(
								filePathDict[burstStr],probeCalib, analysisSavePath)
			
			
			self.saveData(analysisSavePath,analysedData)

			self.logThatShit('Performed Probe_Interferometry Analysis for ' + burstStr)
			print('Analysed Probe_Interferometry '+ burstStr)

	# ELECTRON ANALYSIS 
	def performESpecAnalysis(self,useCalibration=True,overwriteData=False):
		# Load the espec images for the run and analyse
		# if it exists get it, if not, run initESpecAnalysis and update log
		diag = 'HighESpec'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		
		if useCalibration:
			eSpecCalib = self.loadCalibrationData(diag)
		# NEED TO PUT ALTERNATIVE CALIBRATION TUPLE HERE TO HELP NICS CODE RUN

		for burstStr in filePathDict.keys():		
			if useCalibration:
				analysedData = ESpecAnalysis.ESpecSCEC(filePathDict[burstStr],eSpecCalib)
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecAnalysis')
				self.saveData(analysisSavePath,analysedData)

			else:
				analysedData = ESpecAnalysis.ESpecSCEC(filePathDict[burstStr])
				# Save the data
				analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecAnalysis_NoCalibration')
				self.saveData(analysisSavePath,analysedData)
			self.logThatShit('Performed HighESpec Analysis for ' + burstStr)
			print('Analysed ESpec Spectrum, Charge, totalEnergy, cutoffEnergy95 for Burst '+ burstStr)



	# SPIDER ANALYSIS CALLS -- THIS IS CURRENTLY JUST AN EXAMPLE
	def performSPIDERAnalysis(self):
		diag = 'SPIDER'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)

		for burstStr in filePathDict.keys():	
			for filePath in filePathDict[burstStr]:	
				analysedData = SPIDERAnalysis.analyseSPIDERData(filePath)
				
				# Save the data
				l = Path(filePath)		# Splitting the filepath in all operating systems, '\\' and '/'
				filename = l.parts[-1]
				filename = filename[0:-4] + '_Analysis'

				analysisSavePath = os.path.join(analysisPath,burstStr,filename)
				print (analysisPath, analysisSavePath)
				self.saveData(analysisSavePath,analysedData)
			self.logThatShit('Performed SPIDER Analysis for ' + burstStr)
			print('Analysed SPIDER '+ burstStr)
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
			self.logThatShit('Performed HASO Analysis for ' + burstStr)
			print('Analysed HASO '+ burstStr)


	# PRE COMP NF (BEAM ENERGY) Anlaysis
	def performPreCompNFAnalysis(self):
		diag = 'PreCompNF'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)

		NFCalib = self.loadCalibrationData(diag)

		for burstStr in filePathDict.keys():
			avgEnergy = 0
			for filePath in filePathDict[burstStr]:	
				analysedData = PreCompNFAnalysis.analyseNFImage(filePath,NFCalib)
				avgEnergy = avgEnergy + analysedData[0]

				# Save the data
				l = Path(filePath)		# Splitting the filepath in all operating systems, '\\' and '/'
				filename = l.parts[-1]				
				filename = filename[0:-5] + '_Analysis'

				analysisSavePath = os.path.join(analysisPath,burstStr,filename)
				self.saveData(analysisSavePath,analysedData)
			avgEnergy = avgEnergy/len(filePathDict[burstStr])
			energyStr = '%.3f J' %(avgEnergy)
			self.logThatShit('Performed NF Analysis for ' + burstStr +'. Avg Pulse Energy On Target = ' +  energyStr)
			print('Performed NF Analysis for ' + burstStr +'. Avg Pulse Energy On Target = ' +  energyStr)
		return 0	


	# X-Ray Anlaysis
	def performXRayAnalysis(self,justGetCounts=False):
		diag = 'XRay'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)

		xrayCalib = self.loadCalibrationData(diag)

		for burstStr in filePathDict.keys():
			if justGetCounts:
				# Here we're going to avoid all actual x-ray code and just sum the image counts
				# Not even using a background. This is to mimic what happened during optimization
				imgCounts  = 0
				cntr = 0
				for filePath in filePathDict[burstStr]:	
					imgCounts += np.sum(plt.imread(filePath).astype(float))
					cntr +=1
				analysedData = imgCounts/cntr
				analysisSavePath = os.path.join(analysisPath,burstStr,'XRayImgCounts')
				self.saveData(analysisSavePath,analysedData)
				self.logThatShit('Averaged Counts for XRay images for ' + burstStr)
				print('Averaged Counts for XRay images for ' + burstStr )
				
			else:
				analysedData = XRayAnalysis.XRayEcrit(filePathDict[burstStr],xrayCalib)
				analysisSavePath = os.path.join(analysisPath,burstStr,'XRayAnalysis')
				self.saveData(analysisSavePath,analysedData)
				self.logThatShit('Performed XRay Analysis for ' + burstStr)
				print('Performed XRay Analysis for ' + burstStr )
			
		
		return 0	

	# -------------------------------------------------------------------------------------------------------------------
	# -----								A SELECTION OF GENERIC FUNCTIONS											-----
	# -------------------------------------------------------------------------------------------------------------------


	# SUBROUTINES

	def getDiagDataPath(self,diag):
		# Gives you back the relative paths to retrieve data from
		baseDataFolder 	= self.baseDataFolder
		runDate 		   	= self.runDate
		runName				= self.runName
		dataPath = os.path.join(baseDataFolder, diag, runDate, runName)

		return dataPath


	def getDiagAnalysisPath(self,diag):
		# Gives you back the relative analysis path to save analysis data or retrieve analysis data from
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

	def averageReferenceImages(self,diag, refRunDate,refRunName):
		# Average the images of a particular diagnostic
		baseDataFolder = self.baseDataFolder

		dataStyle, shots, diagShotDict = runsOrBursts(runDate,runName,diagList)
		refPath = os.path.join(baseDataFolder,diag,refRunDate,refRunName)
		if dataStyle == 'burst':
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

	def logThatShit(self, string2Log):
		'''Need to open logger file, append the string and close it'''
		#logging.basicConfig(filename=self.loggerFile, filemode='a', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S',level=logging.INFO)
		#logger.info(string2Log)
		##logger = logging.getLogger(self.loggerFile )
		#logging.shutdown()
		loggerFile = self.loggerFile
		f = open(loggerFile, "a")
		now = datetime. now()
		current_time = now. strftime('%d/%m/%Y %H:%M:%S')	
		f.write('\n' + current_time + ' - ' + string2Log)
		f.close()


		return 0

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
		self.logThatShit('Saving Run Object\n\n')
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
		print ("dateRunString to search for: ", dateRunString)

		# Now open the csv file and find the row in which the first column entry matches the dateRunString
		calibrationFolder = Path(self.calibrationFolder)
		calibrationPath = calibrationFolder / 'CalibrationPaths.csv'
		''' CIDU
		The code using the csv reader has been failing.
		Trying using pandas
		'''
		import pandas as pd

		def db_index(db, dateRunString, diag):
		    # Located the region in the database that corresponds to the correct run and diagnostic
		    runCalFiles = db[db[' '] == dateRunString]
		    keys = db.keys().tolist()
		#     print (runCalFiles)
			# Find the index of the run
		    inds = db.index[db[' '] == dateRunString].tolist()[0]
		    for i, k in enumerate(keys):
		    	# Find the index of the diag (column)
		        if k == diag:
		            keyIndex = i
			# print ("Database Index")
			# print(inds, keyIndex)		            
		    return inds, keyIndex

		db = pd.read_csv(calibrationPath)
		inds, keyIndex = db_index(db, dateRunString, diag)
		
		calibrationFilePath = str(db.iloc[inds, keyIndex])
		print ('The file path')
		print (calibrationFilePath)
		
		fileNameComponents = calibrationFilePath.split("\\")


		print("The cal file path from CSV", calibrationFilePath)
		print("The components of the calibration file path", fileNameComponents)
		# Make the file path with pathlib, all OS system compatibility
		calibrationFilePath = Path(fileNameComponents[0])
		if len(fileNameComponents) > 1:
			for p in fileNameComponents[1:]:
				calibrationFilePath = calibrationFilePath / p
		
		# Now Load in the data 
		calibrationFilePath = Path(calibrationFolder, calibrationFilePath)
		try:
			calibData = np.load(calibrationFilePath,allow_pickle=True)
		except:
			print('Info Recieved')
			print(calibrationFilePath)
			raise Exception('COULD NOT LOAD CALIBRATION FILE, PLEASE CHECK IT EXISTS AND THE PATH IN calibrationPaths.csv IS CORRECT')


		return calibData








