# dataRun.py
# 

# Authors:          Rob Shalloo, Matt Streeter & Nic Gruse
# Affiliation:      Imperial College London
# Creation Date:    2020


import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import logging
import csv 
import glob
# Works with file paths on all operating systems
from pathlib import Path

try:
	import cPickle as pickle # only for python2
except ModuleNotFoundError:
	import pickle
from datetime import datetime








class dataRun:


	# -------------------------------------------------------------------------------------------------------------------
	# -----										HOUSEKEEPING AND ADMIN	 											-----
	# -------------------------------------------------------------------------------------------------------------------


	def __init__(self, baseDataFolder, baseAnalysisFolder, calibrationFolder, runDate, runName,verbose=False,overwrite=False):
		""" 
		Initializes the run object, stores key information pertinant to the run and starts logging the analysis actions

		Args:
		baseDataFolder (string): 		location of the MIRAGE Data folder (folder containing the raw data for the whole 
										experiment)
		baseAnalysisFolder (string): 	location of the folder in which the analysed data is to be saved. This folder has 
										an identical structure to that of the MIRAGE folder
		calibrationFolder (string): 	Folder in which calibrations for each of the diagnostics is stored. This folder 
										also has an indentical structure to the MIRAGE folder
		runDate (string): 				date ('YYYYMDD') on which the run was taken. runs after midnight are still labelled 
										as the previous day
		runName (string):				A string identifiying the specific run. Can be anything, although ususally is something
										sensible like 'run015'
		verbose	(bool):					verbose output, 1 for verbose output (printed to console)
		overwrite (bool):				To overwrite logger file and create new run object or not overwrite and load old run
		"""
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
		'''
		 By following through the folder system, this will tell us what the diagnostics were for the run
		'''
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
		analysisPath,isItReal = self.getDiagAnalysisPath('General')
		gasCellCsvFilePath = os.path.join(analysisPath,'GasCell.csv')
		runName = self.runName
		runDate = self.runDate
		if os.path.isfile(gasCellCsvFilePath):
			gasCell_df = pd.read_csv(gasCellCsvFilePath)
		else:
			try:
				from .sqlDatabase import connectToSQL
			except:
				from sqlDatabase import connectToSQL

			
			db = connectToSQL(True)
			keys = ['run','shot_or_burst','GasCellPressure','GasCellLength']

			# Get all data pertaining to the run
			cursor = db.cursor()
			command = "SELECT " + '%s,%s,%s,%s' % tuple(keys) + ' FROM shot_data'

			cursor.execute(command)
			SQLData = cursor.fetchall() ## it returns list of tables present in the database
			gasCell_df = pd.DataFrame(SQLData,columns=keys)
			gasCell_df.to_csv(gasCellCsvFilePath,index=False)
		

		## showing all the tables one by one
		runSel = gasCell_df['run']==runName
		gasCellPressure = gasCell_df[runSel]['GasCellPressure'].values
		gasCellLength = gasCell_df[runSel]['GasCellLength'].values

		
		self.saveData(os.path.join(analysisPath,'GasCellPressure'),gasCellPressure)
		self.saveData(os.path.join(analysisPath,'gasCellLength'),gasCellLength)

		


	# -------------------------------------------------------------------------------------------------------------------
	# -----										DIAGNOSTIC FUNCTION CALLS 											-----
	# -------------------------------------------------------------------------------------------------------------------

	# PROBE ANALYSIS
	def performProbeDensityAnalysis(self, overwrite = True, verbose = False, visualise = False, 
				Debugging = False):
		try:
			from . import probe_density_extraction
		except:
			import probe_density_extraction
		
		diag = 'Probe_Interferometry'
		print ("In ", diag)

		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		if not pathExists:
			os.makedirs(analysisPath)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		print (analysisPath, pathExists)
		probeCalib = self.loadCalibrationData(diag)
		folders = list(filePathDict)
		# print(folders)
		if Debugging:
			folders = folders[:1]
		for burstStr in folders:		
			
			analysisSavePath = os.path.join(analysisPath,burstStr,'{}_Analysis'.format(diag))
			fileExists = os.path.isfile(analysisSavePath + '.npy')
			print ("analysisSavePath: ", analysisSavePath)
			print (fileExists)

			if (overwrite == False and fileExists):
				print ("File exists so not overwriting")
			else:
				analysedData = probe_density_extraction.extract_plasma_density(
								filePathDict[burstStr],probeCalib, analysisSavePath, 
								verbose = verbose, visualise = visualise)			
				self.saveData(analysisSavePath,analysedData)

				self.logThatShit('Performed Probe_Interferometry Analysis for ' + burstStr)
				print('Analysed Probe_Interferometry '+ burstStr)

	# ELECTRON ANALYSIS 
	def performESpecAnalysis(self,useCalibration=True):
		# Load the espec images for the run and analyse
		# if it exists get it, if not, run initESpecAnalysis and update log
		try:
			from . import ESpecAnalysis
		except:
			import ESpecAnalysis
		
		diag = 'HighESpec'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		
		if useCalibration:
			eSpecCalib = self.loadCalibrationData(diag)
		# NEED TO PUT ALTERNATIVE CALIBRATION TUPLE HERE TO HELP NICS CODE RUN

		for burstStr in filePathDict.keys():		
			if useCalibration:
				for shotFile in filePathDict[burstStr]:
					shotID = shotFile.split('\\')[-1].split('.')[0]
					analysedData = ESpecAnalysis.ESpecSCEC_individual(shotFile,eSpecCalib)
					# Save the data
					analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecAnalysis_'+shotID)
					self.saveData(analysisSavePath,analysedData)

			else:
				for shotFile in filePathDict[burstStr]:
					shotID = shotFile.split('\\')[-1].split('.')[0]
					analysedData = ESpecAnalysis.ESpecSCEC_individual(shotFile)
					# Save the data
					analysisSavePath = os.path.join(analysisPath,burstStr,'ESpecAnalysis_NoCalibration_'+shotID)
					self.saveData(analysisSavePath,analysedData)
			self.logThatShit('Performed HighESpec Analysis for ' + burstStr)
			print('Analysed ESpec Spectrum, Charge, totalEnergy, cutoffEnergy95 for '+ burstStr)

	# SPIDER ANALYSIS CALLS 
	def performSPIDERAnalysis(self):
		try:
			from . import SPIDERAnalysis
		except:
			import SPIDERAnalysis
		
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
	def performHASOAnalysis(self,useChamberCalibration=True,getIndividualShots=False,overwriteAnalysis=True):
		try:
			from . import HASOAnalysis
		except:
			import HASOAnalysis
			
		diag = 'HASO'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)
		
		if useChamberCalibration:
			zernikeOffsets = self.loadCalibrationData(diag)
		
		for burstStr in filePathDict.keys():		
			if useChamberCalibration:
				if getIndividualShots:
					for shotFile in filePathDict[burstStr]:
						# Check if analysis already exists:
						shotID = shotFile.split('\\')[-1].split('.')[0]
						fileCheck = os.path.exists(os.path.join(analysisPath,burstStr,shotID+'.npy'))
						if fileCheck and not overwriteAnalysis:
							print(burstStr + ' ' + shotID +': Already Analysed')
						else:
							analysedData = HASOAnalysis.extractCalibratedWavefrontInfo(shotFile,zernikeOffsets)
							# Save the data
							analysisSavePath = os.path.join(analysisPath,burstStr,shotID)
							self.saveData(analysisSavePath,analysedData)
				else:
					analysedData = HASOAnalysis.extractCalibratedWavefrontInfo(filePathDict[burstStr],zernikeOffsets)
					# Save the data
					analysisSavePath = os.path.join(analysisPath,burstStr,'calibratedWavefront')
					self.saveData(analysisSavePath,analysedData)

			else:
				if getIndividualShots:
					for shotFile in filePathDict[burstStr]:
						# Check if analysis already exists:
						shotID = shotFile.split('\\')[-1].split('.')[0]
						fileCheck = os.path.exists(os.path.join(analysisPath,burstStr,shotID+'.npy'))
						if fileCheck and not overwriteAnalysis:
							print(burstStr + ' ' + shotID +': Already Analysed')
						else:
							analysedData = HASOAnalysis.extractCalibratedWavefrontInfo(shotFile)
							# Save the data
							analysisSavePath = os.path.join(analysisPath,burstStr,shotID)
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
		try:
			from . import PreCompNFAnalysis
		except:
			import PreCompNFAnalysis
		
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
		try:
			from . import XRayAnalysis
		except:
			import XRayAnalysis
		
		diag = 'XRay'
		filePathDict = self.createRunPathLists(diag)
		analysisPath, pathExists = self.getDiagAnalysisPath(diag)

		if justGetCounts==False:
			xrayCalib = self.loadCalibrationData(diag)

		for burstStr in filePathDict.keys():
			if justGetCounts:
				# Here we're going to avoid all actual x-ray code and just sum the image counts
				# Not even using a background. This is to mimic what happened during optimization
				for filePath in filePathDict[burstStr]:	
					imgCounts = np.sum(plt.imread(filePath).astype(float))
					shotName = filePath.split('\\')[-1][:-4]
					analysisSavePath = os.path.join(analysisPath,burstStr,'XRayImgCounts_'+shotName)
					self.saveData(analysisSavePath,imgCounts)
					self.logThatShit('Saved counts for ' + burstStr + ' ' + shotName)
					print('Saved counts for ' + burstStr + ' ' + shotName )
				
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


	def createDataBuckets(self,arr):
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

	# -------------------------------------------------------------------------------------------------------------------
	# -----								LOADING ANALYSED DATA FUNCTION CALLS										-----
	# -------------------------------------------------------------------------------------------------------------------

	def loadESpecData(self):
		# General function to pull in the electron spectrum data
		# Each shot has 5 data points saved:
		# WarpedImageWithoutBckgnd, Spectrum, Charge, totalEnergy, cutoffEnergy95
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName
		diag='HighESpec'	

		def returnAverageESpecPerBurst(data):
			ave = []
			std = []
			
			for i in range(np.shape(data)[1]):
				try:
					a = np.average( np.array(data[:,i]))
					s = np.std(     np.array(data[:,i]))
				except AttributeError:            
					a = np.average( np.array(data[:,i], dtype = float))
					s = np.std(     np.array(data[:,i], dtype = float))
				ave.append( a )
				std.append( s )
			return ave, std

		eSpecCalib = self.loadCalibrationData(diag)
		_, _, _ , E, dxoverdE, _, L, CutOff, _ = eSpecCalib
		Energy = E[CutOff:]
		# the following parameters will be cut only if they are intended to be used.
		#    Length = L[:, CutOff:]
		dxoverdE = dxoverdE[CutOff:]  # MIGHT BE WRONG		
		calData = (Energy, dxoverdE, L)

		print ("Loading the data from {}".format(diag))
		analysedDataDir = os.path.join(baseAnalysisFolder,diag,runDate,runName)
		bursts = [f for f in os.listdir(analysedDataDir) if not f.startswith('.')]
		print (analysedDataDir, "\nBurst, Number of shots in burst")

		runOutputEspec = {}
		for burst in bursts:
			analysedFiles = os.listdir(os.path.join(analysedDataDir,burst))
			if 'ESpecAnalysis.npy' in analysedFiles:
				filePath = os.path.join(analysedDataDir, burst, 'ESpecAnalysis.npy')
				d = np.load( filePath, allow_pickle = True)
				print (burst, len(d))

				runOutputEspec[burst] = returnAverageESpecPerBurst(d)
		BurstIDS = list(runOutputEspec)
		eDataBurst = []
		for burst in BurstIDS:
			eDataBurst.append(runOutputEspec[burst])

		# eDataBurst contains the average and the std of the burst data
		return BurstIDS, eDataBurst, calData


	def loadAnalysedXRayCountsData(self,getShots=False):
		# General function to pull all of the XRay data from the analysis folder
		# It will automatically sort the data
		# It will pull data averaged by burst, with the std deviation
		# unless getShots is True, in which case, it'll pull all shots individually
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag='XRay'
		
		analysedDataDir = os.path.join(baseAnalysisFolder,diag,runDate,runName)
		bursts = [f for f in os.listdir(analysedDataDir) if not f.startswith('.')]
		
		if getShots:
			imgCounts = []
			shotID = []
			for burst in bursts:
				shots = os.listdir(os.path.join(analysedDataDir,burst))
				for shot in shots:
					imgCounts.append(np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True).astype(float))
					for elem in shot.replace('.','_').split('_'):
						if 'Shot' in elem:
							shotName = elem
					shotID.append((burst+shotName))
		else:
			imgCounts = []
			shotID = []
			for burst in bursts:
				tmpImgCounts = []
				shots = os.listdir(os.path.join(analysedDataDir,burst))
				for shot in shots:
					tmpImgCounts.append(np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True))
				shotID.append(burst)
				imgCounts.append((np.mean(tmpImgCounts),np.std(tmpImgCounts)))
		
		# NOW SORT THE DATA
		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]
				# Append them to lists
				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			imgCounts_sorted = np.asarray(imgCounts)[indxOrder]
		else:
		# bursts
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			imgCounts_sorted = np.asarray(imgCounts)[indxOrder]
		
		return shotID_sorted,imgCounts_sorted        

	def loadLaserEnergy(self,getShots=False,removeDuds=False):
		# General function to pull all of the XRay data from the analysis folder
		# It will automatically sort the data
		# It will pull data averaged by burst, with the std deviation
		# unless getShots is True, in which case, it'll pull all shots individually
		
		energyThreshold = 0.01 # 10 mJ
		
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName
		diag='PreCompNF'
		
		analysedDataDir = os.path.join(baseAnalysisFolder,diag,runDate,runName)
		bursts = [f for f in os.listdir(analysedDataDir) if not f.startswith('.')]
		
		if getShots:
			laserEnergy = []
			shotID = []
			for burst in bursts:
				shots = os.listdir(os.path.join(analysedDataDir,burst))
				for shot in shots:
					tmpLaserEnergy = np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True)[0].astype(float)
					if tmpLaserEnergy > energyThreshold or removeDuds is False: # greater than 10 mJ
						laserEnergy.append(tmpLaserEnergy)
						for elem in shot.replace('.','_').split('_'):
							if 'Shot' in elem:
								shotName = elem
						shotID.append((burst+shotName))
		else:
			laserEnergy = []
			shotID = []
			for burst in bursts:
				shotLaserEnergy = []
				shots = os.listdir(os.path.join(analysedDataDir,burst))
				for shot in shots:
					tmpLaserEnergy = np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True)[0]
					if tmpLaserEnergy > energyThreshold or removeDuds is False:
						shotLaserEnergy.append(tmpLaserEnergy)
				shotID.append(burst)
				laserEnergy.append((np.mean(shotLaserEnergy),np.std(shotLaserEnergy)))
		
		# NOW SORT THE DATA
		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]
				
				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			laserEnergy_sorted = np.asarray(laserEnergy)[indxOrder]
		else:
		# bursts
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			laserEnergy_sorted = np.asarray(laserEnergy)[indxOrder]
		
		return shotID_sorted,laserEnergy_sorted    


	def loadAnalysedSpecPhase(self, getShots=False,removeDuds=False):
		# Retrieves the spectral phase from the spider

		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag = 'SPIDER'
			
		runDir = os.path.join(baseAnalysisFolder , diag,  runDate , runName)
		bursts = [f for f in os.listdir(runDir) if not f.startswith('.')]
			

		if getShots:
			# Get individual shots
			GDD = []
			TOD = []
			FOD = []
			shotID = []
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]
				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					timeProfile, specPhaseOrders = np.load(shotPath,allow_pickle=True) 
					# Check for duds
					t,I = timeProfile
					indxs = np.argwhere(np.abs(t)>400) # places further than 400 fs from the middle
					testSum = np.sum(I[indxs])
					if testSum < 1 or removeDuds is False:
						for elem in shot.replace('.','_').split('_'):
							if 'Shot' in elem:
								shotName = elem
						shotID.append((burst+shotName))
						GDD.append(specPhaseOrders[0])
						TOD.append(specPhaseOrders[1])
						FOD.append(specPhaseOrders[2])
							
		else:
			GDD = []
			TOD = []
			FOD = []
			shotID = []
			
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]

				tmpGDD = []
				tmpTOD = []
				tmpFOD = []
				
				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					timeProfile, specPhaseOrders = np.load(shotPath,allow_pickle=True) 
				
					# Check for duds
					t,I = timeProfile
					indxs = np.argwhere(np.abs(t)>400) # places further than 400 fs from the middle
					testSum = np.sum(I[indxs])
					if testSum < 1 or removeDuds is False:
						tmpGDD.append(specPhaseOrders[0])
						tmpTOD.append(specPhaseOrders[1])
						tmpFOD.append(specPhaseOrders[2])
						
				shotID.append(burst)        
				GDD.append((np.mean(tmpGDD),np.std(tmpGDD)))
				TOD.append((np.mean(tmpTOD),np.std(tmpTOD)))
				FOD.append((np.mean(tmpFOD),np.std(tmpFOD)))

		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]

				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			GDD_sorted = np.asarray(GDD)[indxOrder] 
			TOD_sorted = np.asarray(TOD)[indxOrder] 
			FOD_sorted = np.asarray(FOD)[indxOrder] 
		else:
			# bursts
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			GDD_sorted = np.asarray(GDD)[indxOrder] 
			TOD_sorted = np.asarray(TOD)[indxOrder] 
			FOD_sorted = np.asarray(FOD)[indxOrder] 
				
		return shotID_sorted , GDD_sorted, TOD_sorted,FOD_sorted

	def loadPulseDuration(self, getShots=False,removeDuds=False):
		# Returns the FWHM Pulse duration as measured by the SPIDER
		
		def getPulseDuration(timeArr,I_t):		
			# Returns FWHM pulse duration of a laser temporal profile
			zeroTimeIndx = np.argmin(abs(timeArr))
			earlyI_t = I_t[0:zeroTimeIndx]
			earlytimeArr = timeArr[0:zeroTimeIndx]
			lateI_t = I_t[zeroTimeIndx:-1]
			latetimeArr = timeArr[zeroTimeIndx:-1]
			
			# quick and dirty function to find the FWHM of the intensity profile of the pulse
			t1 = earlytimeArr[np.argmin(  abs(earlyI_t - np.amax(I_t)/2)   ) ]
			t2 = latetimeArr[np.argmin(abs(lateI_t - np.amax(I_t)/2))]
			fwhm = t2-t1
							
			return fwhm

		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag = 'SPIDER'
			
		runDir = os.path.join(baseAnalysisFolder , diag,  runDate , runName)
		bursts = [f for f in os.listdir(runDir) if not f.startswith('.')]
			

		if getShots:
			# Get individual shots
			pulseDuration = []
			shotID = []
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]
				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					timeProfile, specPhaseOrders = np.load(shotPath,allow_pickle=True) 
					# Check for duds
					t,I = timeProfile
					indxs = np.argwhere(np.abs(t)>400) # places further than 400 fs from the middle
					testSum = np.sum(I[indxs])
					if testSum < 1 or removeDuds is False:
						for elem in shot.replace('.','_').split('_'):
							if 'Shot' in elem:
								shotName = elem
						shotID.append((burst+shotName))
						pulseDuration.append(getPulseDuration(t,I))
							
		else:
			pulseDuration = []
			shotID = []
			
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]

				tmpPulseDuration = []
				
				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					timeProfile, specPhaseOrders = np.load(shotPath,allow_pickle=True) 
				
					# Check for duds
					t,I = timeProfile
					indxs = np.argwhere(np.abs(t)>400) # places further than 400 fs from the middle
					testSum = np.sum(I[indxs])
					if testSum < 1 or removeDuds is False:
						tmpPulseDuration.append(getPulseDuration(t,I))
						
				shotID.append(burst)        
				pulseDuration.append((np.mean(tmpPulseDuration),np.std(tmpPulseDuration)))


		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]

				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			pulseDuration_sorted = np.asarray(pulseDuration)[indxOrder] 

		else:
			# bursts
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			pulseDuration_sorted = np.asarray(pulseDuration)[indxOrder] 

				
		return shotID_sorted , pulseDuration_sorted
		
	def loadPulseShape(self, getShots=False,removeDuds=False,tAxis=None):
		# Returns the Pulse shape as measured by the SPIDER
		

		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag = 'SPIDER'
			
		runDir = os.path.join(baseAnalysisFolder , diag,  runDate , runName)
		bursts = [f for f in os.listdir(runDir) if not f.startswith('.')]
		
		I_t_list = []

		# Get individual shots
		
		for burst in bursts:
			I_t_burst = []
			burstDir = os.path.join(runDir,burst)
			shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]
			for shot in shots:
				shotPath = os.path.join(burstDir,shot)
				timeProfile, specPhaseOrders = np.load(shotPath,allow_pickle=True) 
				# Check for duds
				t,I = timeProfile
				if (np.min(I)/np.max(I))<0.1:
					if tAxis is None:
						tAxis = t
					I = I/trapz(I,x=t)
					f_t = interp1d(t,I,kind='linear', 
						bounds_error=False, fill_value=0,assume_sorted=True)
					I_t_burst.append(f_t(tAxis))
			if getShots:
				I_t_list.append(I_t_burst)
			else:
				I_t_list.append(np.mean(I_t_burst,axis=0))
		return tAxis, I_t_list
			
	def loadGasSetPressure(self):
		# Retrieve the gas set pressure
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		dataPath= os.path.join(baseAnalysisFolder,'General',runDate,runName,'GasCellPressure.npy')
		tmpGasPressure = np.load(dataPath,allow_pickle=True)
		tmpGasPressure = tmpGasPressure.item()
		gasPressure = []
		shotID = []

		for i in range(len(tmpGasPressure)):
			gasPressure.append(tmpGasPressure[i+1])
			shotID.append('Burst'+str(i+1))
		
		burstNums = []
		for burst in shotID:
			burstNums.append(int(burst[5:]))
		indxOrder = np.argsort(burstNums)
		shotID_sorted = np.asarray(shotID)[indxOrder]
		gasPressure_sorted = np.asarray(gasPressure)[indxOrder] 

		return shotID_sorted, gasPressure_sorted

	def loadGasCellLength(self):
		# Retrieve the gas set pressure
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		dataPath= os.path.join(baseAnalysisFolder,'General',runDate,runName,'gasCellLength.npy')
		tmpGasCellLength = np.load(dataPath,allow_pickle=True)
		tmpGasCellLength = tmpGasCellLength.item()
		gasLength = []
		shotID = []

		for i in range(len(tmpGasCellLength)):
			gasLength.append(tmpGasCellLength[i+1])
			shotID.append('Burst'+str(i+1))
		
		burstNums = []
		for burst in shotID:
			burstNums.append(int(burst[5:]))
		indxOrder = np.argsort(burstNums)
		shotID_sorted = np.asarray(shotID)[indxOrder]
		gasLength_sorted = np.asarray(gasLength)[indxOrder] 

		return shotID_sorted, gasLength_sorted		

	def loadHASOFocusData(self, fileOption_0_1 = 1,getShots=False,returnFocusShift=False):
		# Perhaps here we want not only to load burst, but individual shots
		# We need to get adapt the earlier HASOAnalysis function above to do this.

		filenameoptions = ['calibratedWavefront.npy', 'waveFrontOnLeakage.npy']
		fileName = filenameoptions[fileOption_0_1]

		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag = 'HASO'
		
		runDir = os.path.join(baseAnalysisFolder , diag,  runDate , runName)
		bursts = [f for f in os.listdir(runDir) if not f.startswith('.')]
		
		if getShots:
			z4 = []
			shotID = []
			for burst in bursts:
				shots = glob.glob(os.path.join(runDir,burst,'Shot*'))
				for shot in shots:
					zernikes = np.load(shot,allow_pickle=True)
					z4.append(zernikes[4])
					shotName = shot.split('\\')[-1]
					for elem in shotName.replace('.','_').split('_'):
						if 'Shot' in elem:
							shotName = elem
					shotID.append((burst+shotName))

		else:
			# Should be a condition here to check if individual shot data exists,
			# such that we can provide an error

			# TO DO

			z4 = []
			shotID = []
			for burst in bursts:
				filename = os.path.join(runDir,burst, fileName)
				zernikes  = np.load(filename, allow_pickle = True)

				# z4.append(zernikes[-1]) This looks like a mistake!
				z4.append(zernikes[4])
				shotID.append(burst)
		

		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]
				
				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			z4_sorted = np.asarray(z4)[indxOrder]
		else:
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			z4_sorted = np.asarray(z4)[indxOrder] 

		if returnFocusShift:
			a4_to_RealSpace = 0.001169   # 1.169 mm per measured focal term
			direction = -1				# From Matts email "More z3 focuses harder" so, more focal term beings focus negative, towards parabola
			focusShift = direction*z4_sorted*a4_to_RealSpace
			return shotID_sorted, z4_sorted , focusShift
		else:
			return shotID_sorted, z4_sorted

	def loadAnalysedESpec(self,getShots=False):
		''' Retrieves the Analysed ESpec data

		Each analysed image has data stored in the form 
		WarpedImageWithoutBckgnd, Spectrum, Charge, totalEnergy, cutoffEnergy95,

		NOTE: getShots = False, does not work...need to use getShots=True
		'''
		
		baseAnalysisFolder = self.baseAnalysisFolder
		runDate = self.runDate
		runName = self.runName

		diag = 'HighESpec'
			
		runDir = os.path.join(baseAnalysisFolder , diag,  runDate , runName)
		bursts = [f for f in os.listdir(runDir) if not f.startswith('.')]
			

		if getShots:
			# Get individual shots
			Spectrum2D = []
			Spectrum1D = []
			Charge = []
			totalEnergy = []
			cutoffEnergy95 = []
			shotID = []
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]
				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					loadedData = np.load(shotPath,allow_pickle=True) 
					for elem in shot.replace('.','_').split('_'):
							if 'Shot' in elem:
								shotName = elem
					shotID.append((burst+shotName))
					Spectrum2D.append(loadedData[1])
					Spectrum1D.append(loadedData[2])
					Charge.append(loadedData[3])
					totalEnergy.append(loadedData[4])
					cutoffEnergy95.append(loadedData[5])
							
		else:
			Spectrum2D = []
			Spectrum1D = []
			Charge = []
			totalEnergy = []
			cutoffEnergy95 = []
			shotID = []
			
			for burst in bursts:
				burstDir = os.path.join(runDir,burst)
				shots = [f for f in os.listdir(burstDir) if not f.startswith('.')]

				tmpSpec2D = []
				tmpSpec1D = []
				tmpCharge = []
				tmpTotalEnergy = []
				tmpCutOffEnergy = []

				for shot in shots:
					shotPath = os.path.join(burstDir,shot)
					loadedData = np.load(shotPath,allow_pickle=True) 
				
					tmpSpec2D.append(loadedData[1])
					tmpSpec1D.append(loadedData[2])
					tmpCharge.append(loadedData[3])
					tmpTotalEnergy.append(loadedData[4])
					tmpCutOffEnergy.append(loadedData[5])
						
				shotID.append(burst)        
				Spectrum2D.append((np.mean(tmpSpec2D),np.std(tmpSpec2D)))
				Spectrum1D.append((np.mean(tmpSpec1D),np.std(tmpSpec1D)))
				Charge.append((np.mean(tmpCharge),np.std(tmpCharge)))
				totalEnergy.append((np.mean(tmpTotalEnergy),np.std(tmpTotalEnergy)))
				cutoffEnergy95.append((np.mean(tmpCutOffEnergy),np.std(tmpCutOffEnergy)))

		if 'Shot' in shotID[0]:
			# Shots
			burstNums = []
			shotNums = []
			for burstShot in shotID:
				# Split up burst and shot strings
				tmpSpl = burstShot.split('S')
				tmpSpl[1] = 'S'+tmpSpl[1]
				burst = tmpSpl[0]
				shot = tmpSpl[1]

				burstNums.append(int(burst[5:]))
				shotNums.append(int(shot[4:]))
			orderVal = []
			maxShotsInBurst = np.amax(shotNums)
			for i in range(len(burstNums)):
				orderVal.append((burstNums[i]-1)*maxShotsInBurst+shotNums[i])
			indxOrder = np.argsort(orderVal)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			Spectrum2D_sorted = np.asarray(Spectrum2D)[indxOrder] 
			Spectrum1D_sorted = np.asarray(Spectrum1D)[indxOrder] 
			Charge_sorted = np.asarray(Charge)[indxOrder]
			totalEnergy_sorted = np.asarray(totalEnergy)[indxOrder] 
			cutoffEnergy95_sorted = np.asarray(cutoffEnergy95)[indxOrder] 

		else:
			# bursts
			burstNums = []
			for burst in shotID:
				burstNums.append(int(burst[5:]))
			indxOrder = np.argsort(burstNums)
			shotID_sorted = np.asarray(shotID)[indxOrder]
			Spectrum2D_sorted = np.asarray(Spectrum2D)[indxOrder] 
			Spectrum1D_sorted = np.asarray(Spectrum1D)[indxOrder] 
			Charge_sorted = np.asarray(Charge)[indxOrder]
			totalEnergy_sorted = np.asarray(totalEnergy)[indxOrder] 
			cutoffEnergy95_sorted = np.asarray(cutoffEnergy95)[indxOrder] 
		
		EnergyAxis = loadedData[0]

		return shotID_sorted , EnergyAxis, Spectrum2D_sorted, Spectrum1D_sorted,Charge_sorted,totalEnergy_sorted,cutoffEnergy95_sorted
		


# HELPER FUNCTIONS 
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