# DataRun
This class is used to call all the analysis tools.
It can be used for two things: 

1. Initial extraction 
2. Loading the data

Below are the functions and their input flags.

## DIAGNOSTIC FUNCTION CALLS
* performProbeDensityAnalysis()
	* overwrite = True
	* verbose = False
	* visualise = False
	* Debugging = False
* performESpecAnalysis()
	* 	useCalibration=True
	*  overwriteData=False
*  performHASOAnalysis()
	*  useChamberCalibration=True
*  performXRayAnalysis()
	*  	justGetCounts=False
*  performSPIDERAnalysis()
*  performPreCompNFAnalysis()

## LOADING DIAGNOSTICS
* loadGasSetPressure()
* loadGasCellLength()
* loadHASOFocusData()
	* fileOption_0_1 = 1
* loadPulseDuration()
	* getShots=False
	* removeDuds=False
* loadAnalysedSpecPhase()
	* getShots=False
	* removeDuds=False
* loadLaserEnergy()
	* getShots=False
	* removeDuds=False
* loadAnalysedXRayCountsData()
	* getShots=False
* load_Averaged_ESpecData()
	
