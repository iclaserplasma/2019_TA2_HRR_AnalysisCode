# Set of functions to load analysed data
import numpy as np
import os



def pullAnalysedXRayCountsData(baseAnalysisFolder,runDate,runName,getShots=False):
    # General function to pull all of the XRay data from the analysis folder
    # It will automatically sort the data
    # It will pull data averaged by burst, with the std deviation
    # unless getShots is True, in which case, it'll pull all shots individually
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
    
    
def getLaserEnergy(baseAnalysisFolder,runDate,runName,getShots=False):
    # General function to pull all of the XRay data from the analysis folder
    # It will automatically sort the data
    # It will pull data averaged by burst, with the std deviation
    # unless getShots is True, in which case, it'll pull all shots individually
    diag='PreCompNF'
    
    analysedDataDir = os.path.join(baseAnalysisFolder,diag,runDate,runName)
    bursts = [f for f in os.listdir(analysedDataDir) if not f.startswith('.')]
    
    if getShots:
        imgCounts = []
        shotID = []
        for burst in bursts:
            shots = os.listdir(os.path.join(analysedDataDir,burst))
            for shot in shots:
                imgCounts.append(np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True)[0].astype(float))
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
                tmpImgCounts.append(np.load(os.path.join(analysedDataDir,burst,shot),allow_pickle=True)[0])
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