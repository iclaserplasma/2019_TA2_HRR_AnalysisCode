#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 06.01.2020
ESpec
@author: Gruse
v0.1 - the code was constructed mainly by the code produce during the experiment. It will be simplified and debugged.
"""
import os
import scipy.io
from scipy.interpolate import interp2d
import numpy as np
import cv2
import csv
import re

'''
Start the new code here ->
'''


def simpleFolderFinder(runPath):
    #  This function finds the first folder containing images (becuase, some are single shot and some are bursts)
    Filetype = ('.tif', '.tiff', '.TIFF', '.TIF')
    filesInPath = os.listdir(runPath)
    flag = 0
    subdirectoryForLater = 0
    for files in filesInPath:
        if os.path.isdir(os.path.join(runPath, files)) and subdirectoryForLater == 0:
            subdirectoryForLater = files
        for endings in Filetype:
            if files.endswith(endings):
                flag = 1
    if flag == 0:
        if subdirectoryForLater is not 0:
            newPath = os.path.join(runPath, subdirectoryForLater)
            runPath = simpleFolderFinder(newPath)
        else:
            runPath = 0
    return runPath


def TupleOfFiles(path="", Filetype=('.tif', '.tiff', '.TIFF', '.TIF')):
    if not isinstance(Filetype, list):
        Filetype = list(Filetype)
    if len(path) == 0:
        path = "."
    FileList = []
    for files in os.listdir(path):
        for endings in Filetype:
            if files.endswith(endings):
                FileList.append(os.path.join(path, files))
    def shotNr(name):
        return int(re.findall(r'\d+', name)[-1])
    FileList.sort(key=shotNr)
    return FileList


def ImportImageFiles(FileList):
    from PIL import Image
    im = Image.open(FileList[0])
    imarray = np.array(im)
    if imarray.dtype == np.uint16:
        # 12-bit data in a 16-bit image
        # The format is 0x8XXX
        # So just mask out the top nybble and we're good to go
        imarray &= 0xfff
    Images = np.zeros(shape=(imarray.shape[0], imarray.shape[1], len(FileList)))
    for i in range(0, len(FileList)):
        im = Image.open(FileList[i])
        imarray = np.array(im)
        if imarray.dtype == np.uint16:
            # 12-bit data in a 16-bit image
            # The format is 0x8XXX
            # So just mask out the top nybble and we're good to go
            imarray &= 0xfff
        Images[:, :, i] = imarray
    return Images


def ESpecSCEC(FileList, calibrationTuple):
    """
    This function is supposed to be called from runData. It takes the list of file(path)s and the calibration tuple.
    :param FileList:
    :param calibrationTuple:
    :return: List of analysed tuples. Each tuple (corresponding to
    """
    images = ImportImageFiles(FileList)
    analysedData = []
    for i in range(0, images.shape[2]):
        analysedImage = analyseImage(images[:, :, i], calibrationTuple)
        analysedData.append(analysedImage)
    return analysedData

def ESpecSCEC_individual(FilePath, calibrationTuple):
    """
    This function is supposed to be called from runData. It takes the a file path and the calibration tuple.
    :param FilePath:
    :param calibrationTuple:
    :return: analysed tuples
    """
    from PIL import Image
    im = Image.open(FilePath)
    image = np.array(im)
    if image.dtype == np.uint16:
        # 12-bit data in a 16-bit image
        # The format is 0x8XXX
        # So just mask out the top nybble and we're good to go
        image &= 0xfff
    analysedData = analyseImage(image, calibrationTuple)
    return analysedData

def analyseImage(rawImage, calibrationTuple):
    """
    This function is the main analysis function. It takes an image together with the calibration parameter to calculate
    various information from the high energy electrons spectrometer.
    :param rawImage: image matrix as doubles.
    :param calibrationTupel: (J, W, pts, E, dxoverdE, BackgroundImage, L, CutOff, BackgroundNoise) as a tupel
    Explicit information about the calibrationTupel see function 'createNewCalibrationFiles'
    :return:
    WarpedImageWithoutBckgnd - The warped image ( should be in fC/pixel)
    E
    Energy scale of the spectrum (differs from shot to shot). Is a list containing the main energy scale, and the top and bottom limts:
    [E average, [E top, E bottom]]
    Spectrum - spectra in the similar style as energy [Spectrum average, [Spectrum top, Spectrum bottom]]
    Divergence - divergence of the beam in mrad [divergence, standard error of the divergence]
    Charge - charge of the electron beam in fC [charge, standard error of the charge]
    totalEnergy - integrated energy of the electron beam in J [average energy, [upper limit, lower limit]]
    cutOffEnergy95 - cut off energy at 95% charge of the electron beam in MeV, [upper limit, lower limit]]
    imagedEdOmega - this is the warped image without background in fC / MeV / mrad, which is the first output argument divided by
        the energy width per pixel and divided by the steradiant per pixel (latter the width of the pixel / propagation length)
    """
    _, _, _, _, L, _, BackgroundNoise, _ = calibrationTuple
    # preparation of the calibration parameter (later move into the calibration prep above)
    # The cut off is an index, which indicates, where the calibration between image plate and camera image did not work.
    # This was due to a camera artifact (an imaging problem with the perplex glass)
    # the following parameters will be cut only if they are intended to be used.
    #    Length = L[:, CutOff:]

    # here hard coded: the analysis of the image plates yield to a count to fC calibration of:
    fCperCounts = 2.7e-3
    fCperCountsSigma = 6.5e-4

    # now to the analysis:
    image = rawImage.astype(float)
    WarpedImageWithoutBckgnd = imageTransformation(image, calibrationTuple)
    WarpedImageWithoutBckgnd = WarpedImageWithoutBckgnd * fCperCounts
    ChargeAv = np.sum(WarpedImageWithoutBckgnd)
    ChargeStd = np.sum(WarpedImageWithoutBckgnd)/fCperCounts*fCperCountsSigma
    Charge = [ChargeAv, ChargeStd]
    # correctly integrating the spectrum and change the units from counts/mm -> counts/MeV
    Spectrum, Divergence, E, dxoverdE, dOmega = getDivergenceAndEnergySpectrum(WarpedImageWithoutBckgnd, calibrationTuple)
    dEdOmega = np.tile(np.mean( np.diff( L ) )/dxoverdE[0], (WarpedImageWithoutBckgnd.shape[0], 1)) * dOmega
    imagedEdOmega = WarpedImageWithoutBckgnd / dEdOmega
    cutOffEnergy95 = dataDistributer(determine95percentCharge, E, Spectrum, dxoverdE, BackgroundNoise, L)
    totalEnergy = dataDistributer(determineTotalEnergy, E, Spectrum, dxoverdE, BackgroundNoise, L)
    return WarpedImageWithoutBckgnd, E, Spectrum, Divergence, Charge, totalEnergy, cutOffEnergy95, imagedEdOmega


def findFirstLastTruth(AboveFWHM):
    j=0
    while AboveFWHM[j] == 0:
        j += 1
    first = j
    j=1
    while AboveFWHM[-j] == 0:
        j += 1
    last = j
    return first, last

def FWHMCalculator(CroppedImage, W):
    FWHM = np.zeros(CroppedImage.shape[1])
    for i in range(0, CroppedImage.shape[1]):
        LineOut = CroppedImage[:, i]
        if LineOut.max() == 0:
            FWHM[i] = 0
            continue
        LineOut = LineOut/LineOut.max()
        AboveFWHM = LineOut >= .5
        if any(AboveFWHM):
            first, last = findFirstLastTruth(AboveFWHM)
            FWHM[i] = W[-last] - W[first]
        else:
            FWHM[i] = 0
    return FWHM


def getDivergenceAndEnergySpectrum(image, calibrationTuple):
    _, W, _, _, L, CutOff, BackgroundNoise, EnergyDivergenceTuple = calibrationTuple
    EAv, dxoverdEAv, EPos, dxoverdEPos, ENeg, dxoverdENeg, Div, PropL, ErrPropL, Cell2Magnet = EnergyDivergenceTuple
    Cell2Magnet = Cell2Magnet[0][0]
    #  W in mm
    CroppedImage = image
    CroppedImage[CroppedImage < CroppedImage.max()*.05] = 0
    TmpSumCounts = np.sum(CroppedImage, 0)
    TmpSpectrum = np.multiply(TmpSumCounts, dxoverdEAv[0,:]) / np.mean(np.diff(L))  
    WMatrix = np.reshape( np.repeat(W, CroppedImage.shape[1]), CroppedImage.shape )
    normFac = np.sum(CroppedImage, axis=0)
    normFac[normFac <= normFac.max()*0.01] = normFac.max()*0.01
    CMS = np.sum( WMatrix * CroppedImage, axis=0) / normFac
    # the total length of flight:
    TotalLengthOfFlight = np.sqrt( PropL**2  + (CMS*1e-3)**2 )
    dOmega = np.mean(np.diff(W)) / ( np.sqrt(  np.tile(PropL**2, (image.shape[0], 1)) + np.tile((W*1e-3)**2, (image.shape[1], 1)).transpose() ) )
    #  the FWHM for each length (0 if it does not fulfill the criteria)
    FWHM = FWHMCalculator(CroppedImage, W)
    # and finally the divergence for each length and then averaged with a weightening on the signal strength:
    RawDivergence = FWHM/TotalLengthOfFlight
    AveragedDivergence = np.sum( RawDivergence*TmpSpectrum/TmpSpectrum.sum() )
    # error propagation:
    StdDivergence = np.sqrt( np.sum( (AveragedDivergence - RawDivergence)**2 * TmpSpectrum/TmpSpectrum.sum() ) )
    # this is def a high error and the error on the propagation length can be ignored in respect to this
    # Now energy time! yay...
    FWHMAtMagnet = AveragedDivergence * Cell2Magnet * 1e-3
    relEnergyScaleID = np.argmin( abs( FWHMAtMagnet - Div ) )
    E = [EAv[relEnergyScaleID, :], [EPos[relEnergyScaleID, :], ENeg[relEnergyScaleID, :]]]
    dxoverdE = [dxoverdEAv[relEnergyScaleID, :], [dxoverdEPos[relEnergyScaleID, :], dxoverdENeg[relEnergyScaleID, :]]]
    SpectrumAv = electronSpectrum(image, dxoverdE[0], L)
    SpectrumPos = electronSpectrum(image, dxoverdE[1][0], L)
    SpectrumNeg = electronSpectrum(image, dxoverdE[1][1], L)
    Spectrum = [SpectrumAv, [SpectrumPos, SpectrumNeg]]
    return Spectrum, [AveragedDivergence, StdDivergence], E, dxoverdE, dOmega


def dataDistributer(Fcn, E, Spectrum, dxoverdE, BackgroundNoise, L):
    def subroutine(Fcn, E, Spectrum, dxoverdE, BackgroundNoise, L):
        SignificanceLevel = 1
        TmpSumCounts = np.sum(BackgroundNoise, axis=0)/np.sqrt(BackgroundNoise.shape[1]-1)
        BackgroundStd_SigmaLevel = np.multiply(TmpSumCounts, dxoverdE) / np.mean(np.diff(L)) * SignificanceLevel
        return Fcn(E, Spectrum, BackgroundStd_SigmaLevel)
    
    Av = subroutine(Fcn, E[0], Spectrum[0], dxoverdE[0], BackgroundNoise, L)
    Pos = subroutine(Fcn, E[1][0], Spectrum[1][0], dxoverdE[1][0], BackgroundNoise, L)
    Neg = subroutine(Fcn, E[1][1], Spectrum[1][1], dxoverdE[1][1], BackgroundNoise, L)
    return [Av, [Pos, Neg]]



def imageTransformation(image, calibrationTuple):
    J, _, pts, BackgroundImage, _, CutOff, _, _ = calibrationTuple # this is updated to the new calibration tuple
    WarpedImage, __ = four_point_transform(image, pts, J)  # warp the image
    WarpedImage = np.fliplr(WarpedImage)  # the axis is flipped (high and low energy)
    WarpedImageWithoutBckgnd = WarpedImage - BackgroundImage  # background subtraction
    WarpedImageWithoutBckgnd = WarpedImageWithoutBckgnd[:, CutOff:]  # as mentioned above, the low energy part needs to be removed due artifacts
    return WarpedImageWithoutBckgnd


def electronSpectrum(Image, dxoverdE, L):
    """
    The sum of the counts transversally will result in a unit of counts. Dividing this by the area of a pixel will result in MeV/mm in this case.
    We then change the variables into Counts/MeV by multiplying with dx/dE.
    Output is then Counts/MeV.
    """
    SumCounts = np.sum(Image, 0)
    Spectrum = np.multiply(SumCounts, dxoverdE) / np.mean(np.diff(L))
    return Spectrum


def determine95percentCharge(E, Spectrum, BckStd_SigmaLevel):
    """
    The Background is subtracted from the spectrum thus the noise level equals zero.
    However the std of the noise should be determined and then a sigma level of that should be set know where to ognore the charge.
    Then the cummulative trapz integral is calculated to determine where the cut off is by looking where 95% of the charge is.
    To test this and plot this use:

    plt.plot(E, np.append(np.array([True]),np.array([CumTrapzMinus1 < 0.95])))
    plt.plot(E[:-1], CumTrapzMinus1)
    plt.plot(E, TmpSpectrum/np.max(TmpSpectrum))
    plt.legend(('Within 95 %','Cummulative Integration','Spectrum'))
    plt.xlabel('Energy [MeV]')

    Note: BckStd_SigmaLevel has the dimension (num,) which means that Spectrum, having the dimension (num,1) needs to be taken
    """
    TmpSpectrum = np.zeros(Spectrum.shape)
    Mask = np.greater(Spectrum, BckStd_SigmaLevel)
    if any(Mask) is True:
        TmpSpectrum[Mask] = Spectrum[Mask]
        TmpSpectrum = TmpSpectrum / np.trapz(TmpSpectrum, E)
        CumTrapzMinus1IndexPrep = (TmpSpectrum[:-1] + TmpSpectrum[1:]) / 2 * np.diff(E, axis=0)
        CumTrapzMinus1 = np.cumsum(CumTrapzMinus1IndexPrep)
        MaskCumSum = np.append(np.array([True]), np.array([CumTrapzMinus1 < 0.95]))
        MaximumEnergy = np.amax(E[MaskCumSum])
    else:
        MaximumEnergy = 0
    return MaximumEnergy


def determineTotalEnergy(E, Spectrum, BckStd_SigmaLevel):
    """
    The Background is subtracted from the spectrum thus the noise level equals zero.
    However the std of the noise should be determined and then a sigma level of that should be set to estimate the high
    energy cut off.
    Note: BckStd_SigmaLevel has the dimension (num,) which means that Spectrum, having the dimension (num,1) needs to be taken
    """
    Mask = np.greater(Spectrum, BckStd_SigmaLevel)
    if any(Mask) is True:
        TotalEnergy = np.trapz(np.multiply(np.multiply(E, Mask), Spectrum), E)
    else:
        TotalEnergy = 0
    return TotalEnergy


'''
Below will be the code for creating the calibration file(s).
For now though, the calibration files have been created during the experiment and it seems that they are reasonable good
for this version of code (v0.1 20200106)
'''


def findCalibrationEntry(runName, calPath):
    csvFile = os.path.join(calPath, 'CalibrationPaths.csv')  # change to the correct name
    TotalFile = csv.reader(open(csvFile))
    entries = list(TotalFile)
    diagnostic = 'HighESpec'
    identifiedRun = 0
    identifiedDiag = 0
    for i in range(0, len(entries[0])):
        if entries[0][i] == diagnostic:
            identifiedDiag = i
    for j in range(0, len(entries)):
        if entries[j][0] == runName:
            identifiedRun = j
    if identifiedRun == 0 and identifiedDiag == 0:
        raise ValueError('The runName (%s) and diagnostic (%s) was found!' % (runName, 'HighESpec'))
    return entries, identifiedRun, identifiedDiag


def changeFileEntry(NewEntry, runName, calPath=r'Y:\\ProcessedCalibrations'):
    entries, identifiedRun, identifiedDiag = findCalibrationEntry(runName, calPath)
    oldEntry = entries[identifiedRun][identifiedDiag]
    if oldEntry == '':
        oldEntry = 'empty'
    entries[identifiedRun][identifiedDiag] = NewEntry
    csvFile = os.path.join(calPath, 'CalibrationPaths.csv')  # change to the correct name
    writer = csv.writer(open(csvFile, 'w', newline=''))
    writer.writerows(entries)
    print('Changed run: %s of diagnostic %s to: \'%s\'. Previously it was \'%s\'.' % (
        runName, 'HighESpec', NewEntry, oldEntry))
    return entries


def createNewCalibrationFiles(runName, basePath=r'Z:\\', calPath='Y:\\ProcessedCalibrations', BackgroundImage=0,
                              BackgroundNoise=0):
    """
    This function creates a calibration file for a specific run.
    :param runName: run name, which this calibration file will be for. Format: YYYYMMDD/run000
    :param basePath: The folder in which the folders 'MIRAGE' and 'Calibrations' can be found
    :param calPath: The folder in which the 'CalibrationPaths.csv' data base can be found.
    :param BackgroundImage: Certain days do not have any darkfields. If desired, one can load the background image from
    a previous day and have it as input of this function to change that.
    :param BackgroundNoise: see previous param
    Explanation of the calibration tupel:
    J is the Jacobian to compensate for warping the image flat.
    W is the width of the lanex screen (this is the divergence)
    pts are the points in the camera image, which are taken as edge points to flatten the camera image.
    E is the energy information along the lanex screen (length of the lanex is the energy axis)
    dxoverdE is to convert the lanex axis from mm into MeV
    BackgroundImage is the darkfield of the day the run was taken on. Its the average of 100 shots.
    L is the longitudinal information of the lanex screen. This is the energy axis, here in mm.
    CutOff contains an index of the longitudinal axis. The images obtained had artifacts of imaging the lanex screen and
    do not contain real information. Its most-likely from the perspex window.
    BackgroundNoise is the standard deviation from the 100 shots taken for the darkfields (see BackgroundImage)
    :return: saves the calibration tupel and changes the intry in the calibration database entry.
    """
    runPath = os.path.join(basePath, 'MIRAGE', 'HighESpec', runName)
    imageFolderPath = simpleFolderFinder(runPath)
    FileList = TupleOfFiles(imageFolderPath)
    testImage = ImportImageFiles([FileList[0]])
    if testImage.shape[0] < 1216:
        pts = np.array([[40, 659 - 328], [41, 380 - 328], [1905, 389 - 328],
                        [1907, 637 - 328]])  # this was after we cropped the image (> 13. September)
    else:
        pts = np.array(
            [[40, 659], [41, 380], [1905, 389], [1907, 637]])  # this was before we cropped the image (< 11.September)
    Length = 350 - 120
    Width = 30
    ScreenStart = 120
    CutOff = 300
    #  distinguish between two cases:
    #  1) the S and L CalibrationParameter
    WarpedImage, M = four_point_transform(testImage[:, :, 0], pts)
    WarpedImage = np.fliplr(WarpedImage)
    J, L = getJacobianAndSpatialCalibration(testImage[:, :, 0].shape, WarpedImage.shape, M, Length,
                                            ScreenStart)
    PixelWidth = WarpedImage.shape[0]
    W = np.arange(0, PixelWidth) - round(PixelWidth / 2)
    W = W / PixelWidth * Width
    #  Darkfields:
    #  The first section is to find the darkfields of the specific day. It might be that there were two sets of that day
    #  since the image size might have varied during the day.
    if BackgroundImage != 0:
        foundDarkfields = 1
    else:
        foundDarkfields = 0
    runDate = convertRunNameToDate(runName)
    backgroundPath = os.path.join(basePath, 'MIRAGE', 'HighESpec', '%d' % runDate, 'darkfield01')
    if not os.path.exists(backgroundPath):
        print('WARNING E1: No darkfields were found. The calibration database will be updated, but the background '
              'image is missing.')
    else:
        backgroundImagesPath = simpleFolderFinder(backgroundPath)
        FileList = TupleOfFiles(backgroundImagesPath)
        testBackground = ImportImageFiles([FileList[0]])
        if testImage.shape[0] != testBackground.shape[0]:
            backgroundPath = os.path.join(basePath, 'MIRAGE', 'HighESpec', '%d' % runDate, 'darkfield02')
            if not os.path.exists(backgroundPath):
                print(
                    'WARNING E2: No darkfields with the right image dimensions were found. The calibration database '
                    'will be updated, but the background image is missing.')
            else:
                backgroundImagesPath = simpleFolderFinder(backgroundPath)
                FileList = TupleOfFiles(backgroundImagesPath)
                testBackground = ImportImageFiles([FileList[0]])
                if testImage.shape[0] != testBackground.shape[0]:
                    print(
                        'WARNING E3: No darkfields with the right image dimensions were found. The calibration '
                        'database will be updated, but the background image is missing.')
                else:
                    foundDarkfields = 1
        else:
            foundDarkfields = 1
    if foundDarkfields:
        if BackgroundImage == 0:
            BackgroundImage, BackgroundNoise = backgroundImages(backgroundImagesPath, J, pts)
            BackgroundNoise = BackgroundNoise[:, CutOff:]
        #  now loading the right magnet map
        FileLocation = os.path.join(basePath, 'Calibrations', 'HighESpec')
        # E, dxoverdE = getEnergyTransformation(FileLocation, L)
        AverageEnergy, dxoverdEAverage, ErrorEnergyPos, dxoverdEPos, ErrorEnergyNeg, dxoverdENeg, Div, PropagationLength, ErrorPropagationLength, Cell2Magnet = getEnergyTransformation(FileLocation, L)
        tupleOut = applyCutOffFcn((AverageEnergy, dxoverdEAverage, ErrorEnergyPos, dxoverdEPos, ErrorEnergyNeg, dxoverdENeg, PropagationLength, ErrorPropagationLength), 300)
        AverageEnergy, dxoverdEAverage, ErrorEnergyPos, dxoverdEPos, ErrorEnergyNeg, dxoverdENeg, PropagationLength, ErrorPropagationLength = tupleOut
        EnergyDivergenceTuple = (AverageEnergy, dxoverdEAverage, ErrorEnergyPos, dxoverdEPos, ErrorEnergyNeg, dxoverdENeg, Div, PropagationLength, ErrorPropagationLength, Cell2Magnet)
        totalCalibrationFilePath = os.path.join(calPath, 'HighEspec', '%d' % runDate)
        if not os.path.exists(totalCalibrationFilePath):
            os.mkdir(totalCalibrationFilePath)
        simpleRunName = runName[9:]
        calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
        calibrationTuple = (J, W, pts, BackgroundImage, L, CutOff, BackgroundNoise, EnergyDivergenceTuple)
        np.save(calFile, calibrationTuple)
        #  the entry for the database is a relative path:
        relPathForDatabase = os.path.join('HighESpec', '%d' % runDate, '%s.npy' % simpleRunName)
        changeFileEntry(relPathForDatabase, runName, calPath)
    else:
        totalCalibrationFilePath = os.path.join(calPath, 'HighESpec', '%d' % runDate)
        if not os.path.exists(totalCalibrationFilePath):
            os.mkdir(totalCalibrationFilePath)
        simpleRunName = runName[9:]
        calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
        if os.path.exists(calFile):
            os.remove(calFile)
            changeFileEntry('', runName, calPath)


def applyCutOffFcn(TupleIn, CutOff):
    TupleOut = []
    for i in range(0, len(TupleIn)):
        tmp = TupleIn[i]
        if len( tmp.shape ) > 1:
            TupleOut.append(TupleIn[i][:, CutOff:])
        else:
            TupleOut.append(TupleIn[i][CutOff:])
    return tuple(TupleOut)


def getEnergyTransformation(FileLocation, L):
    ImageWarpingFile = 'PositionVsEnergy.mat'
    ImageWarpingVariable = ['Screen', 'AverageEnergy', 'ErrorEnergyPos', 'ErrorEnergyNeg', 'Div',
                           'PropagationLength', 'ErrorPropagationLength', 'Cell2Magnet']
    ScreenEnergyOnAverage = loadMatFile(FileLocation, ImageWarpingFile, ImageWarpingVariable)
    Screen, AverageEnergyB, ErrorEnergyPosB, ErrorEnergyNegB, Div, PropagationLengthB, ErrorPropagationLengthB, Cell2Magnet = ScreenEnergyOnAverage
    Screen = Screen[0,]*1e3
    PropagationLength = np.interp(L, Screen, PropagationLengthB[0,])
    ErrorPropagationLength = np.interp(L, Screen, ErrorPropagationLengthB[0,])
    AverageEnergy, dxoverdEAverage = calculateEdxoverdE(L, Screen, AverageEnergyB)
    ErrorEnergyPos, dxoverdEPos = calculateEdxoverdE(L, Screen, ErrorEnergyPosB)
    ErrorEnergyNeg, dxoverdENeg = calculateEdxoverdE(L, Screen, ErrorEnergyNegB)
    return AverageEnergy, dxoverdEAverage, ErrorEnergyPos, dxoverdEPos, ErrorEnergyNeg, dxoverdENeg, Div, PropagationLength, ErrorPropagationLength, Cell2Magnet


def calculateEdxoverdE(L, Screen, AverageEnergyB):
    dx = np.diff(L)
    AverageEnergy = np.zeros([AverageEnergyB.shape[0], L.shape[0]])
    dxoverdEAverage = np.zeros([AverageEnergyB.shape[0], L.shape[0]])
    for i in range(0, AverageEnergyB.shape[0]):
        AverageEnergy[i, :] = np.interp(L, Screen, AverageEnergyB[i, :])
        dE = np.diff(AverageEnergy[i, :])
        dxoverdE = dx/dE
        dxoverdE = np.append(dxoverdE, dxoverdE[-1])
        dxoverdEAverage[i, :] = dxoverdE
    return AverageEnergy, dxoverdEAverage


def loadMatFile(SettingPath, FileName, VariableName):
    PathToLoad = os.path.join(SettingPath, FileName)
    VabiableLibrary = scipy.io.loadmat(PathToLoad)
    Variable = []
    for i in range(0, len(VariableName)):
        if VariableName[i] in VabiableLibrary:
            Variable.append(VabiableLibrary[VariableName[i]])
        else:
            print('Warning! Variable %s does not exist in the calibration file. Setting it to 0 ->' % VariableName[i])
            # This is a fix for now. Simply, because previous calibration files did not contain CutOff, not
            # BackgroundNoise. And it is weirdly saved (it has to be done by CutOff = CutOff[0][0] later. Therefore:
            Variable.append([[0]])
    return Variable


def convertRunNameToDate(runName):
    if isinstance(runName, str):
        if "\\" in r"%r" % runName:
            # This has to be done, because a backslash is a special character and cannot be identified
            # otherwise. However the line below introduces not only "\\" (which can be identified), but
            # also " ' " in front (and in the back) of the string, which needs to be skipped.
            runName = r"%r" % runName
            walkThrough = 1
        else:
            walkThrough = 0
        runNameAccumulate = runName[walkThrough]
        walkThrough += 1
        while walkThrough < len(runName):
            if runName[walkThrough] == '/' or runName[walkThrough] == '\\':
                break
            runNameAccumulate += runName[walkThrough]
            walkThrough += 1
        runName = float(runNameAccumulate)
    return runName


def backgroundImages(Path, J, pts):
    if len(Path) > 0:
        FileList = TupleOfFiles(Path)
        BackgroundImages = ImportImageFiles(FileList)
        BackgroundImage = np.mean(BackgroundImages, axis=2)
        BackgroundNoise = np.std(BackgroundImages, axis=2) / np.sqrt(BackgroundImages.shape[-1])
        WarpedBackgroundImage, ___ = four_point_transform(BackgroundImage, pts, J)
        BackgroundNoise, ___ = four_point_transform(BackgroundNoise, pts, J)
        WarpedBackgroundImage = np.fliplr(WarpedBackgroundImage)
        BackgroundNoise = np.flip(BackgroundNoise)
    else:
        WarpedBackgroundImage = 0
        BackgroundNoise = 0
    return WarpedBackgroundImage, BackgroundNoise


'''
Image transformation code:
'''


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, J=1):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    WarpedImage = np.multiply(warped, 1 / J)
    return WarpedImage, M


def ProjectiveTransformJacobian(params, ImSize):
    u = np.arange(0, ImSize[1])
    v = np.arange(0, ImSize[0])
    u, v = np.meshgrid(u, v)
    a = params[0, 0]
    b = params[0, 1]
    c = params[0, 2]
    d = params[1, 0]
    e = params[1, 1]
    f = params[1, 2]
    g = params[2, 0]
    h = params[2, 1]
    lam = 1 / (g * u + h * v + 1)
    J1 = lam ** 2 * (a * e - b * d)
    J2 = (d * h - e * g) * (a * u + b * v + c)
    J3 = (b * g - a * h) * (d * u + e * v + f)
    J23 = J2 + J3
    J = J1 + np.multiply(lam ** 3, J23)
    return J


def getJacobianAndSpatialCalibration(OrgImageSize, WarpedImageSize, M, Length, ScreenStart):
    """
    The function is used to calculate the Jacobian, which compensates for projecting the image.
    The pixel values need to be adjusted for stretching. It used a warped image and the transformation matrix.
    The Length is the total length of the spectrometer. The CentrePoint is a point on the screen which distance
    to the start of the electron spectrometer screen is known. The CentrePoint is the pixel row and CentrePointDistance
    is the physical distance.
    :param ScreenStart:
    :param WarpedImageSize:
    :param OrgImageSize:
    :param M:
    :param Length:
    :return:
    """
    Jxy = ProjectiveTransformJacobian(M, OrgImageSize)
    Juv = cv2.warpPerspective(Jxy, M, (WarpedImageSize[1], WarpedImageSize[0]))
    PixelLength = WarpedImageSize[1]
    L = np.arange(0, PixelLength)
    L = L / PixelLength * Length
    L = L + ScreenStart
    return Juv, L
