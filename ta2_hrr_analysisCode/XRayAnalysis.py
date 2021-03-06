#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:45:56 2019
XRayFilterPack
@author: Gruse
"""
from typing import Tuple, Union
import os
import scipy.io
import numpy as np
import numpy.matlib as ml
from numpy.core.multiarray import ndarray
from scipy.special import kv
from scipy.integrate import trapz
from scipy import optimize
import csv
from shapely.geometry import Point, Polygon
import pkg_resources
import re


def createYLimits(BackgroundNoise, CameraTuple, ecrit, Y):
    '''
    This is a bit tricky to understand. Essentially, it calculates the minimal critical energy, and counts on the camera to be
    above the detection threshold, which is hereby defined for being 1 photon per pixel. The camera noise is taken into account.
    There is a DebuggingXRayAnalysis notebook, which talks and shows a bit more about this topic.
    '''
    (_, _, _, Alpha, _, energy, TQ) = CameraTuple
    AverageNoise = np.mean( np.array(BackgroundNoise) )
    tmp = [x[-1]/x[0] for x in Y]
    minimalCounts = [scipy.integrate.trapz( Alpha * TQ * synchrotronFunction(energy, ec), energy) for ec in ecrit]
    minimalCountsHighFilter = np.array(tmp) * np.array(minimalCounts)
    idrel = np.argmin( np.abs( minimalCountsHighFilter - AverageNoise ) )
    Ylimits = np.zeros(Y.shape[1])
    Ylimits[0] = minimalCounts[idrel]
    for i in range(1, Y.shape[1]):
        tmp = [x[i]/x[0] for x in Y]
        tmp2 = np.array(tmp) * np.array(minimalCounts)
        Ylimits[i] = tmp2[idrel]
    return Ylimits


def averageXXpercent(images, XX):
    '''
    This function returns the top xx% images in respect to their counts.
    images is an np array with the third dimension specifying the number of images. XX is the percentage.
    E.g. XX=60 and images.shape[2]=10 will return an np array with images.shape[2]=6, which are the brightest pictures of the org stack.
    '''
    xxPercent = (1 - XX/100)
    totalCounts = np.sum( np.sum(images, axis=0), axis=0)
    totNum = totalCounts.shape[0]
    # counts80 = ( np.argsort(totalCounts) >= np.round(totNum*halfPercent) ) == ( np.argsort(totalCounts) <= np.round(totNum*(1 - halfPercent))-1 )
    countsXX = ( np.argsort(totalCounts) >= np.round(totNum*xxPercent) )
    return np.mean( images[:, :, countsXX], axis=2)


def XRayEcrit(FileList, calibrationTuple, AnalysisMethod):
    (ImageTransformationTuple, CameraTuple, TransmissionTuple) = calibrationTuple
    (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error, energy, TQ) = CameraTuple
    (filterNames, ecrit, Y, YLimits) = TransmissionTuple
    (P, BackgroundImage, BackgroundNoise, PBList) = ImageTransformationTuple
    images = ImportImageFiles(FileList)
    data = []
    # I = individual, M = Median, A = Average, A60 = top 60% averaged
    method = AnalysisMethod

    if method == 'I':
        Peaklist = []
    elif method == 'M':
        images = np.expand_dims( np.median(images, axis=2), axis=2)
    elif method == 'A':
        images = np.expand_dims( np.mean(images, axis=2), axis=2)
    elif method == 'A60':
        images = np.expand_dims( averageXXpercent(images, 60), axis=2)

    for i in range(0, images.shape[2]):
        image = images[:, :, i] - BackgroundImage
        # tungsten filtered:
        WList = getValues(image, PBList)
        ValueList = getValues(image - np.mean(WList), P)
        # not tungsten filtered (legacy code, but should remain to see what has been done before):
        # ValueList = getValues(image, P)
        # this makes sure only values above the std of the noise gets taken into account of calculating the critical
        # energy
        preparedData, PeakIntensity, PeakIntensityStd = cleanValues(ValueList, YLimits)
        if len(preparedData) != 0:
            data.append(preparedData)
            if method == 'I':
                Peaklist.append(PeakIntensity)
    if len(data) > 0:
        AverageValues, StdValues = combineImageValues(data)
        bestEcrit, ecritStd = determineEcrit(AverageValues, StdValues, ecrit, Y)
        if method == 'I':
            PeakIntensityStd = np.std(np.array(Peaklist))
            PeakIntensity = np.mean(np.array(Peaklist))
        if method == 'M' or method == 'A' or method == 'A60':
            PeakIntensityStd = PeakIntensityStd[0]
            PeakIntensity = PeakIntensity[0]
        NPhotons, sigma_NPhotons, relevantNPhotons_Omega_s, sigma_relevantNPhotons_Omega_s, relevantNPh_mrad_01BW, sigma_relevantNPh_mrad_01BW = getPhotonFlux(bestEcrit, ecritStd, PeakIntensity, PeakIntensityStd, CameraTuple)
        analysedData = (AverageValues, StdValues, PeakIntensity, PeakIntensityStd, bestEcrit, ecritStd, NPhotons, sigma_NPhotons, relevantNPhotons_Omega_s, sigma_relevantNPhotons_Omega_s, relevantNPh_mrad_01BW, sigma_relevantNPh_mrad_01BW)
    else:
        analysedData = []
        print('Not enough signal on the x-ray camera to calculate a critical energy')
    return analysedData


def getPhotonFlux(ecrit, ecritStd, PeakIntensity, PeakIntensityStd, CameraTuple):
    (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error, energy, TQ) = CameraTuple
    Theta2 = (PixelSize / GasCell2Camera) ** 2 * (1e3 * 1e3) # mrad^2

    S = numberSpectrum(energy, ecrit)
    NormFactor = scipy.integrate.trapz(S, energy)

    integralFilter = scipy.integrate.trapz(energy * S / NormFactor * TQ, energy)
    NPhotons = PeakIntensity / (Alpha * integralFilter)
    QS1 = PeakIntensityStd / (Alpha * integralFilter)
    QS2 = PeakIntensityStd * Alpha_error / (Alpha ** 2 * integralFilter)

    SPlus = numberSpectrum(energy, ecrit + ecritStd)
    NormFactorPlus = scipy.integrate.trapz(SPlus, energy)
    integralFilterPlus = scipy.integrate.trapz(energy * SPlus / NormFactorPlus * TQ, energy)
    integralFilterPlus = integralFilter - integralFilterPlus
    SNegative = numberSpectrum(energy, ecrit - ecritStd)
    NormFactorNegative = scipy.integrate.trapz(SNegative, energy)
    integralFilterNegative = scipy.integrate.trapz(energy * SNegative / NormFactorNegative * TQ, energy)
    integralFilterNegative = integralFilter - integralFilterNegative
    QS3P = PeakIntensityStd * integralFilterPlus / (Alpha * integralFilter ** 2)
    QS3N = PeakIntensityStd * integralFilterNegative / (Alpha * integralFilter ** 2)
    sigma_NPhotonsP = np.sqrt(QS1 ** 2 + QS2 ** 2 + QS3P ** 2)
    sigma_NPhotonsN = np.sqrt(QS1 ** 2 + QS2 ** 2 + QS3N ** 2)
    sigma_NPhotons = (sigma_NPhotonsN, sigma_NPhotonsP)

    limitedPhotons = numberSpectrumBandlimited(energy, ecrit, NormFactor)
    relevantNPhotons = NPhotons * limitedPhotons
    sigma_relevantNPhotons = sigma_NPhotonsP * limitedPhotons
    relevantNPhotons_Omega_s = relevantNPhotons * RepRate / Theta2
    sigma_relevantNPhotons_Omega_s = sigma_relevantNPhotons * RepRate / Theta2
    relevantNPh_mrad_01BW = np.max( NPhotons * S * 0.001 / NormFactor * energy / Theta2)
    EnergySpectrum = NPhotons * S * 0.001 / NormFactor * energy / Theta2
    ArgNPh = np.argmax( NPhotons * S / NormFactor * energy / Theta2)
    sigma_relevantNPh_mrad_01BW = EnergySpectrum[ArgNPh]
    return NPhotons, sigma_NPhotons, relevantNPhotons_Omega_s, sigma_relevantNPhotons_Omega_s, relevantNPh_mrad_01BW, sigma_relevantNPh_mrad_01BW


def numberSpectrumBandlimited(energy, ecrit, NormFactor):
    '''
    Calculates the number of photons above the k-edge of aluminium, in respect to the entire number of photons
    (which is the input variable "NormFactor") for a specific critical energy.
    energy is the original range of defined possible photons.
    Al k-edge: 1.5596 keV.
    Units here in keV
    '''
    AlEdge = 1.0 #1.5596
    energyLimited = np.arange(AlEdge, energy[-1], (energy[-1] - AlEdge) / energy.size )
    # energy01percent = np.arange(0.9995 * ecrit, 1.0005 * ecrit, (1.0005 * ecrit - 0.9995 * ecrit) / 100) <-- this was to calculate the no of photons in 0.1% bandwidth
    S01percent = numberSpectrum(energyLimited, ecrit) / NormFactor
    return scipy.integrate.trapz(S01percent, energyLimited)


def determineEcrit(AverageValues, StdValues, ecrit, Y):
    N = len(AverageValues)
    AverageValuesMat = prepDim(AverageValues, ecrit)
    StdValuesMat = prepDim(StdValues, ecrit)
    Chi2 = np.sum((AverageValuesMat - Y)**2/StdValuesMat**2, 1)/(N-1)
    BestId = np.argmin(Chi2)
    bestEcrit = ecrit[BestId]
    # uncertainty:
    if BestId >= ecrit.shape[0]-1:
        BestId -= 2
    elif BestId <= 1:
        BestId += 2
    DeltaF = (Y[BestId+1] - Y[BestId-1]) / (ecrit[BestId+1] - ecrit[BestId-1])
    alphaInverted = 1/np.sum(DeltaF * DeltaF)
    ecritStd = np.sqrt(Chi2[BestId] * alphaInverted)
    return bestEcrit, ecritStd


def prepDim(vec, e):
    vec = np.array(vec)
    vec = ml.repmat(vec, len(e), 1)
    return vec


def combineImageValues(data):
    AverageValues = []
    StdValues = []
    for i in range(0, len(data[0])):
        ValuesHere = [placeHolder for x in data for placeHolder in x[i]]
        AverageValues.append(np.mean(ValuesHere))
        StdValues.append(np.std(ValuesHere))
    return AverageValues, StdValues


def cleanValues(ValueList, backgroundNoiseList):
    cleanV = []
    PeakIntensity = []
    PeakIntensityStd = []
    ToNormalise = np.zeros(len(ValueList))
    ToNormaliseStd = np.zeros(len(ValueList))
    for i in range(0, len(ValueList)):
        tmpV = ValueList[i]
        # Vs = tmpV[tmpV > backgroundNoiseList[i]]
        Vs = tmpV[tmpV > -1e6]
        if len(Vs) == 0:
            print('Values in \'clean Values\' are empty. Std of background is too high')
            cleanV = []
            break
        cleanV.append(Vs)
        ToNormalise[i] = np.mean(cleanV[-1])
        ToNormaliseStd[i] = np.std(cleanV[-1])
        if i == 0:
            PeakIntensity.append(ToNormalise[i])
            PeakIntensityStd.append(ToNormaliseStd[i])
    if len(Vs) != 0:
        Norm = np.sum(ToNormalise)
        cleanV = [x/Norm for x in cleanV]
    return cleanV, PeakIntensity, PeakIntensityStd


def synchrotronFunction(energy, ecrit):
    NumberSpectrum = numberSpectrum(energy, ecrit)
    Intensity = NumberSpectrum * energy
    return Intensity


def numberSpectrum(energy, ecrit):
    phi = energy / (2 * ecrit)
    K2over3 = scipy.special.kv(2 / 3, phi)
    NumberSpectrum = energy / ecrit ** 2 * K2over3 ** 2
    return NumberSpectrum


def dataVStheoryToMinimise(data, theory):
    Value = 0
    for i in range(0, len(data)):
        Value += (data[i] - theory[i]) ** 2
    return Value


def getValues(Image, indicesList):
    ValueList = []
    for i in range(0, len(indicesList)):
        Indices = indicesList[i]
        Values = np.zeros(len(Indices))
        for j in range(0, len(Indices)):
            polygonPoints = Indices[j]
            Values[j] = Image[polygonPoints[1], polygonPoints[0]]
        ValueList.append(Values)
    return ValueList


# def MeanOfImage(self, Image):
#    BrightMean, BrightStd = getMeanOfImageWithMask(Image, self.BrightFilterMask)
#    DarkMean, DarkStd = getMeanOfImageWithMask(Image, self.DarkFilterMask)
#    return BrightMean, DarkMean, BrightStd, DarkStd


"""
Calibration File of the X-Ray Analysis Scripts --------------------------------------------------
"""


def getMaskFromPts(Pts, idx, idy, Mask):
    """
    This is rather a list of indices than a mask.
    :param Pts: set of points defining a polygon. Points within this polygon will be added to the list of indiecs.
    :param idx: through a meshgrid formed list of indices, partnered with v
    :param idy: through a meshgrid formed list of indices, partnered with ^
    :param Mask: empty list, pairs of indices will be added to this list.
    :return: Mask will be returned. Details see above.
    """
    for k in range(len(Pts)):
        poly = Polygon(Pts[k])
        for i in range(idx.size):
            if Point(idx[i], idy[i]).within(poly):
                Mask.append([idx[i], idy[i]])
    return Mask


def findCalibrationEntry(runName, calPath):
    csvFile = os.path.join(calPath, 'CalibrationPaths.csv')  # change to the correct name
    TotalFile = csv.reader(open(csvFile))
    entries = list(TotalFile)
    diagnostic = 'XRay'
    identifiedRun = 0
    identifiedDiag = 0
    for i in range(0, len(entries[0])):
        if entries[0][i] == diagnostic:
            identifiedDiag = i
    for j in range(0, len(entries)):
        if entries[j][0] == runName:
            identifiedRun = j
    if identifiedRun == 0 and identifiedDiag == 0:
        raise ValueError('The runName (%s) and diagnostic (%s) was found!' % (runName, diagnostic))
    return entries, identifiedRun, identifiedDiag


def changeFileEntry(NewEntry, runName, calPath=r'Y:\ProcessedCalibrations'):
    entries, identifiedRun, identifiedDiag = findCalibrationEntry(runName, calPath)
    oldEntry = entries[identifiedRun][identifiedDiag]
    if oldEntry == '':
        oldEntry = 'empty'
    entries[identifiedRun][identifiedDiag] = NewEntry
    csvFile = os.path.join(calPath, 'CalibrationPaths.csv')  # change to the correct name
    writer = csv.writer(open(csvFile, 'w', newline=''))
    writer.writerows(entries)
    print('Changed run: %s of diagnostic %s to: \'%s\'. Previously it was \'%s\'.' % (
        runName, 'XRay', NewEntry, oldEntry))
    return entries


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
    Images = np.zeros(shape=(imarray.shape[0], imarray.shape[1], len(FileList)))
    for i in range(0, len(FileList)):
        im = Image.open(FileList[i])
        Images[:, :, i] = np.array(im)
    return Images


def importXRayTransmissions(txtFile):
    LineLength = 0
    with open(txtFile) as txtContent:
        txtReader = csv.reader(txtContent, delimiter=',')
        for row in txtReader:
            LineLength += 1
    cnt = 0
    ecrit = np.empty([LineLength - 1, ])
    with open(txtFile) as txtContent:
        txtReader = csv.reader(txtContent, delimiter=',')
        for row in txtReader:
            if cnt == 0:
                VariableNames = row
                Y = np.empty([LineLength - 1, len(VariableNames) - 1])
                cnt += 1
                continue
            ecrit[cnt - 1] = float(row[0])
            Y[cnt - 1, :] = np.array(row[1:]).astype('float')
            cnt += 1
    return VariableNames, ecrit, Y


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


def backgroundImages(Path):
    FileList = TupleOfFiles(Path)
    BackgroundImages = ImportImageFiles(FileList)
    BackgroundImage = np.mean(BackgroundImages, axis=2)
    BackgroundNoise = np.std(BackgroundImages, axis=2)
    return BackgroundImage, BackgroundNoise


def mainCalibrationFunction(runName, basePath='Z:\\', calPath='Y:\\ProcessedCalibrations', BackgroundImage=0,
                            BackgroundNoise=0):
    """
    The most important parameter, which is constructed here is the calibrationTuple. This is a tuple containing the
    information necessary, to analyse an image.
    calibrationTuple = (ImageTransformationTuple, CameraTuple, TransmissionTuple, PBList)
    The three parameter tuple in there correspond to three parts of the analysis:

    TransmissionTuple = (filterNames, ecrit, Y, YLimits)
    This contains the filter pack names, the vector of different critical energies, which was used to contruct Y.
    Y is the normalised transmission through the 4 different filters and through the non-filter background.
    The normalisation is based on the sum of the transmission. The dimension is (ecrit, filter)

    CameraTuple = (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error)
    These parameter are pretty much used to get the photon flux, once the critical energy is estimated.

    ImageTransformationTuple = (P, BackgroundImage, BackgroundNoise, PBList) In order to transform the raw images into
    usuable data, one has to subtract the darkfield and then identify the pixel values for the different filter. P is
    a list of a list of indices, which pixel are contained for a specific filter are contained. The BackgroundNoise
    is the std of the darkfields and the idea is to use it to estimate a lowest level of eliminating criteria. PB is the
    area of the tungstan, which enable the background on shot as a lot of noise is generated
    :param runName:
    :param basePath:
    :param calPath:
    :param BackgroundImage: in case a background images, one can copy one from another day and put it into this function
    :param BackgroundNoise: ^
    :return: Saves the calibration tuple and changes the entry in the database
    """
    runDate = convertRunNameToDate(runName)
    if runDate >= 20191108:
        txtFile = 'Compact2019OctoberTA2.txt'
        txtFilePath = os.path.join(basePath, 'Calibrations', 'XRay', txtFile)
        filterNames, ecrit, Y = importXRayTransmissions(txtFilePath)
        runPath = os.path.join(basePath, 'MIRAGE', 'XRay', runName)
        imageFolderPath = simpleFolderFinder(runPath)
        FileList = TupleOfFiles(imageFolderPath)
        testImage = ImportImageFiles([FileList[0]])

        # Lead background
        B = 'W'
        PB = (np.array([[0, 57], [3, 55], [0, 44]]))
        #  the non-filter background
        F1 = 'Vac'
        P1 = (np.array([[9, 61], [0, 66], [0, 255], [255, 255], [255, 188], [62, 245], [0, 4], [0, 36]]),
            np.array([[24, 100], [25, 105], [63, 93], [102, 230], [107, 229], [71, 98], [110, 87], [147, 218], [156, 216],
                        [124, 110], [163, 97], [196, 205], [200, 203], [175, 118], [215, 104], [241, 189], [246, 187],
                        [223, 111], [256, 99], [256, 93], [220, 105], [191, 0], [184, 0], [213, 97], [173, 111], [140, 0],
                        [134, 0], [161, 92], [121, 103], [90, 0], [85, 0], [109, 83], [69, 94], [41, 0], [35, 0], [61, 89]])
            )
        #  second to brightest area
        F2 = 'C10H8O4'
        P2 = (np.array([[181, 120], [210, 113], [231, 188], [204, 197]]),
            np.array([[47, 4], [79, 4], [98, 77], [73, 86]]))
        F3 = 'Mg'
        P3 = (np.array([[157, 213], [190, 203], [160, 102], [129, 114]]),
            np.array([[156, 87], [124, 97], [95, 0], [130, 0]]))
        F4 = 'Al(0.95)Mg(0.05)'
        P4 = (np.array([[141, 216], [111, 223], [77, 104], [106, 94]]),
            np.array([[207, 93], [176, 103], [146, 0], [179, 0]]))
        #  Darkest area:
        F5 = 'Al(0.98)Mg(0.01)Si(0.01)'
        P5 = (np.array([[98, 229], [67, 239], [31, 111], [59, 103]]),
            np.array([[224, 97], [197, 3], [229, 3], [253, 90]]))
        Fold = [F1, F2, F3, F4, F5]
        Pold = [P1, P2, P3, P4, P5]
        F = []
        P = []

        sizex, sizey, _ = testImage.shape
        XM, YM = np.meshgrid(range(sizex), range(sizey))
        idx = XM.ravel()
        idy = YM.ravel()

        for fN in filterNames:
            cc = 0
            Mask = []
            for scan in Fold:
                if scan == fN:
                    # check if this works... Not sure, because I overwrite it again...
                    P.append(getMaskFromPts(Pold[cc], idx, idy, []))
                cc += 1
        PBList = [ getMaskFromPts([ PB ], idx, idy, []) ]
        if sizex > 1000:
            PixelSize = 13e-6
        else:
            PixelSize = 13e-6 * 4
        GasCell2Camera = 1.228
        RepRate = 1
        Alpha = 274
        Alpha_error = 0.06
        # additionally to the camera settings, the energy scale and the transmission through the background is necessary for
        # for the photon flux:
        txtFile = '2019OctoberTA2.txt'
        txtFilePath = os.path.join(basePath, 'Calibrations', 'XRay', txtFile)
        _, energy, TQ = importXRayTransmissions(txtFilePath)
        TQ = TQ[:, 0]
        CameraTuple = (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error, energy, TQ)

        if BackgroundImage != 0:
            foundDarkfields = 1
        else:
            foundDarkfields = 0
        backgroundPath = os.path.join(basePath, 'MIRAGE', 'XRay', '%d' % runDate, 'darkfield01')
        if not os.path.exists(backgroundPath):
            print('WARNING E1: No darkfields were found. The calibration database will be updated, but the background '
                'image is missing.')
        else:
            backgroundImagesPath = simpleFolderFinder(backgroundPath)
            FileList = TupleOfFiles(backgroundImagesPath)
            testBackground = ImportImageFiles([FileList[0]])
            if testImage.shape[0] != testBackground.shape[0]:
                backgroundPath = os.path.join(basePath, 'MIRAGE', 'XRay', '%d' % runDate, 'darkfield02')
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
                BackgroundImage, BackgroundNoise = backgroundImages(backgroundImagesPath)
            ImageTransformationTuple = (P, BackgroundImage, BackgroundNoise, PBList)
            totalCalibrationFilePath = os.path.join(calPath, 'XRay', '%d' % runDate)
            if not os.path.exists(totalCalibrationFilePath):
                os.mkdir(totalCalibrationFilePath)
            simpleRunName = runName[9:]
            calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
            YLimits = createYLimits(BackgroundNoise, CameraTuple, ecrit, Y)
            TransmissionTuple = (filterNames, ecrit, Y, YLimits)
            calibrationTuple = (ImageTransformationTuple, CameraTuple, TransmissionTuple)
            np.save(calFile, calibrationTuple)
            #  the entry for the database is a relative path:
            relPathForDatabase = os.path.join('XRay', '%d' % runDate, '%s.npy' % simpleRunName)
            changeFileEntry(relPathForDatabase, runName, calPath)
        else:
            totalCalibrationFilePath = os.path.join(calPath, 'XRay', '%d' % runDate)
            if not os.path.exists(totalCalibrationFilePath):
                os.mkdir(totalCalibrationFilePath)
            simpleRunName = runName[9:]
            calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
            if os.path.exists(calFile):
                os.remove(calFile)
                changeFileEntry('', runName, calPath)
    else:
        print('There is no filter in the beam line')