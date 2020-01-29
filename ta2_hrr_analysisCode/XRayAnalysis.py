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



def XRayEcrit(FileList, calibrationTuple):
    (ImageTransformationTuple, CameraTuple, TransmissionTuple) = calibrationTuple
    (filterNames, ecrit, Y) = TransmissionTuple
    (P, BackgroundImage, BackgroundNoise) = ImageTransformationTuple
    backgroundNoiseList = getValues(BackgroundNoise, P)
    images = ImportImageFiles(FileList)
    data = []
    for i in range(0, images.shape[2]):
        image = images[:, :, i] - BackgroundImage
        ValueList = getValues(image, P)
        # this makes sure only values above the std of the noise gets taken into account of calculating the critical
        # energy
        preparedData, PeakIntensity = cleanValues(ValueList, backgroundNoiseList)
        if len(preparedData) != 0:
            data.append(preparedData)
            AverageValues, StdValues = combineImageValues(data)
            bestEcrit, ecritStd = determineEcrit(AverageValues, StdValues, ecrit, Y)
            PeakIntensityStd = np.std(np.array(PeakIntensity))
            PeakIntensity = np.mean(np.array(PeakIntensity))
            NPhotons, sigma_NPhotons, _, NPhotons_01percent_omega_s = getPhotonFlux(bestEcrit, ecritStd, PeakIntensity, PeakIntensityStd, CameraTuple)
            analysedData = (bestEcrit, ecritStd, NPhotons, sigma_NPhotons, NPhotons_01percent_omega_s)
    return analysedData


def getPhotonFlux(ecrit, ecritStd, PeakIntensity, PeakIntensityStd, CameraTuple):
    (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error, energy, TQ) = CameraTuple
    Theta2 = (PixelSize / GasCell2Camera) ** 2 * 1e6

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

    limitedEnergy = numberSpectrumBandlimited(ecrit, NormFactor)
    NPhotons01percent = NPhotons * limitedEnergy
    NPhotons_01percent_omega_s = NPhotons01percent * RepRate / Theta2
    return NPhotons, sigma_NPhotons, NPhotons01percent, NPhotons_01percent_omega_s


def numberSpectrumBandlimited(ecrit, NormFactor):
    energy01percent = np.arange(0.9995 * ecrit, 1.0005 * ecrit, (1.0005 * ecrit - 0.9995 * ecrit) / 100)
    S01percent = numberSpectrum(energy01percent, ecrit) / NormFactor
    return scipy.integrate.trapz(S01percent, energy01percent)


def determineEcrit(AverageValues, StdValues, ecrit, Y):
    N = len(AverageValues)
    AverageValues = prepDim(AverageValues, ecrit)
    StdValues = prepDim(StdValues, ecrit)
    Chi2 = np.sum((AverageValues - Y)**2/StdValues**2, 1)/N
    BestId = np.argmin(Chi2)
    bestEcrit = ecrit[BestId]
    # uncertainty:
    Chi2 = np.sum((AverageValues - Y)**2, 1)/N 
    DeltaF = (Y[BestId+1] - Y[BestId-1]) / (ecrit[BestId+1] - ecrit[BestId-1])
    alphaInverted = 1/np.sum(DeltaF * DeltaF.T)
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
        ValuesHere = [x[i] for x in data]
        AverageValues.append(np.mean(ValuesHere))
        StdValues.append(np.std(ValuesHere))
    return AverageValues, StdValues


def cleanValues(ValueList, backgroundNoiseList):
    cleanV = []
    PeakIntensity = []
    ToNormalise = np.zeros(len(ValueList))
    for i in range(0, len(ValueList)):
        tmpV = ValueList[i]
        tempB = backgroundNoiseList[i]
        Vs = tmpV[tmpV > tempB]
        if len(Vs) == 0:
            print('Values in \'clean Values\' are empty. Std of background is too high')
            cleanV = []
            break
        cleanV.append(Vs)
        ToNormalise[i] = np.mean(cleanV[-1])
        if i == 0:
            PeakIntensity.append(ToNormalise[i])
    if len(Vs) != 0:
        Norm = np.sum(ToNormalise)
        cleanV = [x/Norm for x in cleanV]
    return cleanV, PeakIntensity


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
    calibrationTuple = (ImageTransformationTuple, CameraTuple, TransmissionTuple)
    The three parameter tuple in there correspond to three parts of the analysis:

    TransmissionTuple = (filterNames, ecrit, Y)
    This contains the filter pack names, the vector of different critical energies, which was used to contruct Y.
    Y is the normalised transmission through the 4 different filters and through the non-filter background.
    The normalisation is based on the sum of the transmission. The dimension is (ecrit, filter)

    CameraTuple = (PixelSize, GasCell2Camera, RepRate, Alpha, Alpha_error)
    These parameter are pretty much used to get the photon flux, once the critical energy is estimated.

    ImageTransformationTuple = (P, BackgroundImage, BackgroundNoise) In order to transform the raw images into
    usuable data, one has to subtract the darkfield and then identify the pixel values for the different filter. P is
    a list of a list of indices, which pixel are contained for a specific filter are contained. The BackgroundNoise
    is the std of the darkfields and the idea is to use it to estimate a lowest level of eliminating criteria.
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
        TransmissionTuple = (filterNames, ecrit, Y)

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
        X, Y = np.meshgrid(range(sizex), range(sizey))
        idx = X.ravel()
        idy = Y.ravel()

        for fN in filterNames:
            cc = 0
            Mask = []
            for scan in Fold:
                if scan == fN:
                    # check if this works... Not sure, because I overwrite it again...
                    P.append(getMaskFromPts(Pold[cc], idx, idy, []))
                cc += 1
        if sizex > 1000:
            PixelSize = 13e-6
        else:
            PixelSize = 13e-6 * 4
        GasCell2Camera = 1.2
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
            ImageTransformationTuple = (P, BackgroundImage, BackgroundNoise)
            totalCalibrationFilePath = os.path.join(calPath, 'XRay', '%d' % runDate)
            if not os.path.exists(totalCalibrationFilePath):
                os.mkdir(totalCalibrationFilePath)
            simpleRunName = runName[9:]
            calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
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
