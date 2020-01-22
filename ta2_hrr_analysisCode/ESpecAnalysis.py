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


def TupelOfFiles(path="", Filetype=('.tif', '.tiff', '.TIFF', '.TIF')):
    if not isinstance(Filetype, list):
        Filetype = list(Filetype)
    if len(path) == 0:
        path = "."
    FileList = []
    for files in os.listdir(path):
        for endings in Filetype:
            if files.endswith(endings):
                FileList.append(os.path.join(path, files))
    return FileList.sort(key=lambda entries: int(re.findall(r'\d+', entries)[0]))


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


def analyseImage(rawImage, calibrationTupel):
    """
    This function is the main analysis function. It takes an image together with the calibration parameter to calculate
    various information from the high energy electrons spectrometer.
    :param rawImage: image matrix as doubles.
    :param calibrationTupel: (J, W, pts, E, dxoverdE, BackgroundImage, L, CutOff, BackgroundNoise) as a tupel
    Explicit information about the calibrationTupel see function 'createNewCalibrationFiles'
    :return:

    """
    J, W, pts, E, dxoverdE, BackgroundImage, L, CutOff, BackgroundNoise = calibrationTupel
    # preparation of the calibration parameter (later move into the calibration prep above)
    # The cut off is an index, which indicates, where the calibration between image plate and camera image did not work.
    # This was due to a camera artifact (an imaging problem with the perplex glass)
    Energy = E.T[self.CutOff:, :]
    # the following parameters will be cut only if they are intended to be used.
    #    Length = L[:, self.CutOff:]
    #    dxoverdE = dxoverdE[:, self.CutOff:]  # MIGHT BE WRONG
    #    BackgroundImage = BckgndImage[:, self.CutOff:]
    BackgroundNoise = BackgroundNoise[:, self.CutOff:]

    # here hard coded: the analysis of the image plates yield to a count to fC calibration of:
    fCperCounts = 2.7e-3
    fCperCountsSigma = 6.5e-4

    # now to the analysis:
    image = rawImage.astype(float)
    WarpedImageWithoutBckgnd = imageTransformation(image, calibrationTupel)
    # correctly integrating the spectrum and change the units from counts/mm -> counts/MeV
    Spectrum = electronSpectrum(WarpedImageWithoutBckgnd, dxoverdE, L)
    ChargeInCounts = np.trapz(Spectrum.T, E.T)  # correct integration of the entire image
    Charge = ChargeInCounts * fCperCounts  # Charge in fC
    Spectrum = Spectrum[:, CutOff:]  # cutting off the low energy bit as described above
    Spectrum = Spectrum * fCperCounts  # changing the units from counts/MeV -> fC/MeV
    BackgroundStd_SigmaLevel = np.sum(BackgroundNoise, axis=0) * SignificanceLevel
    cutoffEnergy95 = determine95percentCharge(Energy, Spectrum, BackgroundStd_SigmaLevel)
    totalEnergy = determineTotalEnergy(Energy, Spectrum, BackgroundStd_SigmaLevel)
    return Spectrum, Charge, totalEnergy, cutoffEnergy95


def imageTransformation(image, calibrationTupel):
    J, _, pts, _, _, BackgroundImage, _, CutOff, _ = calibrationTupel
    WarpedImage, __ = four_point_transform(image, pts, J)  # warp the image
    WarpedImage = np.fliplr(WarpedImage)  # the axis is flipped (high and low energy)
    WarpedImage = WarpedImage[:, CutOff:]  # as mentioned above, the low energy part needs to be removed due artifacts
    WarpedImageWithoutBckgnd = WarpedImage - BackgroundImage  # background subtraction
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
    Mask = np.greater(Spectrum[:, 0], BckStd_SigmaLevel)
    if any(Mask) is True:
        TmpSpectrum[Mask, 0] = Spectrum[Mask, 0]
        TmpSpectrum = TmpSpectrum / np.trapz(TmpSpectrum.T, E.T)
        CumTrapzMinus1IndexPrep = (TmpSpectrum[:-1, 0] + TmpSpectrum[1:, 0]) / 2 * np.diff(E, axis=0)[:, 0]
        CumTrapzMinus1 = np.cumsum(CumTrapzMinus1IndexPrep)
        MaskCumSum = np.append(np.array([True]), np.array([CumTrapzMinus1 < 0.95]))
        MaximumEnergy = np.amax(E[MaskCumSum, :])
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
    Mask = np.greater(Spectrum[:, 0], BckStd_SigmaLevel)
    if any(Mask) is True:
        TotalEnergy = np.trapz(np.multiply(np.multiply(E[:, 0], Mask), Spectrum[:, 0]), E[:, 0])
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
    FileList = TupelOfFiles(imageFolderPath)
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
        FileList = TupelOfFiles(backgroundImagesPath)
        testBackground = ImportImageFiles([FileList[0]])
        if testImage.shape[0] != testBackground.shape[0]:
            backgroundPath = os.path.join(basePath, 'MIRAGE', 'HighESpec', '%d' % runDate, 'darkfield02')
            if not os.path.exists(backgroundPath):
                print(
                    'WARNING E2: No darkfields with the right image dimensions were found. The calibration database '
                    'will be updated, but the background image is missing.')
            else:
                backgroundImagesPath = simpleFolderFinder(backgroundPath)
                FileList = TupelOfFiles(backgroundImagesPath)
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
        if BackgroundImage != 0:
            BackgroundImage, BackgroundNoise = backgroundImages(backgroundImagesPath, J, pts)
        #  now loading the right magnet map
        FileLocation = os.path.join(basePath, 'Calibrations', 'HighESpec')
        E, dxoverdE = getEnergyTransformation(FileLocation, L)

        totalCalibrationFilePath = os.path.join(calPath, 'HighEspec', '%d' % runDate)
        if not os.path.exists(totalCalibrationFilePath):
            os.mkdir(totalCalibrationFilePath)
        simpleRunName = runName[9:]
        calFile = os.path.join(totalCalibrationFilePath, simpleRunName)
        calibrationTupel = (J, W, pts, E, dxoverdE, BackgroundImage, L, CutOff, BackgroundNoise)
        np.save(calFile, calibrationTupel)
        #  the entry for the database is a relative path:
        relPathForDatabase = os.path.join('HighESpec', '%d' % runDate, '%s.npy' % simpleRunName)
        changeFileEntry(relPathForDatabase, runName, calPath)
    else:
        if os.path.exists(calFile):
            os.remove(calFile)
            changeFileEntry('', runName, calPath)


def getEnergyTransformation(FileLocation, L):
    ImageWarpingFile = 'PositionVsEnergy.mat'
    ImageWarpingVariable = ['Screen', 'EnergyOnAverage']
    ScreenEnergyOnAverage = loadMatFile(FileLocation, ImageWarpingFile, ImageWarpingVariable)
    Screen, EnergyOnAverage = ScreenEnergyOnAverage
    Screen = Screen * 1e3  # in mm
    poly3 = np.poly1d(np.polyfit(Screen[0, :], EnergyOnAverage[0, :], 3))
    E = poly3(L)
    dpoly3 = np.polyder(poly3)
    dEoverdx = dpoly3(L)
    dxoverdE = 1 / dEoverdx
    return E, dxoverdE


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
        FileList = TupelOfFiles(Path)
        BackgroundImages = ImportImageFiles(FileList)
        BackgroundImage = np.mean(BackgroundImages, axis=2)
        BackgroundNoise = np.std(BackgroundImages, axis=2)
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


'''
<- End the new code here
------------------------------------------------------------------------------------------------------------------------
Old code below ->
'''


def determineHighEnergyCutOff(E, Spectrum, BckStd_SigmaLevel):
    """
    The Background is subtracted from the spectrum thus the noise level equals zero.
    However the std of the noise should be determined and then a sigma level of that should be set to estimate the high
    energy cut off.
    Note: BckStd_SigmaLevel has the dimension (num,) which means that Spectrum, having the dimension (num,1) needs to be
    taken
    """
    Mask = np.greater(Spectrum[:, 0], BckStd_SigmaLevel)
    if any(Mask) is True:
        MaximumEnergy = np.amax(E[Mask])
    else:
        MaximumEnergy = 0
    return MaximumEnergy


def map_function_AverageCutOff(data):
    data = np.array(data)
    Mask = data > 0
    NullShots = sum(data == 0)
    if NullShots is data.shape[0]:
        EnergyAv = 0
        EnergyStd = None
    elif NullShots is data.shape[0] - 1:
        EnergyAv = data[Mask]
        EnergyStd = None
    else:
        EnergyAv = np.mean(data[Mask])
        EnergyStd = np.std(data[Mask]) / len(data[Mask])
    return EnergyAv, EnergyStd, NullShots


def uint16ToDoubleJ2(J2, UintConversion):
    J2 = J2.astype(float) / UintConversion[0] * UintConversion[1] + UintConversion[2]
    return J2


def determineHighEnergyCutOff(E, Spectrum, BckStd_SigmaLevel):
    """
    The Background is subtracted from the spectrum thus the noise level equals zero.
    However the std of the noise should be determined and then a sigma level of that should be set to estimate the high
    energy cut off.
    Note: BckStd_SigmaLevel has the dimension (num,) which means that Spectrum, having the dimension (num,1) needs to be taken
    """
    Mask = np.greater(Spectrum[:, 0], BckStd_SigmaLevel)
    if any(Mask) is True:
        MaximumEnergy = np.amax(E[Mask])
    else:
        MaximumEnergy = 0
    return MaximumEnergy


def map_function_AverageCutOff(data):
    data = np.array(data)
    Mask = data > 0
    NullShots = sum(data == 0)
    if NullShots == data.shape[0]:
        EnergyAv = 0
        EnergyStd = None
    elif NullShots == data.shape[0] - 1:
        EnergyAv = data[Mask]
        EnergyStd = None
    else:
        print('No Nullshots!')
        EnergyAv = np.mean(data[Mask])
        EnergyStd = np.std(data[Mask]) / len(data[Mask])
    return EnergyAv, EnergyStd, NullShots


def returnEnergyCutOff(RunName):
    HighESpecClass = Espec(RunName, ta2_hrr_2019.utils.DATA_FOLDER, 'HighESpec')
    BackgroundNoise = HighESpecClass.BackgroundNoise * HighESpecClass.fCperCounts

    def map_function_CutOffEnergy(data):
        Spectrum, ____ = HighESpecClass.SpectrumFromImagefC(data)
        SignificanceLevel = 5
        MaximumEnergy = determineHighEnergyCutOff(HighESpecClass.Energy, (Spectrum.T),
                                                  np.sum(BackgroundNoise, axis=0) * SignificanceLevel)
        return MaximumEnergy

    pipeline_CutOff = mirage_analysis.DataPipeline('HighESpec', map_function_CutOffEnergy, map_function_AverageCutOff)
    return pipeline_CutOff.run(RunName)


def returnTotalEnergy(RunName):
    HighESpecClass = Espec(RunName, ta2_hrr_2019.utils.DATA_FOLDER, 'HighESpec')
    BackgroundNoise = HighESpecClass.BackgroundNoise * HighESpecClass.fCperCounts

    def map_function_TotalEnergy(data):
        Spectrum, ____ = HighESpecClass.SpectrumFromImagefC(data)
        SignificanceLevel = 5
        TotalEnergy = determineTotalEnergy(HighESpecClass.Energy, (Spectrum.T),
                                           np.sum(BackgroundNoise, axis=0) * SignificanceLevel)
        return TotalEnergy

    pipeline_TotalEnergy = mirage_analysis.DataPipeline('HighESpec', map_function_TotalEnergy,
                                                        map_function_AverageCutOff)
    return pipeline_TotalEnergy.run(RunName)


'''
def getCalibrationParameter(CalibrationFilePath):
    """
    The calibration files in the folder defined by "CalibrationFilePath" can be loaded with this function.
    The output is a tupel, containing all important parameter:
    J               Jacobian matrix for the correct normalisation of the warping of the espec image
    W               Width of the camera image in mm
    pts             The four points, which are used to warp the image and make it flat
    E               Energy information of the length of the lanex screen
    dxoverdE        For later to change the units of the lanex screen between mm to MeV
    BckgndImage     Taken previous to subtract from the image
    L               Length of the lanex. This is the energy dependend axis. In mm.
    CutOff          Beginning of the lanex screen has imaging artifacts and doesn't contain real data. Its simply cut.
    BackgroundNoise Std of the background images (to put the noise level of the image into statistic relevant context.
    """
    ImageWarpingFile = 'CalibrationParamters.mat'
    ImageWarpingVariable = ['Jacobian', 'Length', 'Width', 'pts', 'BckgndImage', 'CutOff', 'BackgroundNoise']
    CompleteSetVar = loadMatFile(CalibrationFilePath, ImageWarpingFile, ImageWarpingVariable)
    J, L, W, pts, BckgndImage, CutOff, BackgroundNoise = CompleteSetVar
    ImageWarpingFile = 'PositionVsEnergy.mat'
    ImageWarpingVariable = ['Screen', 'EnergyOnAverage']
    ScreenEnergyOnAverage = loadMatFile(CalibrationFilePath, ImageWarpingFile, ImageWarpingVariable)
    Screen, EnergyOnAverage = ScreenEnergyOnAverage
    Screen = Screen * 1e3  # in mm
    poly3 = np.poly1d(np.polyfit(Screen[0, :], EnergyOnAverage[0, :], 3))
    E = poly3(L)
    dpoly3 = np.polyder(poly3)
    dEoverdx = dpoly3(L)
    dxoverdE = 1 / dEoverdx
    return J, W, pts, E, dxoverdE, BckgndImage, L, CutOff, BackgroundNoise


def getCalibrationParameter(CalibrationIndicator):
    """
    The calibration indicator contains of a letter, describing if the images referring to this date is a cropped image
    and a date, which represent a specific background image.
    The output is a tupel, containing all important parameter:
    J               Jacobian matrix for the correct normalisation of the warping of the espec image
    W               Width of the camera image in mm
    pts             The four points, which are used to warp the image and make it flat
    E               Energy information of the length of the lanex screen
    dxoverdE        For later to change the units of the lanex screen between mm to MeV
    BckgndImage     Taken previous to subtract from the image
    L               Length of the lanex. This is the energy dependend axis. In mm.
    CutOff          Beginning of the lanex screen has imaging artifacts and doesn't contain real data. Its simply cut.
    BackgroundNoise Std of the background images (to put the noise level of the image into statistic relevant context.
    """
    ImageWarpingFile = 'CalibrationParamters.mat'
    ImageWarpingVariable = ['Jacobian', 'Length', 'Width', 'pts', 'BckgndImage', 'CutOff', 'BackgroundNoise']
    CompleteSetVar = loadMatFile(CalibrationFilePath, ImageWarpingFile, ImageWarpingVariable)
    J, L, W, pts, BckgndImage, CutOff, BackgroundNoise = CompleteSetVar
    ImageWarpingFile = 'PositionVsEnergy.mat'
    ImageWarpingVariable = ['Screen', 'EnergyOnAverage']
    ScreenEnergyOnAverage = loadMatFile(CalibrationFilePath, ImageWarpingFile, ImageWarpingVariable)
    Screen, EnergyOnAverage = ScreenEnergyOnAverage
    Screen = Screen * 1e3  # in mm
    poly3 = np.poly1d(np.polyfit(Screen[0, :], EnergyOnAverage[0, :], 3))
    E = poly3(L)
    dpoly3 = np.polyder(poly3)
    dEoverdx = dpoly3(L)
    dxoverdE = 1 / dEoverdx
    return J, W, pts, E, dxoverdE, BckgndImage, L, CutOff, BackgroundNoise
'''
