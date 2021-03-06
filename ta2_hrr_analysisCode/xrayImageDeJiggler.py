import numpy as np
from scipy.ndimage import median_filter as mf
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from ta2_hrr_analysisCode.gp_opt import BasicOptimiser_discrete  
import sys


import mirage_analysis

import ta2_hrr_2019.utils as utils
utils.setup_mirage_analysis()
def avgCounts(data):
    return np.mean(data.astype(float)) # Return the sum of the pixels in each image
def avgBurst(data):
    return np.mean(data,axis=0) # Return the sum of the pixels in each image

def null_fcn(data):
    return data# Return the sum of the pixels in each image

xray_pipeline = mirage_analysis.DataPipeline('XRay', null_fcn, null_fcn)
xray_bkg_pipe = mirage_analysis.DataPipeline('XRay', null_fcn, avgBurst)
xray_meanCounts_pipe = mirage_analysis.DataPipeline('XRay', avgCounts, avgBurst)


def xcorrImages(imgTest,imgRef,corMfSize):
    imgCor = mf(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(imgTest)*np.conj(np.fft.fft2(imgRef))))),corMfSize)
    y0_ind,x0_ind = np.unravel_index(imgCor.argmax(), imgCor.shape)
    return x0_ind,y0_ind

def shiftImage(imgTest,x):
    Ny,Nx = np.shape(imgTest)
    imgTest = np.roll(imgTest,int(x[1]), axis=0)
    imgTest = np.roll(imgTest,int(x[0]), axis=1)
    return imgTest

def flatCorrectImage(img,darkfield,flatfield):
    return (img-darkfield)/(flatfield-darkfield)



class xrayDeJiggler:
    def __init__(self,imgList=None,bounds=[[-50,50],[-50,50]],bkgImg=None, flatImg = None):
        self.imgRef = None
        self.compRegion = None
        self.beamRegion = None
        self.bounds = bounds
        self.bkgImg = bkgImg
        self.flatImg = flatImg
        self.imgComb =  None
        self.imgList = imgList
        if imgList is not None:
            self.Ny,self.Nx = np.shape(imgList[0])
            self.X,self.Y = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny))

         
        #kernel = kernel+WhiteKernel()
        
        

            
    def imgRollDiff(self,img,x):
        imgDiff =mf((self.imgRef-shiftImage(img,x))[self.compRegion],(3,3))
        dRMS = np.sqrt(np.mean(imgDiff**2))
        y = 1e1*np.exp(-(3*dRMS)**2)
        return y

    def modelGridMax(self,BO,gridLims):
        x = np.arange(gridLims[0][0],gridLims[0][1]+1)
        y = np.arange(gridLims[1][0],gridLims[1][1]+1)
        X,Y = np.meshgrid(x,y)
        X=X.reshape(-1,1)
        Y=Y.reshape(-1,1)
        XY_test = np.concatenate((X,Y),axis=1)
        Z_pred=[]
        for xy in XY_test:
            Z_pred.append(BO.model.submodel_samples.predict(xy.reshape(-1,2)))

        Z_pred =np.array(Z_pred)
        i_opt = np.argmax(Z_pred.flatten())
        x_opt = [X[i_opt],Y[i_opt]]
        return x_opt

    def findCenterByGP(self,img):
               
        length_scale=[]
        length_scale_bounds=[]
        for n in range(2):
            length_scale.append(10)
            length_scale_bounds.append([1,1000])

        kernel =2**2 * RBF(length_scale=length_scale,length_scale_bounds=length_scale_bounds)
        kernel.k1.constant_value_bounds = (0.001,10000)

        BO = BasicOptimiser_discrete(2, mean_cutoff=None,kernel=kernel, sample_scale=1, maximise_effort=100, bounds=self.bounds,
            scale=None, use_efficiency=True, fit_white_noise=True)

        BO.model.submodel_samples.kernel.k2.noise_level_bounds = [1e-10,10]
        bounds = self.bounds
        nDims = 2
        nTest = 50

        for n in range(0,nTest):
            x_test=[]
            if n<10:
                for nD in range(nDims):
                    r = max(bounds[nD]) - min(bounds[nD])
                    x_test.append(int(np.random.rand()*r+min(bounds[nD])))
                x_test = np.array(x_test)
            else:
                x_test = BO.ask(1e-20)
                if x_test is None:
                    x_test = []
                    for nD in range(nDims):
                        r = max(bounds[nD]) - min(bounds[nD])
                        x_test.append(int(np.random.rand()*r+min(bounds[nD])))
                    x_test = np.array(x_test)

            y_val = self.imgRollDiff(img,x_test)
            BO.tell(x_test,y_val,y_val*0.01)
        BO.maximise_effort=10000
        best_pos, best_val = BO.optimum()
        gridLims = [[(-10+best_pos[0]),(10+best_pos[0])],[(-10+best_pos[1]),(10+best_pos[1])]]
        x_opt = self.modelGridMax(BO,gridLims)
        return x_opt

    def alignImgList(self,use_existing_ref=False):
        imgList = self.imgList
        imgCounts = np.mean(np.mean(np.array(imgList),axis=2),axis=1)
        sList = np.argsort(imgCounts)[::-1]
        N = len(sList)
        x_rot=[]
        y_rot=[]
        imgComb=[]
        imgRef = None
        buildRef = True
        if use_existing_ref:
            if self.imgRef is not None:
                imgRef = self.imgRef
                buildRef = False
                
        
            

        imgMeanThresh = 200

        mfSize = [3,3]
        corMfSize = mfSize[1]
        mfSize = mfSize[0]
        for n in range(0,N):
            img = flatCorrectImage(imgList[sList[n]],self.bkgImg,self.flatImg)
            img = img/np.mean(img[self.beamRegion])
            if imgRef is None:
                imgRef = img
                self.imgRef = imgRef
                imgComb.append(img)
                x_rot.append(0)
                y_rot.append(0)
            elif imgCounts[sList[n]]>imgMeanThresh:
                sys.stdout.flush()
                print(' Shot: ', sList[n]+1, ' Number: ', n+1, '/', N ,'...', end='\r')

    
                #imgRef = mf(np.mean(imgComb,axis=0),mfSize)
                #self.imgRef = imgRef
                x_opt = self.findCenterByGP(img)
                x_rot.append(int(x_opt[0]))
                y_rot.append(int(x_opt[1]))
                imgComb.append(shiftImage(img,x_opt))
            
            if buildRef:
                imgRef = np.median(imgComb,axis=0)
                self.imgRef = imgRef
        sys.stdout.flush()
        print('\r','Done')   
        
        return imgComb, x_rot , y_rot

    def alignRun(self,bkgRunName,bkgBurst,flatRunName,flatBurst,imgRunName,imgBurstNumbers):
        

        bNum, bkgImage = xray_bkg_pipe.run(bkgRunName,bkgBurst)
        bNum, flatImage = xray_bkg_pipe.run(flatRunName,flatBurst)

        self.runImages = []
        for bNum in imgBurstNumbers:
            _, radioImages = xray_pipeline.run(imgRunName,bNum)
            self.imgList = radioImages
            imgComb, x_rot , y_rot = self.alignImgList(use_existing_ref=True)
            
            self.runImages = self.runImages +imgComb
        return self.runImages

    def alignProcessedImgList(self,imgList,N_ref,n_start=0):
       
        N = len(imgList)
        x_rot=[]
        y_rot=[]
        if self.imgComb is None:
            self.imgComb=[]
            imgRef = None
        else:
            self.imgComb = self.imgComb[:n_start]
            imgRef = np.median(self.imgComb,axis=0)
    
        mfSize = [3,3]
        corMfSize = mfSize[1]
        mfSize = mfSize[0]
        for n in range(n_start,N):
            img = imgList[n]
            if imgRef is None:
                imgRef = img
                self.imgRef = imgRef
                self.imgComb.append(img)
                x_rot.append(0)
                y_rot.append(0)
            else:
                sys.stdout.flush()
                print(' Image: ', n+1, '/', N ,'...', end='\r')

                x_opt = self.findCenterByGP(img)
                x_rot.append(int(x_opt[0]))
                y_rot.append(int(x_opt[1]))
                self.imgComb.append(shiftImage(img,x_opt))
                if n<N_ref:
                    imgRef = np.median(self.imgComb,axis=0)
                    self.imgRef = imgRef
        sys.stdout.flush()
        print('\r','Done')   
        self.x_rot = x_rot
        self.y_rot = y_rot
        return self.imgComb, x_rot , y_rot



            