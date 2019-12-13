import numpy as np
from scipy.ndimage import median_filter as mf
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from ta2_hrr_analysisCode.gp_opt import BasicOptimiser_discrete  
import sys
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

        self.imgList = imgList
        if imgList is not None:
            self.Ny,self.Nx = np.shape(imgList[0])
            self.X,self.Y = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny))

         
        #kernel = kernel+WhiteKernel()
        
        

            
    def imgRollDiff(self,img,x):
        imgDiff =mf((self.imgRef-shiftImage(img,x))[self.compRegion],(9,9))
   
        return np.mean(imgDiff**2)


    def findCenterByGP(self,img):
               
        length_scale=[]
        length_scale_bounds=[]
        for n in range(2):
            length_scale.append(10)
            length_scale_bounds.append([1,100])

        kernel =2**2 * RBF(length_scale=length_scale,length_scale_bounds=length_scale_bounds)
        kernel.k1.constant_value_bounds = (0.001,100)

        BO = BasicOptimiser_discrete(2, mean_cutoff=None,kernel=kernel, sample_scale=1, maximise_effort=1000, bounds=self.bounds,
            scale=None, use_efficiency=True, fit_white_noise=True)
        bounds = self.bounds
        nDims = 2
        nTest = 100

        for n in range(0,nTest):
            x_test=[]
            if n<5:
                for nD in range(nDims):
                    r = max(bounds[nD]) - min(bounds[nD])
                    x_test.append(int(np.random.rand()*r+min(bounds[nD])))
                x_test = np.array(x_test)
            else:
                x_test = BO.ask(1e-20)

            y_val = self.imgRollDiff(img,x_test)
            BO.tell(x_test,-y_val,y_val*0.002)

        best_pos, best_val = BO.optimum()
        return best_pos

    def alignImgList(self):
        imgList = self.imgList
        imgCounts = np.mean(np.mean(np.array(imgList),axis=2),axis=1)
        sList = np.argsort(imgCounts)[::-1]
        N = len(sList)
        x_rot=[]
        y_rot=[]
        imgComb=[]
        imgRef = None

        imgMeanThresh = 200

        mfSize = [3,3]
        corMfSize = mfSize[1]
        mfSize = mfSize[0]
        for n in range(0,N):
            img = flatCorrectImage(imgList[sList[n]],self.bkgImg,self.flatImg)
            img = img/np.mean(img[self.beamRegion])
            if imgRef is None:
                imgRef = mf(img,mfSize)
                self.imgRef = imgRef
                imgComb.append(img)
                x_rot.append(0)
                y_rot.append(0)
            elif imgCounts[sList[n]]>imgMeanThresh:
                sys.stdout.flush()
                print(' Shot: ', sList[n], ' Number: ', n, '/', N ,'...', end='\r')

    
                #imgRef = mf(np.mean(imgComb,axis=0),mfSize)
                #self.imgRef = imgRef
                x_opt = self.findCenterByGP(mf(img,mfSize))
                x_rot.append(int(x_opt[0]))
                y_rot.append(int(x_opt[1]))
                imgComb.append(shiftImage(img,x_opt))
        sys.stdout.flush()
        print('\r','Done')   
        
        return imgComb, x_rot , y_rot
