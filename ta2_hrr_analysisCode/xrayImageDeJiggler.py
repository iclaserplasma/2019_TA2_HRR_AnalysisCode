import numpy as np
from scipy.ndimage import median_filter as mf
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

def xcorrImages(imgTest,imgRef,corMfSize):
    imgCor = mf(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(imgTest)*np.conj(np.fft.fft2(imgRef))))),corMfSize)
    y0_ind,x0_ind = np.unravel_index(imgCor.argmax(), imgCor.shape)
    return x0_ind,y0_ind

def alignImageToRef(imgTest,xRoll,yRoll):
    Ny,Nx = np.shape(imgTest)
    imgTest = np.roll(imgTest,yRoll, axis=0)
    imgTest = np.roll(imgTest,xRoll, axis=1)
    return imgTest

def flatCorrectImage(img,darkfield,flatfield):
    return (img-darkfield)/(flatfield-darkfield)

class xrayDeJiggler:
    def __init__(self,imgList=None,bounds=[[-50,50],[-50,50]],bkgImg =bkgImg, flatImg = flatImg):
        self.imgRef = None
        self.compRegion = None
        self.beamRegion = None
        self.bounds = bounds
        self.bkgImg = bkgImg
        self.flatImg = flatImg
        if bounds is not None:
            X,Y = np.meshgrid(np.arange(-50,51),np.arange(-50,51))
            X=X.reshape(-1,1)
            Y=Y.reshape(-1,1)
            self.XY_test = np.concatenate((X,Y),axis=1)
        self.imgList = imgList
        if imgList is not None:
            self.Ny,self.Nx = np.shape(imgList[0])
            self.X,self.Y = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny))

                
        length_scale=[]
        length_scale_bounds=[]
        for n in range(2):
            length_scale.append(10)
            length_scale_bounds.append([1,100])

        kernel =2**2 * RBF(length_scale=length_scale,length_scale_bounds=length_scale_bounds)
        kernel.k1.constant_value_bounds = (0.001,100)
        kernel = kernel+WhiteKernel()
        
        self.model = GP(kernel)

                
            


    def imgRollDiff(self,img,x):
        return np.mean(((self.imgRef - alignImageToRef(img,int(x[1]),int(x[0])))[self.compRegion])**2)

    def findCenterByGP(self,img):
        model = self.model
        XY_test = self.XY_test
        nDims = 2
        nTest = 20
        x_samples = []
        y_samples = []
        for n in range(0,nTest):
            x_test=[]
            if n<5:
                for nD in range(nDims):
                    r = max(bounds[nD]) - min(bounds[nD])
                    x_test.append(np.random.rand()*r+min(bounds[nD]))
            else:
                y_pred,var_pred = model.predict(XY_test,return_std=True)
                iTest = np.argmin(y_pred-2*var_pred)
                for nD in range(nDims):
                    x_test.append(XY_test[iTest,nD])

            y_val = self.imgRollDiff(img,x_test)
            x_samples.append(x_test)
            y_samples.append(y_val)
            model.fit(x_samples,y_samples)

        y_pred = model.predict(XY_test,return_std=False)
        iTest = np.argmin(y_pred-2*var_pred)
        x_opt = []
        for nD in range(nDims):
            x_opt.append(XY_test[iTest,nD])
        return x_opt

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
                imgComb.append(img)
                x_rot.append(0)
                y_rot.append(0)
            elif imgCounts[sList[n]]>imgMeanThresh:
                imgRef = mf(np.mean(imgComb,axis=0),mfSize)
                self.imgReg = imgRef
                x0_ind,y0_ind = self.indCenterByGP(img)
                xRoll = int(x0_ind)
                yRoll = int(y0_ind)
                x_rot.append(xRoll)
                y_rot.append(yRoll)
                img = alignImageToRef(img,xRoll,yRoll)
                imgComb.append(img)
        
        return imgComb
