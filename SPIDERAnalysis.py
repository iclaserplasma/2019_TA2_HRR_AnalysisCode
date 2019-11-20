import zipfile
import numpy as np
import os

def readSPIDER_temporal_profile(path):
    
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if 'time.dat' in filename:
            d = []
            for i, line in enumerate(z.open(filename)):
                 if not i == 0:
                     # The data is in binary, so we need to decode it to append to arrays
                     d.append([float(line.split(b"\t")[0].decode()), float(line.split(b"\t")[2].decode())])
            d = np.array(d)
            return d[:,0], d[:,1]
    # If it hasn't found the data return arrays of zero
    return np.zeros(10)

def readSPIDER_values(path):
    z = zipfile.ZipFile(path)
    for filename in z.namelist():
        if filename.endswith('values.dat'):
            with z.open(filename) as f:
                data = f.read().decode()
    return data

def extract_file_info(data, key):
    for line in data.splitlines():
        if key in line:
            return float(line.strip().split('\t')[-1])
    raise Exception()

def getSpecPhaseOrders(path):
    data = readSPIDER_values(path)
    GDD = float(extract_file_info(data, 'GDD'))
    TOD = float(extract_file_info(data, 'TOD'))
    FOD = float(extract_file_info(data, 'FOD'))
    return GDD, TOD, FOD

def polyOrders(filePathList):
    
    pOrders = []
    P0_TW_per_J=[]
    for file in filePathList:
        if file.endswith(".zip"):
            
            t,f = readSPIDER_temporal_profile(file)
            # a threshold to throw away junk files (i.e. where signal was too low to get a pulse)
            if np.mean(f)<0.1:
                pOrders.append(getSpecPhaseOrders(file))
            else:
                pOrders.append([np.nan]*3)
            
    return pOrders