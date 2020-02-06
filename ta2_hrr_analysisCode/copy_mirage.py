import os  
import numpy as np
import time
import sys
import re
import shutil
from distutils.dir_util import copy_tree
class MirageCopy():
    def __init__(self,destFolder=None,dataFolder = r'E:\Streeter2019\MIRAGE'):

        self.dataFolder = dataFolder
        self.destFolder = destFolder
    def copyFolder(self,src, dst, update=True):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                try:
                    copy_tree(s, d, update=update)
                    # Depend what you need here to catch the problem
                except OSError as exc: 
                    # File already exist
                    if exc.errno == errno.EEXIST:
                        shutil.copy(src, dst)
                    # The dirtory does not exist
                    if exc.errno == errno.ENOENT:
                        shutil.copy(src, dst)
                    else:
                        raise
            else:
                shutil.copy2(s, d)        

   
    def findDiags(self):
        self.diagList = []
        diagList = next(os.walk(self.dataFolder))[1]
        for d in diagList:
            if d[0] is not '.':                
                self.diagList.append(d)

    def copyRun(self,runStr = None):

        if runStr is None:
            print('Enter a run string i.e.  20190904')
            return None
        print('Finding diagnostics...')
        self.findDiags()
        print(self.diagList)
        for diag in self.diagList:

            sourcePath = os.path.join(self.dataFolder,diag,runStr)
            destPath = os.path.join(self.destFolder,diag,runStr)
            if os.path.exists(sourcePath):
                
                print('Copying ' + diag)
                self.copyFolder(sourcePath,destPath)
                print('Done')
            else:
                print('Nothing to copy for ' + diag)

        print('All files copied' + ' \n')




            





