import os
import numpy as np
import cv2
def imdata():
    folder='C:\\Users\\ghait\\Desktop\\data\\eeg_clust\\keras-facenet-master\\data\\lfw'
    I=[]
    V=[]
    for f in os.listdir(folder):
        v=0
        for img in os.listdir(folder+'\\'+f):
            if len(os.listdir(folder+'\\'+f))>=2:
                v=v+1
                i=cv2.resize(cv2.imread(folder+'\\'+f+'\\'+img),(160,160))
                
                if v==2 :
                    V.append(i)
                    break
                else:
                    I.append(i)
        if len(I)==46 and len(V)==46 : break
    I=np.array(I)
    V=np.array(V)
    return I,V

