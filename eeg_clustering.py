import numpy as np
import os
from database_gen import database
from image_dataset import imdata
from new_net import new_net

I,V=imdata()
X,Y = database()
print(len(X))
print(len(Y))


I=I.reshape([-1,1,160,160,3])
V=V.reshape([-1,1,160,160,3])

pathw='C:\\Users\\ghait\\Desktop\\data\\eeg_clust\\keras-facenet-master\\model\\keras\\facenet_keras_weights.h5'
from inception_resnet_v1 import InceptionResNetV1
model=InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8,
                      weights_path=pathw)
embeddings=[]
for i in I:
    embeddings.append(model.predict(i))
embeddings=np.array(embeddings)


NN=new_net(X,embeddings,Y)
##eeg_emb=[]
##X=X.reshape([-1,160,160])
##for i in X:
##    i=i.reshape([-1,160,160])
##    eeg_emb.append(NN.predict(i))
##X_test=[]
##y_test=[]
##X_train=[]
##y_train=[]
##for i in range(0,117):
##    if Y[i]==Y[i+1]:
##        X_train.append(eeg_emb[i])
##        y_train.append(Y[i])
##    else:
##        X_test.append(eeg_emb[i])
##        y_test.append(Y[i])
##
##X_train.append(eeg_emb[-1])
##y_train.append(Y[-1])
##
##X_train=np.array(X_train)
##X_test=np.array(X_test)
##print('********************')
##print(len(X_train))
##print(len(X_test))
##print('********************')
##from sklearn import svm
##
##clf = svm.SVC(kernel='rbf', probability=True).fit(X_train.reshape([-1,128]), y_train)
##print('training acc=')
##print(str(clf.score(X_train.reshape([-1,128]), y_train))+'\n')
##print('validation acc=')
##print(str(clf.score(X_test.reshape([-1,128]), y_test))+'\n')









