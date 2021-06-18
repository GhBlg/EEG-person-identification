import numpy as np
import os

def database():
    Y=[]
    B=[]
    C=[]
    D=[]

    #100 sec
    #BONN dataset
    folder='C:\\Users\\ghait\\Desktop\\data\\bonndata'
    f1=0
    for f in os.listdir(folder):
        f1=f1+1
        print(f)
        a=open(folder+'\\'+f,'r')
        a1=a.readlines()
        a.close()
        V=[]    
        for i in a1[0][1:-1].split(','):
            if i[0]=='[' :
                i=i[1:]
            if i[0]==' ' :
                if i[1]=='[':
                    i=i[2:]
            if i[-1]==']':
                i=i[:-1]
            
            V.append(int(i))
            if len(V)==25600:
                Y.append(f1)
                B.append(V)
                C.append(V)
                D.append(V)
                V=[]





    #KSU dataset
    f2=f1
    folder='C:\\Users\\ghait\\Desktop\\data\\ksu'
    for f in os.listdir(folder):
        f2=f2+1
        print(f)
        a=open(folder+'\\'+f,'r')
        a1=a.readlines()
        a.close()
        
        k=0
        try :
            
            while k<100000:
                d=[]
                for i in range(1,17):
                    a3=a1[i]
                    a2=a3[4:-5]
                    L=list()
                    for x in a2.split('\t'):
                        L.append(float(x))    
                    sig=L[k:25600+k]
                    #d.append(sig)
                    
                    if i==6 and len(sig)==25600:
                        B.append(sig)
                        Y.append(f2)
                    if  i==7 and len(sig)==25600:
                        C.append(sig)
                        Y.append(f3)
                    if  i==8 and len(sig)==25600:
                        D.append(sig)
                        Y.append(f3)
                    print(k)
                k=k+25600
        except:
            pass



    f3=f2
    #MIT dataset
    folder='C:\\Users\\ghait\\Desktop\\data\\mit'
    for f in os.listdir(folder):
        f3=f3+1
        print(f)
        a=open(folder+'\\'+f,'r')
        a1=a.readlines()
        a.close()
        
        
        k=0
        try :
            while k<100000:
                d=[]
                for i in range(1,24):
                    a3=a1[i]
                    pos=a3.find('\t')
                    a2=a3[pos+1:-2]
                    L=list()
                    for x in a2.split('\t'):
                        L.append(float(x))
                    sig=L[k:25600+k]
                    #d.append(sig)

                    
                    if  i==6 and len(sig)==25600:
                        B.append(sig)
                        Y.append(f3)
                    if  i==7 and len(sig)==25600:
                        C.append(sig)
                        Y.append(f3)
                    if  i==8 and len(sig)==25600:
                        D.append(sig)
                        Y.append(f3)
                    print(k)
                k=k+25600
        except:
            pass

   
    ##ss=open('X.txt','w')
    ##for i in B1 :
    ##    for j in i :
    ##        ss.write(str(j)+'\n')
    ##
    ##sv=open('Y.txt','w')
    ##for i in Y :
    ##    sv.write(str(i)+' \n')
        
    B1=[]
    for i in B:
        bb=np.array(i).reshape(160,160)
        B1.append(bb)
    BC=[]
    for i in C:
        bb=np.array(i).reshape(160,160)
        BC.append(bb)
    BD=[]
    for i in D:
        bb=np.array(i).reshape(160,160)
        BD.append(bb)

    B2=[B1[:118],BC[:118],BD]
    Y=Y[:118]
    X=np.array(B1[:118])      
    X=X.reshape([-1,160,160])
    del B2,B,B1,bb,C,D,BC,BD
    return X,Y
