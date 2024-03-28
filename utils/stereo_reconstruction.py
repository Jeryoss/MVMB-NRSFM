
import os
import sys
import numpy as np
import torch
import datetime
from torch import nn
import pandas as pd
import plotly.offline as py
import matplotlib.pyplot as plt
import glob
from numpy import linalg as LA


def NormalizePoints(pts):
  NUM=pts.shape[0]
    
  centroid=np.mean(pts, axis=0)  
  Td=np.eye(3)

  Td[0:2,2]=Td[0:2,2]-centroid.T
  newPts=Td@np.append(pts.T, np.ones((1,NUM)), 0)
  
  newPts=newPts[0:2,:]
  s=np.sqrt(2.0)/np.sqrt(np.trace(newPts@newPts.T)/NUM)  

  #print('s', s)
  Ts=np.array([[s,0,0],[0,s,0],[0,0,1]])
  newPts=np.array([[s,0],[0,s]])@newPts
  
  T=Ts@Td
  newPts=newPts.T

  return newPts,T


def computeFund8Points(data2D_a, data2D_b):
    F = []

   
    # loads the points of the stereo
    # Kep1 is the index of the first image of the stereo
    # Kep2 is the index of the second image of the stereo
    # 3d structure reconstruction is made by using of theese two picture
    # the calibration for theese two images are needed as well (C1, C2)
    numPoints = data2D_a.shape[0]  
   
    if (numPoints < 8):
        print("ComputeFund8Points: At least eight points needed!")
        return None

    xa=data2D_a[:, 0]     # loads the picture coordinates for the appropriate two images from the data2D array
    ya=data2D_a[:, 1]
    xb=data2D_b[:, 0]
    yb=data2D_b[:, 1]

    aa = np.zeros((numPoints, 9))

    # computation of the elemenets of matrix 'a' ( a*f=0 )
    for i in range(numPoints):
        aa[i,0]=xb[i]*xa[i]
        aa[i,1]=xb[i]*ya[i]
        aa[i,2]=xb[i]
        aa[i,3]=yb[i]*xa[i]
        aa[i,4]=yb[i]*ya[i]
        aa[i,5]=yb[i]
        aa[i,6]=xa[i]
        aa[i,7]=ya[i]
        aa[i,8]=1
    
        
    v,d=LA.eig(aa.T@aa)                         # az a*f=0 egyenlet megoldasanak szamitasa

    ei = np.argmin(v)
    f = d[:, ei]

    F=f.reshape(3,3)                            # a fundamentalis matrix szamitasa
    
    return F

def decomposeEssential(E):

    #Bontsuk fel
    U,D,V=LA.svd(E)
    
    Ecorr=U@np.eye(3)@V.T
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z=np.array([[0,-1,0],[1,0,0],[0,0,0]])

    tx=U@Z@U.T
    
    R1=U@W@V
    R2=U@W.T@V
    
    if LA.det(R1)<0:
        R1=-R1
        
    if LA.det(R2)<0:
        R2=-R2

    t1=np.array([tx[2,1], tx[0,2], tx[1, 0]])
    t2=-t1

    return t1,t2,R1,R2,tx
    
    
def null(a, rtol=1e-5):
    u, s, v = LA.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()
    
    
def ComputeFundamentalFromProjections(P1,P2):
    _, e1=null(P1)


    e2=P2@e1


    e2x=np.array([[0,-e2[2, 0],e2[1, 0]],[e2[2, 0],0,-e2[0, 0]],[-e2[1, 0],e2[0, 0],0]])

    
    F=e2x@P2@LA.pinv(P1)
    
    return F

def calculateS(t,a,b,c,d,f1,f2):
	return t*t/(1+f1*f1*t*t)+((c*t+d)**2)/((a*t+b)**2+f2*f2*((c*t+d)**2))


def HartleySturmTriangulation(F,u1,v1):

    T1=np.array([[1,0,-u1[0]],[0,1,-u1[1]],[0,0,1]])

    T2=np.array([[1,0,-v1[0]],[0,1,-v1[1]],[0,0,1]])

    F2=LA.inv(T2.T)@F@LA.inv(T1)
    
    tmpU,tmpS,tmpV=LA.svd(F2)
    
    e1=tmpV[2,:].T
    e2=tmpU[:,2]
    
    s1=e1[0]**2+e1[1]**2
    s2=e2[0]**2+e2[1]**2

    R1=np.array([[e1[0],e1[1],0],[-e1[1],e1[0],0],[0,0,1]])
    R2=np.array([[-e2[0],-e2[1],0],[e2[1],-e2[0],0],[0,0,1]])

    F3=R2@F2@R1.T

    f1=e1[2]
    f2=e2[2]

    a=F3[1,1]
    b=F3[1,2]
    c=F3[2,1]
    d=F3[2,2]

	#Create polinomial:
    t6=-a*c*(f1**4)*(a*d-b*c)
    t5=(a*a+f2*f2*c*c)**2-(a*d+b*c)*(f1**4)*(a*d-b*c)
    t4=2*(a*a+f2*f2*c*c)*(2*a*b+2*c*d*f2*f2)-d*b*(f1**4)*(a*d-b*c)-2*a*c*f1*f1*(a*d-b*c)
    t3=(2*a*b+2*c*d*f2*f2)**2+2*(a*a+f2*f2*c*c)*(b*b+f2*f2*d*d)-2*f1*f1*(a*d-b*c)*(a*d+b*c)
    t2=2*(2*a*b+2*c*d*f2*f2)*(b*b+f2*f2*d*d)-2*(f1*f1*a*d-f1*f1*b*c)*b*d-a*c*(a*d-b*c)
    t1=(b*b+f2*f2*d*d)**2-(a*d+b*c)*(a*d-b*c)
    t0=-(a*d-b*c)*b*d

    r=[t6,t5,t4,t3,t2,t1,t0]

    bestS=float("inf")
    rs=np.roots(r)

    for r in rs:		
        if np.isreal(r):
            val=calculateS(r,a,b,c,d,f1,f2)
            if val<bestS:
                bestS=val
                bestT=r



    #	bestS
    valInf=1/(f1**2)+(c**2)/(a**2+f2**2*c**2)
    assert valInf>=bestS

    point1=np.array([[0],[bestT],[1]])
    line2=F3@point1
    point2=np.array([-line2[0]*line2[2],-line2[1]*line2[2],line2[0]**2+line2[1]**2])
    point2=point2/point2[2]

    u2=LA.inv(R1@T1)@point1
    
    v2=LA.inv(R2@T2)@point2
    
    return u2,v2

def LinearTriangulationEigen(P1,u1,v1,P2,u2,v2):

    A = np.vstack([u1[0]*P1[2] - P1[0],
                        v1[0]*P1[2] - P1[1],
                        u2[0]*P2[2] - P2[0],
                        v2[0]*P2[2] - P2[1]])

    d, v = LA.eig(A.T@A)

    ei = np.argmin(d)
    eivalue = d[ei]
    
    eivec		= v[:, ei]
    ATA			= A.T@A

    pt3D 		= (eivec / eivec[3])
    pt3D        = pt3D[0:3]

    return pt3D
	


def TriangluateBundle(pts1,pts2,K,R1,R2,t1,t2):
    NUMP=pts1.shape[0]

    P1=K@np.append(np.eye(3),np.zeros((3,1)), 1)

    P2s = []
        
    P2s.append(K@np.append(R1,np.expand_dims(t1, 1), 1))
    P2s.append(K@np.append(R1,np.expand_dims(t2, 1), 1))      
    P2s.append(K@np.append(R2,np.expand_dims(t1, 1), 1))
    P2s.append(K@np.append(R2,np.expand_dims(t2, 1), 1))    

    Rs=[R1, R1, R2, R2]        
    ts=[t1, t2, t1, t2]
        
    #Distinguishion ia not required as only the signs can differenct.
    Fs = [ComputeFundamentalFromProjections(P1,s) for s in P2s]            

    counter=np.zeros((4,1))    
    pts3D_pc=np.zeros((4, NUMP,3))    
    pts3D_normals=np.zeros((4, NUMP,3))
    
    print('NUMP', NUMP)

    for idx in range(NUMP):    
        for cam in range(4):
            p1=pts1[idx]
            p2=pts2[idx]
            P2=P2s[cam]
            R2=Rs[cam]
            t2=ts[cam]
            p1prev=p1
            p2prev=p2
            p1,p2=HartleySturmTriangulation(Fs[cam],p1,p2);
            pt3DFirst=LinearTriangulationEigen(P1,p1[0],p1[1],P2,p2[0],p2[1])
            
            pt3DSecond=Rs[cam]@pt3DFirst+ts[cam]

            #Reprojection error (for debug):
            '''
%            
            res1=P1*[pt3DFirst;1.0];
            
            res1=res1/res1(3);
            err1=norm(res1(1:2)-[p1prev(1);p1prev(2)]);
            
            tmp2=P2*[pt3DFirst;1.0];
            tmp2=tmp2/tmp2(3);
            res2=K*[eye(3),zeros(3,1)]*[pt3DSecond;1.0];
            res2=res2/res2(3);
            err2=norm(res2(1:2)-[p2prev(1);p2prev(2)]);
            
            debugW(1:2,idx)=res1(1:2);
            debugW(3:4,idx)=res2(1:2);
            
            '''
            if pt3DFirst[2]>0.0 and pt3DSecond[2]>0.0:
                counter[cam]=counter[cam]+1
            
            pts3D_pc[cam, idx,:]=pt3DFirst.T              
        
        bestCam = np.argmax(counter)
        bestValue = counter[bestCam]
    
    return pts3D_pc,Rs,ts,bestCam
