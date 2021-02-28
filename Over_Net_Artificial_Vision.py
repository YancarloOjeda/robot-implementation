#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:11:17 2020

@author: yan
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:36:24 2020

@author: yan
Network:
    
1->   A--S1---------M"1-_
         | \          \  _
         |  H1        \   _
.4->  CTX             \   _ M'1
         \            \ _    -
         \            \_     -
.6->  x--S2-------M2"-\      -
           \         \\      -
           H2  _ -  D1       -
          _ ------------------
1->   EI-
    
Comments: This is a first simulation with these network in python, with
artificial vision. The network and the simulation are part of my PhD thesis.      
"""


import numpy as np
import random
import os
import pandas as pd
from random import choice
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#from sklearn.linear_model import LinearRegression
#from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import cv2
#from scipy import misc, ndimage
#import scipy
import serial.tools.list_ports
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN)
GPIO.setup(20, GPIO.IN)

#%%-------------READ PARAMETERS------#################
file = open('parameters.txt','r')
arrParameters = file.read().split('\n')
file.close()

#%%-------------FREE PARAMETERS------#################
tau = float(arrParameters[0])
kappa = float(arrParameters[1])
sigma = float(arrParameters[2])
mean = float(arrParameters[3])

alpha = float(arrParameters[4])
beta = float(arrParameters[5])
mu = float(arrParameters[6])
sigmaU = float(arrParameters[7])
#%%-------------SIMULATIONS PARAMETERS-----------------------------------------
total = int(arrParameters[8])
EI = float(arrParameters[9])
save = float(arrParameters[10])
sessionName = arrParameters[11]
CRCriteria = float(arrParameters[19])
CTX = float(arrParameters[20])
totalUnidades = 8
unidades = 12
numero_replicas = 1
aux_RC = 0
aux_Ensayos = 0
#%%-------------SHOW AND PLOT PARAMETERS---------------------------------------
showStimulus = int(arrParameters[12])
showResponse = int(arrParameters[13])
showActivation = int(arrParameters[14])
showTs = int(arrParameters[15])

plotActivation = int(arrParameters[16])
plotStimulus = int(arrParameters[17])
plotWeights = int(arrParameters[18])
#%%-------------WEBCAM PARAMETERS---------------------------------------------- 
cap = cv2.VideoCapture(0)
resolucion = (320,200)
cap.set(3, resolucion[0])
cap.set(4, resolucion[1])
orgText1 = (10,20)
orgText2 = (10,40)
orgText3 = (10,60)
orgText4 = (10,80)
orgText5 = (10,100)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .4
color = (255,255,255)
thickness = 1
#%%-------------FUNCTIONS------------------------------------------------------
def Activacion(Lx, Lx_T1, umbral, unidad, Activacion_T1):
    #Unconditional Activation for M'
    if unidad == '8' and activaciones_T[0] > 0:
        Activacion = activaciones_T[0]
        
    #Unconditional Activation for D unit
    elif unidad == '7' and activaciones_T[0] > 0:
        Activacion = activaciones_T[0]
    
   
    #Conditional Activation    
    else:
        #Reactivation mode
        if Lx >= umbral:
            Activacion = Lx+(tau * Lx_T1)*(1-Lx)
        
        #Deactivation mode
        else:
            Activacion = Lx - kappa*Lx       
    return(Activacion)
    

def Logistica(x):
    Lx = 1/(1+math.e**((-x+mean)/sigma))
    return(Lx)
    
def suma_ponderada(Activaciones, Pesos):
    X = (Activaciones * Pesos).sum(0)
    return(X)


def discrepancia_H(dDt, dHt1, activaciones_T,activaciones_T1):
    dHt = abs(activaciones_T-activaciones_T1) + dDt * (1 - dHt1)
    return(dHt)
 
def discrepancia_D(activaciones_T, activaciones_T1):
    dDt = activaciones_T-activaciones_T1
    return(dDt)

def pij(ext, activaciones_T, pesos_T1):
    p = (activaciones_T * pesos_T1)/ext
    return(p)
    
def regla_Aprendizaje(dt, activaciones_Ti, activaciones_Tj, pesos_T1, p, rjt):
    #Gain
    if dt >= 0.001:
        delta_peso = alpha * activaciones_Tj * dt * p * rjt
        
    #Loss
    else:
        delta_peso = -beta * pesos_T1 * activaciones_Ti * activaciones_Tj
        
    return(delta_peso)

    
#%%-------------START OF SIMULATIONS-------------------------------------------
for aux_Replicas in range(numero_replicas):
    c = 0
    d = 0
    ts = 0
    RC = np.zeros(total)
########################################################
####################----ACTIVACION-----#################
    activaciones_T = np.zeros(unidades)
    activaciones_T[0] = EI                #EI
    activaciones_T[1] = 1                 #A
    activaciones_T[2] = CTX               #CTX    
    activaciones_T[3] = 0.6               #x
    activaciones_T[4:] = 0.00669285092    #NPU
    
    activaciones_T1 = np.zeros(unidades)
    activaciones_T1[0] = EI
    activaciones_T1[1] = 1
    activaciones_T1[2] = CTX
    activaciones_T1[3] = 0.6
    activaciones_T1[4:] = 0.00669285092
########################################################
##################-----PESOS------######################
    pesos_T = np.zeros(12)
    pesos_T[:] = 0.1
    pesos_T1 = np.zeros(12)
    pesos_T1[:] = 0.1
    
    p = np.zeros(unidades)
########################################################
########-----ACTIVACIONES Y PESOS FINALES------#########
    activaciones_Todos = np.zeros((unidades,total))
    pesos_Todos = np.zeros((unidades, total))
########################################################
########-----VALORES INICIALES------####################
    xS1t1 = 0
    xS1t = 0
    xS2t1 = 0
    xS2t = 0
    xHt1 = 0
    xHt = 0
    xDt = 0
    xDt1 = 0
    xRt = 0
    xRt1 = 0
    xMt1 = 0
    xMt = 0
    
    dH1t = 0.01
    dH1t1 = 0
    dH2t = 0.01
    dH2t1 = 0
    dDt = 0.01
    
    xS1t1 = suma_ponderada(activaciones_T1[1:3], pesos_T1[0:2])
    LxS1_T1 = Logistica(xS1t1)
    xS2t1 = suma_ponderada(activaciones_T1[2:4], pesos_T1[2:4])
    LxS2_T1 = Logistica(xS2t1)
    xH1t1 = suma_ponderada(activaciones_T1[4], pesos_T1[4])
    LxH1_T1 = Logistica(xH1t1)
    xH2t1 = suma_ponderada(activaciones_T1[6], pesos_T1[5])
    LxH2_T1 = Logistica(xH2t1)
    xMA1t1 = suma_ponderada(activaciones_T1[4], pesos_T1[6])
    LxMA1_T1 = Logistica(xMA1t1)
    xMA2t1 = suma_ponderada(activaciones_T1[6], pesos_T1[7])
    LxMA2_T1 = Logistica(xMA2t1)
    xDt1 = suma_ponderada(activaciones_T1[8:10], pesos_T1[8:10])
    LxD_T1 = Logistica(xDt1)
    xMOt1 = suma_ponderada(activaciones_T1[8:10], pesos_T1[10:])
    LxMO_T1 = Logistica(xMOt1)

         
#%%-------------ACTIVACIONES---------------------------------------------------
    for aux_Ts in range(total):
        _, frame = cap.read()  
        #BLUE = A
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        lower_blue = np.array([110,50,50]) 
        upper_blue = np.array([130,255,255])  
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue = cv2.bitwise_and(frame,frame, mask= mask_blue)
        
        #GREEN = x
        lower_green = np.array([25, 52, 72]) 
        upper_green = np.array([102,255,255])  
        mask_green = cv2.inRange(hsv, lower_green, upper_green) 
        green = cv2.bitwise_and(frame,frame, mask= mask_green) 
        
        #EI
        if GPIO.input(21):
            activaciones_T[0] = 0
        else:
            activaciones_T[0] = EI
        
        #A
        area_blue = blue[:,:,:]/100
        suma_area_blue = ((np.sum(area_blue)/1000)*.1) 
        if suma_area_blue > 1:
            suma_area_blue = 1
        activaciones_T[3] = suma_area_blue
        
        #CTX
        if GPIO.input(20):
            activaciones_T[2] = CTX
        else:
            activaciones_T[2] = 0
        
        #A
        area_green = green[:,:,:]/100
        suma_area_green = ((np.sum(area_green)/1000)*.1)
        if suma_area_green > 1:
            suma_area_green = 1
        activaciones_T[1] = suma_area_green
        
        total_frame = green + blue + frame#yellow + red + green + blue
        
        #cv2.imshow('frame',frame)
        total_frame = cv2.resize(total_frame, (200,200))
        if showTs == 1:
            cv2.putText(total_frame, 'Ts='+str(aux_Ts),
                    orgText1, font, fontScale, color, thickness)
        if showResponse == 1:
            cv2.putText(total_frame, 'CR count=' + str(aux_RC), orgText2,
                    font, fontScale, color, thickness)
        if showActivation == 1:
            cv2.putText(total_frame, 'Activation R*=' +str(round(activaciones_T[11],3)),
                    orgText3, font, fontScale, color, thickness)
        if showStimulus == 1:
            cv2.putText(total_frame, 'A=' + str(round(activaciones_T[1],3)) + '; CTX='+str(round(activaciones_T[2],3)), orgText4,
                        font, fontScale, color, thickness)
            cv2.putText(total_frame, 'x=' + str(round(activaciones_T[3],3)) + '; UI='+str(round(activaciones_T[0],3)),
                        orgText5, font, fontScale, color, thickness)
            
        
        cv2.imshow('q to quit', total_frame)
           
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                    
        
        Lista_unidad = random.sample('12345678', 8)   
     
        for i in range(totalUnidades):
            unidad = choice(Lista_unidad)
            umbral = np.random.normal(mu, sigmaU, 1)
            Lista_unidad.remove(unidad)
            
            if unidad == '1':#S"1
                xS1t = suma_ponderada(activaciones_T[1:3], pesos_T[0:2])
                LxS1 = Logistica(xS1t)
                activaciones_T[4] = Activacion(LxS1, LxS1_T1, umbral, unidad, activaciones_T1[4])
                LxS1_T1 = LxS1
                activaciones_T1[4] = activaciones_T[4]
            
            if unidad == '2':#H1
                xH1t = suma_ponderada(activaciones_T[4], pesos_T[4])
                LxH1 = Logistica(xH1t)
                activaciones_T[5] = Activacion(LxH1, LxH1_T1, umbral, unidad, activaciones_T1[5])
                LxH1_T1 = LxH1
#                    activaciones_T1[3] = activaciones_T[3] 
              
            if unidad == '3':#S"2
                xS2t = suma_ponderada(activaciones_T[2:4], pesos_T[2:4])
                LxS2 = Logistica(xS2t)
                activaciones_T[6] = Activacion(LxS2, LxS2_T1, umbral, unidad, activaciones_T1[6])
                LxS2_T1 = LxS2
                activaciones_T1[6] = activaciones_T[6]
            
            if unidad == '4':#H2
                xH2t = suma_ponderada(activaciones_T[6], pesos_T[5])
                LxH2 = Logistica(xH2t)
                activaciones_T[7] = Activacion(LxH2, LxH2_T1, umbral, unidad, activaciones_T1[7])
                LxH2_T1 = LxH2
                
            if unidad == '5':#M"1
                xMA1t = suma_ponderada(activaciones_T[4], pesos_T[6])
                LxMA1 = Logistica(xMA1t)
                activaciones_T[8] = Activacion(LxMA1, LxMA1_T1, umbral, unidad, activaciones_T1[8])
                LxMA1_T1 = LxMA1
                activaciones_T1[8] = activaciones_T[8]
            
            if unidad == '6':#M"2
                xMA2t = suma_ponderada(activaciones_T[6], pesos_T[7])
                LxMA2 = Logistica(xMA2t)
                activaciones_T[9] = Activacion(LxMA2, LxMA2_T1, umbral, unidad, activaciones_T1[9])
                LxMA2_T1 = LxMA2
                activaciones_T1[9] = activaciones_T[9]
            
            if unidad == '7':#D
                xDt = suma_ponderada(activaciones_T[8:10], pesos_T[8:10])
                LxD = Logistica(xDt)
                activaciones_T[10] = Activacion(LxD, LxD_T1, umbral, unidad, activaciones_T1[10])
                # print(aux_Ensayos, activaciones_T[10])
                LxD_T1 = LxD 
#                    activaciones_T1[5] = activaciones_T[5]
                
            if unidad == '8':#M'1
                xMOt = suma_ponderada(activaciones_T[8:10], pesos_T[10:])
                LxMO = Logistica(xMOt)
                activaciones_T[11] = Activacion(LxMO, LxMO_T1, umbral, unidad, activaciones_T1[11])
                LxMO_T1 = LxMO
                activaciones_T1[11] = activaciones_T[11] 
                
                
                
                 
#%%-------------PESOS----------------------------------------------------------
        Lista_unidad_pesos = random.sample('123456789abc', 12) 
        dDt = discrepancia_D(activaciones_T[10], activaciones_T1[10])
        dH1t = discrepancia_H(dDt, dH1t1, activaciones_T[5], activaciones_T1[5])
        dH1t1 = dH1t 
        dH2t = discrepancia_H(dDt, dH2t1, activaciones_T[7], activaciones_T1[7])
        dH2t1 = dH2t 
        
        for i in range(unidades):
            unidad = choice(Lista_unidad_pesos)
            Lista_unidad_pesos.remove(unidad)
            
            if unidad == '1':#w1 = A, S"1
                xAS1t = suma_ponderada(activaciones_T[1:3], pesos_T[0:2])
                p[0] = pij(xAS1t, activaciones_T[1], pesos_T1[0])
                rjt = 1- (pesos_T[0:2]).sum(0)
                pesos_T[0] += regla_Aprendizaje(dH1t, activaciones_T[1], activaciones_T[4], pesos_T1[0], p[0], rjt)
                pesos_T1[0] = pesos_T[0]
                
            if unidad == '2':#w2 = CTX, S"1
                xCTXS1t = suma_ponderada(activaciones_T[1:3], pesos_T[0:2])
                p[1] = pij(xCTXS1t, activaciones_T[2], pesos_T1[1])
                rjt = 1- (pesos_T[0:2]).sum(0)
                pesos_T[1] += regla_Aprendizaje(dH1t, activaciones_T[2], activaciones_T[4], pesos_T1[1], p[1], rjt)
                pesos_T1[1] = pesos_T[1]
            
            if unidad == '3':#w3 = CTX, S"2
                xCTXS2t = suma_ponderada(activaciones_T[2:4], pesos_T[2:4])
                p[2] = pij(xCTXS2t, activaciones_T[2], pesos_T1[2])
                rjt = 1- (pesos_T[2:4]).sum(0)
                pesos_T[2] += regla_Aprendizaje(dH2t, activaciones_T[2], activaciones_T[6], pesos_T1[2], p[2], rjt)
                pesos_T1[2] = pesos_T[2]
                
            if unidad == '4':#w4 = x, S"2
                xxS2t = suma_ponderada(activaciones_T[2:4], pesos_T[2:4])
                p[3] = pij(xxS2t, activaciones_T[3], pesos_T1[3])
                rjt = 1- (pesos_T[2:4]).sum(0)
                pesos_T[3] += regla_Aprendizaje(dH2t, activaciones_T[3], activaciones_T[6], pesos_T1[3], p[3], rjt)
                pesos_T1[3] = pesos_T[3]
                
            if unidad == '5':#w5 = S"1, H1
                xS1H1t = suma_ponderada(activaciones_T[4], pesos_T[4])
                p[4] = pij(xS1H1t, activaciones_T[4], pesos_T1[4])
                rjt = 1 - pesos_T[4]
                pesos_T[4] += regla_Aprendizaje(dH1t, activaciones_T[4], activaciones_T[5], pesos_T1[4], p[4], rjt)
                pesos_T1[4] = pesos_T[4]
                
            if unidad == '6':#w6 = S"2, H2
                xS1H1t = suma_ponderada(activaciones_T[6], pesos_T[5])
                p[5] = pij(xS1H1t, activaciones_T[6], pesos_T1[5])
                rjt = 1 - pesos_T[5]
                pesos_T[5] += regla_Aprendizaje(dH2t, activaciones_T[6], activaciones_T[7], pesos_T1[5], p[5], rjt)
                pesos_T1[5] = pesos_T[5]
                
            if unidad == '7':#w7 = S"1, M"1
                xS1MA1t = suma_ponderada(activaciones_T[4], pesos_T[6])
                p[6] = pij(xS1MA1t, activaciones_T[4], pesos_T1[6])
                rjt = 1-pesos_T[6]
                pesos_T[6] += regla_Aprendizaje(dDt, activaciones_T[4], activaciones_T[8], pesos_T1[6], p[6], rjt)
                pesos_T1[6] = pesos_T[6]
            
            if unidad == '8':#w7 = S"2, M"2
                xS2MA2t = suma_ponderada(activaciones_T[6], pesos_T[7])
                p[7] = pij(xS2MA2t, activaciones_T[6], pesos_T1[7])
                rjt = 1-pesos_T[7]
                pesos_T[7] += regla_Aprendizaje(dDt, activaciones_T[6], activaciones_T[9], pesos_T1[7], p[7], rjt)
                pesos_T1[7] = pesos_T[7]
                
            if unidad == '9':#w9 = M"1, D
                xDt = suma_ponderada(activaciones_T[8:10], pesos_T[8:10])
                p[8] = pij(xDt, activaciones_T[8], pesos_T1[8])
                rjt = 1- (pesos_T[8:10]).sum(0)
                pesos_T[8] += regla_Aprendizaje(dDt, activaciones_T[8], activaciones_T[10], pesos_T1[8], p[8], rjt)
                pesos_T1[8] = pesos_T[8]
            
            if unidad == 'a':#w10 = M"2, D
                xDt = suma_ponderada(activaciones_T[8:10], pesos_T[8:10])
                p[9] = pij(xDt, activaciones_T[9], pesos_T1[9])
                rjt = 1- (pesos_T[8:10]).sum(0)
                pesos_T[9] += regla_Aprendizaje(dDt, activaciones_T[9], activaciones_T[10], pesos_T1[9], p[9], rjt)
                pesos_T1[9] = pesos_T[9]
                
            if unidad == 'b':#w11 = M"1, M'1
                xMA1MOt = suma_ponderada(activaciones_T[8:10], pesos_T[10:])
                p[10] = pij(xMA1MOt, activaciones_T[8], pesos_T1[10])
                rjt = 1- (pesos_T[10:]).sum(0)
                pesos_T[10] += regla_Aprendizaje(dDt, activaciones_T[8], activaciones_T[11], pesos_T1[10], p[10], rjt)
                pesos_T1[10] = pesos_T[10]
                
            if unidad == 'c':#w11 = M"2, M'1
                xMA2MOt = suma_ponderada(activaciones_T[8:10], pesos_T[10:])
                p[11] = pij(xMA2MOt, activaciones_T[9], pesos_T1[10])
                rjt = 1- (pesos_T[10:]).sum(0)
                pesos_T[11] += regla_Aprendizaje(dDt, activaciones_T[9], activaciones_T[11], pesos_T1[11], p[11], rjt)
                pesos_T1[11] = pesos_T[11]
                
       
        for i in range(unidades):
            activaciones_Todos[i][c] = activaciones_T[i]
            pesos_Todos[i][c] = pesos_T[i]
        
        c += 1
        
        #print('Activacion: ', activaciones_T[11])
        if activaciones_T[11] > CRCriteria and activaciones_T[11] != activaciones_T[0]:
            aux_RC += 1
            RC[aux_Ts] = activaciones_T[11]
            
            
        
        activaciones_T1[5] = activaciones_T[5]
        activaciones_T1[7] = activaciones_T[7] 
        activaciones_T1[10] = activaciones_T[10] 
        
        # time.sleep()
    
    cap.release()
    GPIO.cleanup()
    #Dev_WebCam_Read_B.release()
    cv2.destroyAllWindows()
#%%---------PLOT----------------------------------------------
if plotActivation == 1:    
    #activaciones_Todos = np.delete(activaciones_Todos, np.where(activaciones_Todos[:][:] == 0), axis = 0)
    plt.subplot(1, 3, 1)
    plt.plot(activaciones_Todos[11][:], ms=7, lw=2, alpha=0.8, label = 'Activation R*')
    plt.plot(RC[:], ms=7, lw=2, alpha=0.8, label = 'RC')
    plt.legend(fontsize = 10)
    plt.ylim(-.005,1.0)
    plt.xlabel('Time steps', fontsize = 15)
    plt.ylabel("Activation level", fontsize = 15)
    
    
if plotStimulus == 1:
    plt.subplot(1, 3, 2)
    plt.plot(activaciones_Todos[1][:], ms=7, lw=2, alpha=0.4, label = 'A')
    plt.plot(activaciones_Todos[2][:], ms=7, lw=2, alpha=0.4, label = 'CTX')
    plt.plot(activaciones_Todos[3][:], ms=7, lw=2, alpha=0.4, label = 'x')
    plt.plot(activaciones_Todos[0][:], ms=7, lw=2, alpha=0.6, label = 'US')
    plt.legend(fontsize = 10)
    plt.ylim(-.005,1.0)
    plt.xlabel('Time steps', fontsize = 15)
    plt.ylabel("Activation level", fontsize = 15)
    
if plotWeights == 1:
    plt.subplot(1, 3, 3)
    plt.plot(pesos_Todos[10][:], ms=7, lw=2, alpha=0.8, label = '$M_1"$''$-R*$')
    plt.plot(pesos_Todos[11][:], ms=7, lw=2, alpha=0.8, label = '$M_2"$''$-R*$')
    plt.ylim(-.005,1.0)
    plt.legend(fontsize = 10)
    plt.xlabel('Time steps', fontsize = 15)
    plt.ylabel('Weights', fontsize = 15)

try:
    plt.show()
except:
    pass

if save  == 1:
    file = open(sessionName + '.cvs', 'w')
    file.write('ts'+','+'A'+','+'CTX'+','+'x'+','+'US'+','+'Act R*'+','+'CR'+'\n')
    for i in range(total):
        file.write(str(i)+','+str(round(activaciones_Todos[1][i],3))+
                   ','+str(round(activaciones_Todos[2][i],3))+
                   ','+str(round(activaciones_Todos[3][i],3))+
                   ','+str(round(activaciones_Todos[0][i],3))+
                   ','+str(round(activaciones_Todos[11][i],3))+
                   ','+str(round(RC[i],3))+','+'\n')
    file.close()