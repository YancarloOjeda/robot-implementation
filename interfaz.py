import tkinter as tk
from tkinter import Tk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL.Image

root = Tk()
root.title('Robot implementetion')
root.geometry('700x500+10+10')

def getValues():
    tau = entTau.get()
    kappa = entKappa.get()
    sigma = entSigma.get()
    mean = entMean.get()
    
    alpha = entAlpha.get()
    beta = entBeta.get()
    desvGaussian = entDesvGaussian.get()
    meanGaussian = entMeanGaussian.get()
    
    ts = entTs.get()
    US = entUS.get()
    saveData = varSaveData.get()
    sessionName = entSessionName.get()
    
    showStimulus = varShowStimulus.get()
    showResponse = varShowResponse.get()
    showActivation = varShowActivation.get()
    showTs = varShowTs.get()
    
    plotActivation = varPlotActivation.get()
    plotStimulus = varPlotStimulus.get()
    plotWeights = varPlotWeights.get()
    
    CRCriteria = entCRCriteria.get()
    CTX = entCTX.get()
    
    arrVar =[str(tau), str(kappa), str(sigma), str(mean), str(alpha), str(beta), str(desvGaussian),
             str(meanGaussian), str(ts), str(US), str(saveData), str(sessionName),
             str(showStimulus), str(showResponse), str(showActivation), str(showTs),
             str(plotActivation), str(plotStimulus), str(plotWeights), str(CRCriteria), str(CTX)]

    variablesFile = open('parameters.txt', 'w')
    for i in arrVar:
        variablesFile.write(i+'\n')
    variablesFile.close()
    
    
    #root.destroy()
    os.system("python Over_Net_Artificial_Vision.py")
    #os.system("python /home/pi/Robot/move_output.py")

def setValuesFree():
    entSigma.insert(0,0.1)
    entTau.insert(0,0.1)
    entKappa.insert(0,0.1)
    entMean.insert(0,0.5)
    
    entAlpha.insert(0,0.5)
    entBeta.insert(0,0.1)
    entDesvGaussian.insert(0,0.15)
    entMeanGaussian.insert(0,0.2)

def setValuesSession():
    entTs.insert(0,1000)
    entUS.insert(0,1.0)
    rdbSaveData.select()
    entCRCriteria.insert(0,0.5)
    entCTX.insert(0,0.4)

def setValuesData():
    rdbShowStimulus.select()
    rdbShowResponse.select()
    rdbShowActivation.select()
    rdbShowTs.select()
    
    rdbPlotActivation.select()
    rdbPlotStimulus.select()
    rdbPlotWeights.select()      
#%%----- Canvas -------------------------------
canFreeParameters = tk.Canvas(root, width=290, height=210)
canFreeParameters.create_rectangle(1,1,290,210, width=1)
canFreeParameters.place(x=5, y=258)

canActivationRule = tk.Canvas(root, width=195, height=150)
canActivationRule.create_rectangle(1,1,195,150)
canActivationRule.place(x=15, y=282)

canLearningRule = tk.Canvas(root, width=70, height=150)
canLearningRule.create_rectangle(1,1,70,150, width=1)
canLearningRule.place(x=215, y=282)

canSimulationParameters = tk.Canvas(root, width=170, height=210)
canSimulationParameters.create_rectangle(1,1,170,210, width=1)
canSimulationParameters.place(x=315, y=258)

canShowData = tk.Canvas(root, width=170, height=210)
canShowData.create_rectangle(1,1,170,210, width=1)
canShowData.place(x=505, y=258)
#%%----- Labels -------------------------------
lblNetwork = tk.Label(root, text='Network (3-2[2]-2[1]-1)')
lblNetwork.config(font=15)
lblNetwork.place(x=250, y=10)

lblFreeParameters = tk.Label(root, text='Free parameters')
lblFreeParameters.place(x=10, y=250)

imgNet = tk.PhotoImage(file='red_doctorado_chica.png')
lblNet = tk.Label(root, image=imgNet)
lblNet.place(x=170, y=30)

lblActivation=tk.Label(root, text='Activation')
lblActivation.place(x=23, y=275)

lblLogistic=tk.Label(root, text='Logistic function')
lblLogistic.place(x=25, y=345)

lblTau = tk.Label(root, text='\u03C4')
lblTau.place(x=25, y=295)

lblKappa = tk.Label(root, text='\u03BA')
lblKappa.place(x=25, y=320)

lblSigma = tk.Label(root, text='\u03C3')
lblSigma.place(x=25, y=370)

lblMean = tk.Label(root, text='\u03BC')
lblMean.place(x=25, y=395)

lblLearning=tk.Label(root, text='Learning')
lblLearning.place(x=220, y=275)

lblAlpha=tk.Label(root, text='\u03B1')
lblAlpha.place(x=222, y=295)

lblBeta=tk.Label(root, text='\u03B2')
lblBeta.place(x=222, y=320)

lblGaussian=tk.Label(root, text='Threshold')
lblGaussian.place(x=140, y=345)

lblDesvGaussian=tk.Label(root, text='\u03C3')
lblDesvGaussian.place(x=140, y=370)

lblMeanGaussian = tk.Label(root, text='\u03BC')
lblMeanGaussian.place(x=140, y=395)

lblSimulationParameters = tk.Label(root, text='Simulation parameters')
lblSimulationParameters.place(x=320, y=250)

lblTs = tk.Label(root, text='Total ts')
lblTs.place(x=325, y=295)

lblUS = tk.Label(root, text='US value')
lblUS.place(x=325, y=320)

lblCRCriteria = tk.Label(root, text='CR criteria')
lblCRCriteria.place(x=325, y=345)

lblSessionName = tk.Label(root, text='Name')
lblSessionName.place(x=325, y=395)

lblCTX = tk.Label(root, text='CTX value')
lblCTX.place(x=325, y=370)

lblShowData = tk.Label(root, text='Show data')
lblShowData.place(x=510, y=250)

lblSimulationValues=tk.Label(root, text='Simulation values')
lblSimulationValues.place(x=520, y=275)

lblPlotValues=tk.Label(root, text='Plot values')
lblPlotValues.place(x=520, y=345)
#%%----- Entries -------------------------------
entTau = tk.Entry(root, bd=1, width=5)
entTau.place(x=40, y=295)

entKappa = tk.Entry(root, bd=1, width=5)
entKappa.place(x=40, y=320)

entSigma = tk.Entry(root, bd=1, width=5)
entSigma.place(x=40, y=370)

entMean = tk.Entry(root, bd=1, width=5)
entMean.place(x=40, y=395)

entAlpha = tk.Entry(root, bd = 1, width=5)
entAlpha.place(x=235, y=295)

entBeta = tk.Entry(root, bd = 1, width=5)
entBeta.place(x=235, y=320)

entDesvGaussian = tk.Entry(root, bd = 1, width=5)
entDesvGaussian.place(x=155, y=370)

entMeanGaussian = tk.Entry(root, bd = 1, width=5)
entMeanGaussian.place(x=155, y=395)

entTs = tk.Entry(root, bd = 1, width=8)
entTs.place(x=400, y=295)

entUS = tk.Entry(root, bd = 1, width=8)
entUS.place(x=400, y=320)

entSessionName = tk.Entry(root, bd = 1, width=8)
entSessionName.place(x=400, y=395)

entCRCriteria = tk.Entry(root, bd = 1, width=8)
entCRCriteria.place(x=400, y=345)

entCTX = tk.Entry(root, bd = 1, width=8)
entCTX.place(x=400, y=370)
#%%----- Checkbuttons----------------------------
varSaveData = tk.IntVar()
varShowStimulus = tk.IntVar()
varShowResponse = tk.IntVar()
varShowActivation = tk.IntVar()
varShowTs = tk.IntVar()

varPlotStimulus = tk.IntVar()
varPlotActivation = tk.IntVar()
varPlotWeights = tk.IntVar()
rdbSaveData = tk.Checkbutton(root, text=' Save data', variable=varSaveData, onvalue=1, offvalue=0)
rdbSaveData.place(x=320, y=270)

rdbShowStimulus = tk.Checkbutton(root, text=' Stimulus', variable=varShowStimulus, onvalue=1, offvalue=0)
rdbShowStimulus.place(x=520, y=295)

rdbShowResponse = tk.Checkbutton(root, text=' CR', variable=varShowResponse, onvalue=1, offvalue=0)
rdbShowResponse.place(x=520, y=320)

rdbShowActivation = tk.Checkbutton(root, text=' R*', variable=varShowActivation, onvalue=1, offvalue=0)
rdbShowActivation.place(x=600, y=295)

rdbShowTs = tk.Checkbutton(root, text=' Ts', variable=varShowTs, onvalue=1, offvalue=0)
rdbShowTs.place(x=600, y=320)

rdbPlotStimulus = tk.Checkbutton(root, text=' Stimulus', variable=varPlotStimulus, onvalue=1, offvalue=0)
rdbPlotStimulus.place(x=520, y=370)

rdbPlotActivation = tk.Checkbutton(root, text=' Activation', variable=varPlotActivation, onvalue=1, offvalue=0)
rdbPlotActivation.place(x=520, y=395)

rdbPlotWeights = tk.Checkbutton(root, text=' Weights', variable=varPlotWeights, onvalue=1, offvalue=0)
rdbPlotWeights.place(x=520, y=420)
#%%----- Buttons -------------------------------
btnDefaultFree = tk.Button(root, text='Default values', bd=0, command=setValuesFree)
btnDefaultFree.place(x=15, y=440)

btnDefaultSession = tk.Button(root, text='Default values', bd=0, command=setValuesSession)
btnDefaultSession.place(x=325, y=440)

btnDefaultData = tk.Button(root, text='Default values', bd=0, command=setValuesData)
btnDefaultData.place(x=525, y=440)

btnRunExperiment = tk.Button(root, text='Run', bd=0, command=getValues)
btnRunExperiment.place(x=40, y=470)

setValuesFree()
setValuesSession()
setValuesData()

root.mainloop()
