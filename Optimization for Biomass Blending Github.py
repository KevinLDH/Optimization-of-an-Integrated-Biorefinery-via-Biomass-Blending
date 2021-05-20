#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:32:50 2021

@author: dahuiliu
"""

#Code to solve Optimization Biomass Blending Model

from gurobipy import *
import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

#Set and Index Setup
equipmentIndex, equipmentIndexBlend, moistureIndex, biomassIndex, scenarioIndex = 13, 12, 3, 4, 400
#Number of equipment before pelleting machine (include pelleting machine), Number of equipment after pelleting machine, Number of moisture levels, Number of Biomass types, and Number of scenarios. 
taulength = 30 #Biochemical conversion checking period (Tau)
timeIndex = 960 #Total time period length (T)
tauIndex = math.ceil(float(timeIndex)/float(taulength)) #Total Tau Period Length
grinderSet = [2,6] #Set of grinder indices in the process system
meteringbinSet = [11] #Set of metering bin indices in the process system
storeageSet = [] #Set of storage bin (for different biomass) indices in the process system
for i in range(biomassIndex):
    storeageSet.append(1+i*3)
equipmentRemain = [] #Set of equipment other than some specific ones (grinders, metering bin, equipment linked with bypass)
for i in range(1, equipmentIndex+1):
    if i not in grinderSet and i not in meteringbinSet and i != 3 and i != 10: #3 and 10 are indices of bypass equipments
        equipmentRemain.append(i)
Msb = 1000 #Large value

#Previous Equipment I_i
previousEquipments = [] #Set of predecessors of equipment before pelleting machine (include pelleting machine)
for i in range(equipmentIndex+1):
    if i == meteringbinSet[0]: #Predecessors of metering bin
        previousEquipments.append([9, 10])
    elif i == 10: #Predecessors of bypass equipment
        previousEquipments.append([grinderSet[0]])
    else: #Predecessors of other equipment
        previousEquipments.append([i-1])
previousEquipmentsBlend = [] #Set of predecessors of equipment after pelleting machine
for i in range(equipmentIndexBlend):
    if i % 3 == 0:
        previousEquipmentsBlend.append([equipmentIndex])
    else:
        previousEquipmentsBlend.append([i-1])
previousEquipmentsReactor = [] #Set of predecessors of reactor
for i in range(biomassIndex):
    previousEquipmentsReactor.append(2+i*3)



#Parameter Define
parameterDictionary = { #Dictionary to store Input Information
                       'SLengthWidthHeight' : [], #l,w,h
                       'DDensity' : [], #d_sb
                       'TAverageDensity' : [], #d_isb
                       'DAverageDensityBlend' : [], #d_is after pelleting machine
                       'TEquipmentCapacity' : [], #u_isb
                       'DEquipmentCapacityBlend' : [], #u_is after pelleting machine
                       'DNumberOfBales' : [], #n_sb
                       'SDryMatterLoss' : [], #mu_i
                       'DBypassRatio' : [], #theta_sb
                       'SMassCapacityInventory' : [], #m_i for metering bin
                       'SMassCapacityInventoryBlend' : [], #m_i for storage bins
                       'SVolumnCapacityInventory' : [], #v_i for metering bin
                       'SVolumnCapacityInventoryBlend' : [], #v_i for storage bins
                       'DProcessingTimeOfBale' : [], #p_sb
                       'SAshContent' : [], #a_b
                       'SThermalContent' : [], #e_b
                       'SCarbohydrateContent' : [], #f_b
                       'STargetAshThermalCarbohydrate' : [], #a^*, e^*, f^*
                       'SReactorUpperLowerBigM' : []} #R^, R_, M
ashCarboDictionary = {'Pash' : [], #Dictionary to store scenario information
                      'Pcarbo' : []}



#Read Parameter Data
file = open(r'/home/dahuil/BBP chance constraint/Input.txt')
listInput = file.readlines()
tempList = []
for i in range(len(listInput)):
    if listInput[i][0] == 'S' or listInput[i][0] == 'D' or listInput[i][0] == 'T':
        keyFirst = listInput[i][0] #Read dictionary keys
        keyName = listInput[i].strip()
    else:
        if keyFirst == 'S': 
            rowList = listInput[i].split()
            for j in range(len(rowList)):
                parameterDictionary[keyName].append(rowList[j])
        elif keyFirst == 'D': #Read 2D list
            parameterDictionary[keyName].append(listInput[i].split())
        else: #Read 3D list
            if listInput[i][0] == 'N':
                parameterDictionary[keyName].append(copy.deepcopy(tempList))
                tempList[:] = []
            else:
                tempList.append(listInput[i].split())
file.close()
processTimeOfBaleIni = copy.deepcopy(parameterDictionary['DProcessingTimeOfBale'])
timeIndexIni = timeIndex

fileAC = open(r'/home/dahuil/BBP chance constraint/carbohydrate 1.txt')
listInputAC = fileAC.readlines()
for i in range(len(listInputAC)):
    if listInputAC[i][0] == 'P':
        keyNameAC = listInputAC[i].strip()
    else:
        ashCarboDictionary[keyNameAC].append(listInputAC[i].split())
fileAC.close()

#Sequences (Problems 1-6)
sequenceList = [[0,2],[0,2],[0,2],[1,2],[1,2],[1,2],[1,2],[1,2],[2,2],[2,2],[0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],\
                [0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],[0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],\
                [0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],[0,3],[0,3],[0,3],[1,3],[1,3],[1,3],[1,3],[1,3],[2,3],[2,3],\
                [0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0],[1,0],[2,0],[2,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0],[1,0],[2,0],[2,0]]
    
'''
sequenceList = [[0,2],[0,2],[0,2],[1,2],[1,2],[1,2],[1,2],[1,2],[2,2],[2,2],[0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],\
                [0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],[0,3],[0,3],[0,3],[1,3],[1,3],[1,3],[1,3],[1,3],[2,3],[2,3],\
                [0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],[0,1],[0,1],[0,1],[1,1],[1,1],[1,1],[1,1],[1,1],[2,1],[2,1],\
                [0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0],[1,0],[2,0],[2,0],[0,0],[0,0],[0,0],[1,0],[1,0],[1,0],[1,0],[1,0],[2,0],[2,0]]
'''
'''
sequenceList = [[0,2],[0,2],[0,1],[1,2],[1,2],[1,1],[1,1],[1,1],[2,2],[2,1],[0,3],[0,1],[0,1],[1,2],[1,2],[1,1],[1,1],[1,1],[2,2],[2,1],\
                [0,2],[0,1],[0,1],[1,3],[1,2],[1,1],[1,1],[1,1],[2,3],[2,1],[0,1],[0,1],[0,0],[1,1],[1,1],[1,1],[1,0],[1,0],[2,1],[2,0],\
                [0,3],[0,0],[0,0],[1,1],[1,1],[1,1],[1,0],[1,0],[2,1],[2,0],[0,3],[0,1],[0,1],[1,3],[1,3],[1,1],[1,0],[1,0],[2,1],[2,1],\
                [0,1],[0,1],[0,0],[1,3],[1,3],[1,0],[1,0],[1,0],[2,3],[2,0],[0,1],[0,0],[0,0],[1,1],[1,1],[1,1],[1,1],[1,0],[2,1],[2,0]]
'''
'''
sequenceList = [[1,2],[1,2],[1,1],[1,1],[1,1],[0,2],[0,2],[0,1],[2,2],[2,1],[1,2],[1,2],[1,1],[1,1],[1,1],[0,3],[0,1],[0,1],[2,2],[2,1],\
                [1,3],[1,2],[1,1],[1,1],[1,1],[0,2],[0,1],[0,1],[2,3],[2,1],[1,1],[1,1],[1,1],[1,0],[1,0],[0,1],[0,1],[0,0],[2,1],[2,0],\
                [1,1],[1,1],[1,1],[1,0],[1,0],[0,3],[0,0],[0,0],[2,1],[2,0],[1,3],[1,3],[1,1],[1,0],[1,0],[0,3],[0,1],[0,1],[2,1],[2,1],\
                [1,3],[1,3],[1,0],[1,0],[1,0],[0,1],[0,1],[0,0],[2,3],[2,0],[1,1],[1,1],[1,1],[1,1],[1,0],[0,1],[0,0],[0,0],[2,1],[2,0]]
'''
'''
sequenceList = [[2,2],[2,1],[0,2],[0,2],[0,1],[1,2],[1,2],[1,1],[1,1],[1,1],[2,2],[2,1],[0,3],[0,1],[0,1],[1,2],[1,2],[1,1],[1,1],[1,1],\
                [2,3],[2,1],[0,2],[0,1],[0,1],[1,3],[1,2],[1,1],[1,1],[1,1],[2,1],[2,0],[0,1],[0,1],[0,0],[1,1],[1,1],[1,1],[1,0],[1,0],\
                [2,1],[2,0],[0,3],[0,0],[0,0],[1,1],[1,1],[1,1],[1,0],[1,0],[2,1],[2,1],[0,3],[0,1],[0,1],[1,3],[1,3],[1,1],[1,0],[1,0],\
                [2,3],[2,0],[0,1],[0,1],[0,0],[1,3],[1,3],[1,0],[1,0],[1,0],[2,1],[2,0],[0,1],[0,0],[0,0],[1,1],[1,1],[1,1],[1,1],[1,0]]
'''
'''
sequenceList = [[2, 1], [1, 1], [1, 2], [0, 1], [1, 1], [1, 1], [2, 0], [1, 1], [0, 1], [1, 0], [1, 2], [1, 0], [0, 0], [0, 1], [1, 1], [0, 1], [2, 3], [1, 1], [1, 1], [1, 3],\
                [1, 0], [0, 1], [0, 1], [2, 0], [1, 0], [1, 1], [0, 0], [0, 0], [2, 2], [1, 3], [0, 3], [0, 2], [1, 2], [0, 0], [1, 3], [0, 0], [1, 1], [1, 0], [1, 0], [1, 1],\
                [0, 3], [1, 1], [1, 1], [1, 2], [1, 3], [1, 1], [0, 1], [2, 1], [1, 1], [1, 2], [2, 1], [0, 1], [1, 1], [2, 3], [2, 1], [2, 2], [1, 0], [0, 2], [0, 0], [1, 1],\
                [2, 1], [2, 0], [1, 1], [1, 0], [2, 0], [2, 1], [1, 1], [2, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [2, 1], [0, 3], [1, 1], [0, 1], [1, 1], [0, 2], [1, 3]]
'''


#Initialize Dispatching
startAtTime = []
for s in range(moistureIndex):
    startAtTimeRowCol = []
    for b in range(biomassIndex):
        startAtTimeRow = []
        for t in range(timeIndex):
            startAtTimeRow.append(0)
        startAtTimeRowCol.append(startAtTimeRow)
    startAtTime.append(startAtTimeRowCol)
currentTime = 0
for i in sequenceList:
    startAtTime[i[0]][i[1]][currentTime] = 1
    currentTime += int(parameterDictionary['DProcessingTimeOfBale'][i[0]][i[1]])

#Initialize Other Parameters
volumnOfBale = float(parameterDictionary['SLengthWidthHeight'][0]) * float(parameterDictionary['SLengthWidthHeight'][1]) * float(parameterDictionary['SLengthWidthHeight'][2]) #Bale volume
countSum = 2

alphaLower = [0.0 for tau in range(tauIndex)]
alphaUpper = [10000.0 for tau in range(tauIndex)]
deltaV = 0.1
epsiloV = 0.1
gammaRV = 0.05 #Experimental Risk Value

usedTimeIndex = 0
judge = 1


#Model
while judge: #Shrink p_sb
    alpha = [(alphaLower[tau] + alphaUpper[tau]) / 2 for tau in range(tauIndex)]
    while countSum !=0: #Binary Search
    
        #Define Model
        model = Model()
        #model.Params.MIPGap=0.001
        model.Params.MIPGap=0.0005 #MIP termination Gap

        #Define Variables
        outflowOfEquipment = model.addVars(equipmentIndex+1, moistureIndex, biomassIndex, timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #X_isbt
        outflowOfEquipmentBlend = model.addVars(equipmentIndexBlend, moistureIndex, timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #X_ist for equipment after pelleting machine
        inventoryLevel = model.addVars(meteringbinSet, moistureIndex, biomassIndex, timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #M_isbt for metering bin
        inventoryLevelBlend = model.addVars(storeageSet, moistureIndex, timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #M_ist for storage bins
        velocityInFlow = model.addVars([0], moistureIndex, biomassIndex, timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #V_0sbt
        processAtTime = model.addVars(moistureIndex, biomassIndex, timeIndex, vtype=GRB.BINARY) #Z_sbt
        processAtTimeO = model.addVars(timeIndex, vtype=GRB.BINARY) #hat{Z}_t
        upperReactor = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS) #U
        upperReactorL = model.addVars(timeIndex, lb=0.0, vtype=GRB.CONTINUOUS) #underline{q}*U
        betaPlus = model.addVars(tauIndex, scenarioIndex, lb=0.0, vtype=GRB.CONTINUOUS) #beta+
        betaMinus = model.addVars(tauIndex, scenarioIndex, lb=0.0, vtype=GRB.CONTINUOUS) #beta-
        #Non-negative constraints are also defined here
    
        #Set Objective
        objective = quicksum(processAtTime[s,b,t] for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex)) + quicksum(processAtTimeO[t] for t in range(timeIndex)) + quicksum(alpha[tau] * betaMinus[tau,j] for tau in range(tauIndex) for j in range(scenarioIndex)) #Minimize Processing Time
        model.setObjective(objective, GRB.MINIMIZE)
        
        #Set Constraints
        #Capacity
        #Equipment Capacity
        model.addConstrs(outflowOfEquipment[1,s,b,t] <= float(parameterDictionary['TEquipmentCapacity'][0][s][b]) * processAtTime[s,b,t] for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipment[i,s,b,t] <= float(parameterDictionary['TEquipmentCapacity'][i-1][s][b]) for i in grinderSet for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(quicksum(outflowOfEquipment[13,s,b,t] for s in range(moistureIndex) for b in range(biomassIndex)) <= quicksum(float(parameterDictionary['TEquipmentCapacity'][12][s][b]) * processAtTime[s,b,t] for s in range(moistureIndex) for b in range(biomassIndex)) for t in range(timeIndex))
        model.addConstrs(quicksum(outflowOfEquipmentBlend[i,s,t] for s in range(moistureIndex)) <= float(parameterDictionary['DEquipmentCapacityBlend'][i][0]) * processAtTimeO[t] for i in previousEquipmentsReactor for t in range(timeIndex))
        #Inventory Capacity
        model.addConstrs(quicksum(inventoryLevel[i,s,b,t] for s in range(moistureIndex) for b in range(biomassIndex)) <= float(parameterDictionary['SMassCapacityInventory'][0]) for i in meteringbinSet for t in range(timeIndex))
        model.addConstrs(quicksum(inventoryLevelBlend[i,s,t] for s in range(moistureIndex)) <= float(parameterDictionary['SMassCapacityInventoryBlend'][int((i-1)/3)]) for i in storeageSet for t in range(timeIndex))
        model.addConstrs(quicksum(inventoryLevel[i,s,b,t] / float(parameterDictionary['TAverageDensity'][0][s][b]) for s in range(moistureIndex) for b in range(biomassIndex)) <= float(parameterDictionary['SVolumnCapacityInventory'][0]) for i in meteringbinSet for t in range(timeIndex))
        model.addConstrs(quicksum(inventoryLevelBlend[i,s,t] / float(parameterDictionary['DAverageDensityBlend'][int((i-1)/3)][s]) for s in range(moistureIndex)) <= float(parameterDictionary['SVolumnCapacityInventoryBlend'][int((i-1)/3)]) for i in storeageSet for t in range(timeIndex))
        #Reactor Utilization
        model.addConstrs(quicksum(outflowOfEquipmentBlend[i,s,t] for i in previousEquipmentsReactor for s in range(moistureIndex)) <= upperReactor for t in range(timeIndex))
        model.addConstrs(quicksum(outflowOfEquipmentBlend[i,s,t] for i in previousEquipmentsReactor for s in range(moistureIndex)) >= 0.9 * upperReactorL[t] for t in range(timeIndex))
        model.addConstr(quicksum(outflowOfEquipmentBlend[i,s,t] for i in previousEquipmentsReactor for s in range(moistureIndex) for t in range(timeIndex)) >= 0.95 * quicksum(upperReactorL[t] for t in range(timeIndex)))
            
        #Operational
        model.addConstrs(quicksum(outflowOfEquipment[0,s,b,t] for t in range(timeIndex)) - volumnOfBale * float(parameterDictionary['DDensity'][s][b]) * float(parameterDictionary['DNumberOfBales'][s][b]) <= 0 for s in range(moistureIndex) for b in range(biomassIndex))
        model.addConstrs(quicksum((float(parameterDictionary['STargetAshThermalCarbohydrate'][2]) - float(ashCarboDictionary['Pcarbo'][int((i-2)/3)][j])) * outflowOfEquipmentBlend[i,s,t] for i in previousEquipmentsReactor for s in range(moistureIndex) for t in range(tau*taulength,min((tau+1)*taulength,timeIndex))) + betaPlus[tau,j] - betaMinus[tau,j] == 0 for tau in range(tauIndex) for j in range(scenarioIndex)) #Biochemical Conversion
        model.addConstr(quicksum(inventoryLevel[i,s,b,timeIndex-1] for i in meteringbinSet for s in range(moistureIndex) for b in range(biomassIndex)) + quicksum(inventoryLevelBlend[i,s,timeIndex-1] for i in storeageSet for s in range(moistureIndex)) <= 0)
        model.addConstrs(quicksum(velocityInFlow[0,s,b,tp] for tp in range(t,t+int(parameterDictionary['DProcessingTimeOfBale'][s][b]))) <=  float(parameterDictionary['SLengthWidthHeight'][0]) + (1-startAtTime[s][b][t]) * Msb for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex-int(parameterDictionary['DProcessingTimeOfBale'][s][b])+1))
        model.addConstrs(quicksum(velocityInFlow[0,s,b,tp] for tp in range(t,t+int(parameterDictionary['DProcessingTimeOfBale'][s][b]))) >=  float(parameterDictionary['SLengthWidthHeight'][0]) - (1-startAtTime[s][b][t]) * Msb for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex-int(parameterDictionary['DProcessingTimeOfBale'][s][b])+1))
        model.addConstrs(outflowOfEquipment[0,s,b,t] == (float(parameterDictionary['SLengthWidthHeight'][1]) * float(parameterDictionary['SLengthWidthHeight'][2]) * float(parameterDictionary['DDensity'][s][b])) * velocityInFlow[0,s,b,t] for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        #Judge if a bale with s, b is under processing, no other biomass bale with s', b' can process at time t.
        model.addConstrs(quicksum(processAtTime[sp,bp,tp] for sp in range(moistureIndex) for bp in range(biomassIndex) for tp in range(t,t+int(parameterDictionary['DProcessingTimeOfBale'][s][b]))) - quicksum(processAtTime[s,b,tp] for tp in range(t,t+int(parameterDictionary['DProcessingTimeOfBale'][s][b]))) <= (1 - startAtTime[s][b][t]) * Msb for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex-int(parameterDictionary['DProcessingTimeOfBale'][s][b])+1))
        model.addConstrs(processAtTimeO[t] <= processAtTimeO[t-1] for t in range(1,timeIndex))
        
        #Flow Balance
        model.addConstrs(outflowOfEquipment[i,s,b,t] == quicksum(outflowOfEquipment[p,s,b,t] for p in previousEquipments[i]) for i in equipmentRemain for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipment[i,s,b,t] == quicksum((1-float(parameterDictionary['SDryMatterLoss'][int((i-2)/4)])) * outflowOfEquipment[p,s,b,t] for p in previousEquipments[i]) for i in grinderSet for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipment[3,s,b,t] == quicksum((1-float(parameterDictionary['DBypassRatio'][s][b])) * outflowOfEquipment[p,s,b,t] for p in previousEquipments[3]) for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipment[10,s,b,t] == quicksum(float(parameterDictionary['DBypassRatio'][s][b]) * outflowOfEquipment[p,s,b,t] for p in previousEquipments[10]) for s in range(moistureIndex) for b in range(biomassIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipmentBlend[i,s,t] == quicksum(outflowOfEquipmentBlend[p,s,t] for p in previousEquipmentsBlend[i]) for i in previousEquipmentsReactor for s in range(moistureIndex) for t in range(timeIndex))
        model.addConstrs(outflowOfEquipmentBlend[i,s,t] == quicksum(outflowOfEquipment[p,s,i/3,t] for p in previousEquipmentsBlend[i]) for i in range(0,equipmentIndexBlend,3) for s in range(moistureIndex) for t in range(timeIndex))
        
        #Inventory Balance
        model.addConstrs(inventoryLevel[i,s,b,t] == inventoryLevel[i,s,b,t-1] + quicksum(outflowOfEquipment[p,s,b,t] for p in previousEquipments[i]) - outflowOfEquipment[i,s,b,t] for i in meteringbinSet for s in range(moistureIndex) for b in range(biomassIndex) for t in range(1,timeIndex))
        model.addConstrs(inventoryLevelBlend[i,s,t] == inventoryLevelBlend[i,s,t-1] + quicksum(outflowOfEquipmentBlend[p,s,t] for p in previousEquipmentsBlend[i]) - outflowOfEquipmentBlend[i,s,t] for i in storeageSet for s in range(moistureIndex) for t in range(1,timeIndex))
        model.addConstrs(inventoryLevel[i,s,b,0] == 0 + quicksum(outflowOfEquipment[p,s,b,0] for p in previousEquipments[i]) - outflowOfEquipment[i,s,b,0] for i in meteringbinSet for s in range(moistureIndex) for b in range(biomassIndex))
        model.addConstrs(inventoryLevelBlend[i,s,0] == 0 + quicksum(outflowOfEquipmentBlend[p,s,0] for p in previousEquipmentsBlend[i]) - outflowOfEquipmentBlend[i,s,0] for i in storeageSet for s in range(moistureIndex))
        
        #Linear
        model.addConstrs(upperReactorL[t] >= 0 for t in range(timeIndex))
        model.addConstrs(upperReactorL[t] <= 1 * processAtTimeO[t] for t in range(timeIndex)) #1 here is used as a large value M
        model.addConstrs(upperReactorL[t] <= upperReactor for t in range(timeIndex))
        model.addConstrs(upperReactorL[t] >= upperReactor + 1 * (processAtTimeO[t] - 1) for t in range(timeIndex))
        
        #Solve Optimization
        model.optimize()
        
        
        #Report Result Data and adjust p_sb
        yList = []
        for s in range(moistureIndex):
            for b in range(biomassIndex):
                for t in range(timeIndex):
                    if startAtTime[s][b][t] >= 1 - pow(10,-6):
                        yList.append([s,b,t])
        print(yList)
        yListSort = sorted(yList, key=lambda x: x[2])
        print(yListSort)
        sumInv = 0
        for s in range(moistureIndex):
            for b in range(biomassIndex):
                sumInv += inventoryLevel[11,s,b,timeIndex-1].x
        for s in range(moistureIndex):
            sumInv += inventoryLevelBlend[1,s,timeIndex-1].x + inventoryLevelBlend[4,s,timeIndex-1].x + inventoryLevelBlend[7,s,timeIndex-1].x + inventoryLevelBlend[10,s,timeIndex-1].x
        print(sumInv)
        print(sum(inventoryLevel[11,s,b,timeIndex-1].x for s in range(moistureIndex) for b in range(biomassIndex)))
        print(sum(inventoryLevelBlend[1,s,timeIndex-1].x for s in range(moistureIndex)))
        count = []
        for i in range(len(yList)):
            count.append(0)
        for i in range(len(yListSort)):
            for t in range(yListSort[i][2],yListSort[i][2]+int(parameterDictionary['DProcessingTimeOfBale'][yListSort[i][0]][yListSort[i][1]])):
                if processAtTime[yListSort[i][0],yListSort[i][1],t].x <= pow(10,-6):
                    count[i] += 1
        print(count)
        
        minReduce = []
        for s in range(moistureIndex):
            minReduceRow = []
            for b in range(biomassIndex):
                minReduceRow.append(1000)
            minReduce.append(minReduceRow)
        countSum = 0
        countTotal,countN,countLoc0,countLoc1 = 0,0,0,0
        for i in range(len(yListSort)):
            if i == 0:
                countN += 1
                countTotal += count[i]
                countLoc0 = yListSort[i][0]
                countLoc1 = yListSort[i][1]
            else:
                if countLoc0 == yListSort[i][0] and countLoc1 == yListSort[i][1]:
                    countN += 1
                    countTotal += count[i]
                else:
                    averageR = int(countTotal/countN)
                    if averageR <= minReduce[yListSort[i-1][0]][yListSort[i-1][1]]:
                        minReduce[yListSort[i-1][0]][yListSort[i-1][1]] = averageR
                    countN = 1
                    countTotal = count[i]
                    countLoc0 = yListSort[i][0]
                    countLoc1 = yListSort[i][1]
        averageR = int(countTotal/countN)
        if averageR <= minReduce[yListSort[len(yListSort)-1][0]][yListSort[len(yListSort)-1][1]]:
            minReduce[yListSort[len(yListSort)-1][0]][yListSort[len(yListSort)-1][1]] = averageR
        for s in range(moistureIndex):
            for b in range(biomassIndex):
                parameterDictionary['DProcessingTimeOfBale'][s][b] = str(int(parameterDictionary['DProcessingTimeOfBale'][s][b])-minReduce[s][b])
                countSum += minReduce[s][b]
        tempSum = 0
        for s in range(moistureIndex):
            for b in range(biomassIndex):
                tempSum += int(parameterDictionary['DNumberOfBales'][s][b]) * int(parameterDictionary['DProcessingTimeOfBale'][s][b])
        totalOutToReactor = 0
        for i in previousEquipmentsReactor:
            for s in range(moistureIndex):
                for t in range(timeIndex):
                    totalOutToReactor += outflowOfEquipmentBlend[i,s,t].x
        print(totalOutToReactor)
        timeIndex = max(int(round(sum(processAtTimeO[t].x for t in range(timeIndex)))),tempSum)
        tauIndex = math.ceil(float(timeIndex)/float(taulength))
        minimumProcessingTime = []
        for s in range(moistureIndex):
            minimumProcessingTimeRow = []
            for b in range(biomassIndex):
                minimumProcessingTimeRow.append(int(parameterDictionary['DProcessingTimeOfBale'][s][b]))
            minimumProcessingTime.append(minimumProcessingTimeRow)
        print(minimumProcessingTime)
        print(tempSum)
        print(timeIndex)
        print(upperReactor.x)
        print(totalOutToReactor/timeIndex)
        #ReInitialize Dispatching
        startAtTime[:] = []
        for s in range(moistureIndex):
            startAtTimeRowCol = []
            for b in range(biomassIndex):
                startAtTimeRow = []
                for t in range(timeIndex):
                    startAtTimeRow.append(0)
                startAtTimeRowCol.append(startAtTimeRow)
            startAtTime.append(startAtTimeRowCol)
        currentTime = 0
        for i in sequenceList:
            startAtTime[i[0]][i[1]][currentTime] = 1
            currentTime += int(parameterDictionary['DProcessingTimeOfBale'][i[0]][i[1]])
    
    
    #Binary Search
    usedTimeIndex = timeIndex
    usedTauIndex = math.ceil(float(usedTimeIndex)/float(taulength))
    countTimeBetaM = [0 for tau in range(usedTauIndex)]
    for tau in range(usedTauIndex):
        for j in range(scenarioIndex):
            if betaMinus[tau,j].x >= pow(10,-7):
                countTimeBetaM[tau] += 1
        if countTimeBetaM[tau] >= gammaRV * scenarioIndex + epsiloV:
            alphaLower[tau] = alpha[tau]
        elif countTimeBetaM[tau] <= gammaRV * scenarioIndex - epsiloV:
            alphaUpper[tau] = alpha[tau]
    #Reinitialize T and p_sb
    timeIndex = timeIndexIni
    tauIndex = math.ceil(float(timeIndex)/float(taulength))
    for s in range(moistureIndex):
        for b in range(biomassIndex):
            parameterDictionary['DProcessingTimeOfBale'][s][b] = processTimeOfBaleIni[s][b]
    #Reinitialize Dispatching
    startAtTime[:] = []
    for s in range(moistureIndex):
        startAtTimeRowCol = []
        for b in range(biomassIndex):
            startAtTimeRow = []
            for t in range(timeIndex):
                startAtTimeRow.append(0)
            startAtTimeRowCol.append(startAtTimeRow)
        startAtTime.append(startAtTimeRowCol)
    currentTime = 0
    for i in sequenceList:
        startAtTime[i[0]][i[1]][currentTime] = 1
        currentTime += int(parameterDictionary['DProcessingTimeOfBale'][i[0]][i[1]])
    #Judge if it meets the termination criteria for Binary Search
    judge = 0
    for tau in range(usedTauIndex):
        if abs(alpha[tau] - (alphaUpper[tau]+alphaLower[tau]) / 2) > deltaV:
            judge = 1
    countSum = 1


#Record Results and Output results to Files for further analysis
timeIndex = usedTimeIndex
yList = []
for s in range(moistureIndex):
    for b in range(biomassIndex):
        for t in range(timeIndex):
            if startAtTime[s][b][t] >= 1 - pow(10,-6):
                yList.append([s,b,t])
print(yList)
yListSort = sorted(yList, key=lambda x: x[2])
print(yListSort)
sumInv = 0
for s in range(moistureIndex):
    for b in range(biomassIndex):
        sumInv += inventoryLevel[11,s,b,timeIndex-1].x
for s in range(moistureIndex):
    sumInv += inventoryLevelBlend[1,s,timeIndex-1].x + inventoryLevelBlend[4,s,timeIndex-1].x + inventoryLevelBlend[7,s,timeIndex-1].x + inventoryLevelBlend[10,s,timeIndex-1].x
print(sumInv)
print(sum(inventoryLevel[11,s,b,timeIndex-1].x for s in range(moistureIndex) for b in range(biomassIndex)))
print(sum(inventoryLevelBlend[1,s,timeIndex-1].x for s in range(moistureIndex)))
count = []
for i in range(len(yList)):
    count.append(0)
for i in range(len(yListSort)):
    for t in range(yListSort[i][2],yListSort[i][2]+int(parameterDictionary['DProcessingTimeOfBale'][yListSort[i][0]][yListSort[i][1]])):
        if processAtTime[yListSort[i][0],yListSort[i][1],t].x <= pow(10,-6):
            count[i] += 1
print(count)
tempSum = 0
for s in range(moistureIndex):
    for b in range(biomassIndex):
        tempSum += int(parameterDictionary['DNumberOfBales'][s][b]) * int(parameterDictionary['DProcessingTimeOfBale'][s][b])
totalOutToReactor = 0
for i in previousEquipmentsReactor:
    for s in range(moistureIndex):
        for t in range(timeIndex):
            totalOutToReactor += outflowOfEquipmentBlend[i,s,t].x
print(totalOutToReactor)
minimumProcessingTime = []
for s in range(moistureIndex):
    minimumProcessingTimeRow = []
    for b in range(biomassIndex):
        minimumProcessingTimeRow.append(int(parameterDictionary['DProcessingTimeOfBale'][s][b]))
    minimumProcessingTime.append(minimumProcessingTimeRow)
print(minimumProcessingTime)
print(tempSum)
print(timeIndex)
print(upperReactor.x)
print(totalOutToReactor/timeIndex)
countScenarioBetaM = [0 for tau in range(tauIndex)]
for tau in range(tauIndex):
    for j in range(scenarioIndex):
        if betaMinus[tau,j].x >= pow(10,-9):
            countScenarioBetaM[tau] += 1
print(countScenarioBetaM)
#Writting Solutions
fileR = open(r'/home/dahuil/BBP chance constraint/ResultR1.txt','w')
fileR.write('Result for Replication 1')
fileR.write('\n')
fileR.write(str(totalOutToReactor))
fileR.write(' ')
fileR.write(str(tempSum))
fileR.write(' ')
fileR.write(str(timeIndex))
fileR.close()
#Writing Solution Data
fileOut = open(r'/home/dahuil/BBP chance constraint/OutputR1.txt','w')
fileOut.write('RInputFlow')
fileOut.write('\n')
for s in range(moistureIndex):
    for b in range(biomassIndex):
        for t in range(timeIndex):
            fileOut.write(str(outflowOfEquipment[0,s,b,t].x))
            fileOut.write(' ')
        fileOut.write('\n')
    fileOut.write('N')
    fileOut.write('\n')
fileOut.write('ROutFlow')
fileOut.write('\n')
for i in previousEquipmentsReactor:
    for s in range(moistureIndex):
        for t in range(timeIndex):
            fileOut.write(str(outflowOfEquipmentBlend[i,s,t].x))
            fileOut.write(' ')
        fileOut.write('\n')
    fileOut.write('N')
    fileOut.write('\n')
fileOut.write('RMetering')
fileOut.write('\n')
for s in range(moistureIndex):
    for b in range(biomassIndex):
        for t in range(timeIndex):
            fileOut.write(str(inventoryLevel[11,s,b,t].x))
            fileOut.write(' ')
        fileOut.write('\n')
    fileOut.write('N')
    fileOut.write('\n')
fileOut.write('RStorages')
fileOut.write('\n')
for i in storeageSet:
    for s in range(moistureIndex):
        for t in range(timeIndex):
            fileOut.write(str(inventoryLevelBlend[i,s,t].x))
            fileOut.write(' ')
        fileOut.write('\n')
    fileOut.write('N')
    fileOut.write('\n')
fileOut.write('Ralpha')
fileOut.write('\n')
for tau in range(tauIndex):
    fileOut.write(str(alpha[tau]))
    fileOut.write(' ')
fileOut.write('\n')
fileOut.write('Rbeta')
fileOut.write('\n')
for tau in range(tauIndex):
    for j in range(scenarioIndex):
        fileOut.write(str(betaMinus[tau,j].x))
        fileOut.write(' ')
    fileOut.write('\n')
fileOut.close()