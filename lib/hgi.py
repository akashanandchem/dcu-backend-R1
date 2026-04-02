# -*- coding: utf-8 -*-

import os
import sys

import math
import joblib	#version 0.13.2
from operator import add
#from dateutil.parser import parse
import pandas as pd	#version 0.25.1
import numpy as np	#version 1.17.1
import csv
import json
import traceback
from statistics import mean

from datetime import datetime
import datetime as dt
import time
import logging #version 0.5.1.2
from logging.handlers import RotatingFileHandler

from HGICalculations import HgiCalculations
from HGICalculations import HgiInputProcessing

os.system('cls' if os.name == 'nt' else 'clear')

dateFormat = '%d-%m-%Y %H:%M'
dateFormatString = 'dd-mm-yyyy hh:mm:ss'

folder = os.path.dirname(os.path.dirname(__file__) )

class module_hgi():
        
    def logData(self, statusText):
        currTime = (datetime.now() - self.startTime)
        print ("{} : {}".format(str(currTime)[:-7], statusText))
        logging.info(statusText)

    def progressBar(self, text, currVal, totalVal, endText=""): 
        print(text + ": ["+ ":"* int(currVal * 100/totalVal) + " " * (100-int(currVal*100/totalVal)) + "] " + str(currVal) + " of " + str(totalVal) , end="\r")
        if currVal == totalVal:
            print("\n", end="")

    def stopScript(self):
        self.logData("## Exiting Script ## \n")
        raise SystemExit

    def __init__(self):
        self.startTime = datetime.now()        
        self.folder = os.path.dirname(os.path.dirname(__file__) )

        self.logFile = os.path.join(self.folder, "runLog.log")
        logging.basicConfig(filename=self.logFile, filemode='a', level=logging.INFO ,format='%(asctime)s: %(message)s', datefmt='%d %b %y %H:%M:%S')
        
        self.logData("## Starting Script ## ")

        self.qcFileName = os.path.join(folder, "output", "qc.csv")
        self.outputFileName = os.path.join(folder, "output", "output.csv")
        
        self.errorOutputFileName = os.path.join(folder, "Output", "errorTags.csv")
        self.lastHgiFileName = os.path.join(folder, "config", "lastHgi.csv")
        try:
            self.dailyAvgInput = pd.read_csv(os.path.join(self.folder, "input", "input.csv"))
            self.dailyStdInput = pd.read_csv(os.path.join(self.folder,"input", "std_input.csv"))
            self.featuresDF = pd.read_csv(os.path.join(self.folder, "config", "features.csv"))
            self.crudetaglist = pd.read_csv(os.path.join(self.folder, "config", "crudetags.csv"))
            self.formulaTagsDF = pd.read_csv(os.path.join(self.folder, "config", "formulaTags.csv"))
            self.errorCodeDF = pd.read_csv(os.path.join(self.folder, "config", "errorCode.csv"))
            self.outTagMapDF = pd.read_csv(os.path.join(self.folder, "config", "outputTagMapping.csv"))
            self.config = pd.read_csv(os.path.join(self.folder, "config", "config.csv"))
            self.graphicsDF = pd.read_csv(os.path.join(self.folder, "config", "graphics.csv"))
            self.desiredHGI = pd.read_csv(os.path.join(self.folder, "config", "desired_hgi.csv"))
            self.dynamic_tagDF = pd.read_csv(os.path.join(self.folder, "config", "dynamic_tag.csv"))

        except FileNotFoundError as FE:
            self.logData("File '" + os.path.basename(str(FE.filename)) + " is missing.")
            self.stopScript()

        try:
            self.dailyAvgInput['Time'] = pd.to_datetime(self.dailyAvgInput['Time'], format = dateFormat)
            self.dailyStdInput['Time'] = pd.to_datetime(self.dailyStdInput['Time'], format = dateFormat)
            self.dailyAvgInput.sort_values(['Time'], axis=0, inplace = True, ascending = False)
            self.dailyAvgInput.reset_index(drop= True, inplace = True)
            self.dailyStdInput.sort_values(['Time'], axis=0, inplace = True, ascending = False)
            self.dailyStdInput.reset_index(drop= True, inplace = True)
        except ValueError:
            self.logData('ERROR: Time Stamps in either input.csv or std_input.csv doesnot have time stamp in format ' + str(dateFormatString))
            self.stopScript()

        self.dailyAvgInput = self.dailyAvgInput 
        self.dailyStdInput = self.dailyStdInput

        if self.dailyAvgInput.shape[0] != self.dailyStdInput.shape[0]:
            self.logData("ERROR: Sample points count mismatch in input.csv and std_input.csv")
            self.stopScript()

        try:
            self.savedModels = dict()
            modelDetails = self.config[['functionName', 'modelName']].dropna()
            for row in modelDetails.index:
                functionName = str(modelDetails.loc[row, 'functionName'])
                modelName = str(modelDetails.loc[row, 'modelName']) + ".joblib"
                modelPath = os.path.join(self.folder, "lib", modelName)
                model = joblib.load(modelPath)
                self.savedModels[functionName] = model
                self.logData("Built model file at "+ modelPath + " loaded successfully." )
        except FileNotFoundError as FE:
            self.logData("Built model file loading unsuccessful, terminating script.")
            self.logData("File '" + os.path.basename(str(FE.filename)) + "' is missing.")
            self.stopScript()


        pointsToRun = int(self.config['pointsToRun'].values[0])
        if pointsToRun == 0 or pointsToRun > (self.dailyAvgInput.shape[0] - self.config['minLastPoints'].values[0]):
            maxSamples = int(self.dailyAvgInput.shape[0] - self.config['minLastPoints'].values[0])
        else:
            maxSamples = pointsToRun

        self.rowsToRun = maxSamples
        
        self.outSkipSpalling = self.config['OutSkipSpalling'].values[0]
        
        self.initialCalcSkipTime = self.config['initialCalcSkipTime'].values[0]
        
        self.crudeDelayTime = self.config['crudeDelayTime'].values[0]
        
        self.hgiPredictions = False        
        if self.config['hgiPredictions'].values[0]:
            self.hgiPredictions = True

        self.recoPredictions = False        
        if self.config['recoPredictions'].values[0]:
            self.recoPredictions = True
            
        self.crudeSlatePrediction = False        
        if self.config['crudeSlatePrediction'].values[0]:
            self.crudeSlatePrediction = True
            
        algoModel = self.savedModels    
        
        self.inputProcessingObj = HgiInputProcessing(errorCodeDF = self.errorCodeDF, configDF = self.config, featuresDF = self.featuresDF, formulaTagsDF=self.formulaTagsDF)
        self.calcObj = HgiCalculations(errorCodeDF = self.errorCodeDF, configDF = self.config, featuresDF = self.featuresDF, algoModel = self.savedModels, desiredHGI = self.desiredHGI, dynamic_tagDF = self.dynamic_tagDF)

    #this is the main function that does all the calculations
    def run_hgi(self): 
        try:
            avgInput = self.dailyAvgInput
            stdInput = self.dailyStdInput
            featuresDF = self.featuresDF
            formulaTagsDF = self.formulaTagsDF
            errorCodeDF = self.errorCodeDF
            outTagMapDF = self.outTagMapDF
            config = self.config
            crudetaglist=self.crudetaglist
            
            crudeDelayTime=self.crudeDelayTime
            

            errList = pd.DataFrame(columns=['Time','Tag','ErrorCode'])

            shutdownErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('shutdown')].values[0][1]
            faultyErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('status_tags_faulty')].values[0][1]
            offlineErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('offline')].values[0][1]
            spallErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('spall')].values[0][1]
            initialErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('initial_delay')].values[0][1]
            nanErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('nan_data')].values[0][1]
            stuckErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('stuck')].values[0][1]
            limitBreachErrCode = errorCodeDF[errorCodeDF['Errors'].str.match('limit_breach')].values[0][1]

            drumNames = featuresDF['drumNames'].dropna().tolist()

            lastPredPoint = int(featuresDF['Values'].values[featuresDF['Constants'].tolist().index('lastPredPoint')])
            lastHgiDFTags = featuresDF['lastRunsTags'].dropna().tolist()

            finalResultDF = pd.DataFrame()

            self.dailyAvgInput = self.inputProcessingObj.get_shutdown_status(self.dailyAvgInput)
            self.dailyAvgInput = self.inputProcessingObj.get_spall_status(self.dailyAvgInput)
            self.dailyAvgInput = self.inputProcessingObj.get_online_status(self.dailyAvgInput)

            if self.config['cycleTimeTagAvailable'].values[0]:
                self.dailyAvgInput.loc[:, 'cycleTime_hrs'] = self.dailyAvgInput.loc[:, str(self.featuresDF['cycleTime_hrsTag'].values[0])]
            else:
                self.dailyAvgInput = self.inputProcessingObj.get_cycle_time(self.dailyAvgInput)


            for currRow in range(self.rowsToRun-1, -1, -1):   
              
                print(currRow)
                
                indexstart = 12 * crudeDelayTime

                avgInput = self.dailyAvgInput.loc[currRow:]
                stdInput = self.dailyStdInput.loc[currRow:]
                
                for tags in crudetaglist['CrudeTagList'].dropna().tolist():
                    
                    avgInput[str(tags)] = pd.to_numeric(avgInput[str(tags)], errors='coerce')
                    avgInput[tags] = avgInput[tags].shift(-indexstart, axis=0, fill_value=-1)
                    

                
                avgInput = avgInput.assign(isError = 0)

                resultDF = pd.DataFrame()
                resultDF.loc[currRow, 'Time'] = avgInput.loc[currRow, 'Time']

                currTime = resultDF.loc[currRow, 'Time']
                currTime = pd.to_datetime(currTime, format = dateFormat)
                
                toUseHgiDF = False
                hgiDFAvailable = False
                try:
                    lastHgiDF = pd.read_csv(self.lastHgiFileName)
                    hgiDFAvailable = True
                    lastHgiDF['Time'] = pd.to_datetime(lastHgiDF['Time'], format=dateFormat)
                    if lastHgiDF['Time'].values[0] < currTime:
                        toUseHgiDF = True
                        self.logData("Previous record will be used.")
                    else:
                        self.logData("WARN: Sample point is older than already ran points, not using previous records.")

                except FileNotFoundError:
                    hgiDFAvailable = False
                    lastHgiDF = pd.DataFrame(columns = ['Time', 'Drum', 'CycleTime', 'Hgi'])
                    self.logData("No previous record found.")
                except (IndexError, KeyError):
                    self.logData("Sample point is older than already ran points")
                    toUseHgiDF = False

                               
                if avgInput.loc[currRow, 'shutdownStatus'] == 1:
                    avgInput.loc[currRow, 'isError'] = 1
                    for errTags in avgInput.loc[currRow, 'shutdownTags']:
                        if errTags != None:
                            errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':errTags,'ErrorCode':shutdownErrCode}, ignore_index=True)

                resultDF.loc[currRow, 'onlineDrum'] = avgInput.loc[currRow, 'onlineDrum']
                for drum in drumNames:
                    resultDF.loc[currRow, str(drum)+"DrumStatus"] = avgInput.loc[currRow, str(drum)+"DrumStatus"]
                resultDF.loc[currRow, 'cycleTime_hrs'] = avgInput.loc[currRow,'cycleTime_hrs']

                if avgInput.loc[currRow, 'shutdownStatus'] != 1:
                    if avgInput.loc[currRow, 'onlineDrum'] == 'Offline':
                        avgInput.loc[currRow,'isError'] = 1
                        errTags = [tags for tags in avgInput.loc[currRow, 'StatusBadTag'] if tags]
                        if len(errTags) != 0:
                            for errTags in avgInput.loc[currRow, 'StatusBadTag']:
                                if errTags != None:
                                    errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':errTags,'ErrorCode':faultyErrCode}, ignore_index=True)
                        else:        
                            errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':'Drums Offline','ErrorCode':offlineErrCode}, ignore_index=True)
                    
                    elif avgInput.loc[currRow, 'spallStatus'] == 1:
                        if self.outSkipSpalling:
                            avgInput.loc[currRow,'isError'] = 1
                            errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':'Drum on spall','ErrorCode':spallErrCode}, ignore_index=True)

                for idx in avgInput.index:
                    cycleEndRow = idx
                    if avgInput.loc[idx, 'cycleTime_hrs'] == 0:
                        break




                avgInput = avgInput.loc[currRow:cycleEndRow]
                stdInput = stdInput.loc[currRow:cycleEndRow]

                #generating a error code if the cycle time is at SOR and the model will skip calculaations
                if avgInput.loc[currRow,'isError'] != 1:
                    if avgInput.loc[currRow,'cycleTime_hrs'] < self.initialCalcSkipTime:
                        avgInput.loc[currRow,'isError'] = 1
                        errList = errList.append({'Time': avgInput.loc[currRow, 'Time'], 'Tag': 'Cycle Time', 'ErrorCode': initialErrCode}, ignore_index = True)

                #specialTagCreations
                if avgInput.loc[currRow, 'isError'] != 1:
                    avgInput = self.inputProcessingObj.get_special_tags(avgInput)                                  

                #getting calc tags  
                # calcTagsDF = pd.DataFrame(columns=['calcTag', 'nan_InputTag', 'stuck_InputTag'])  
                if avgInput.loc[currRow,'isError'] != 1:
                    calcBlockOut = self.inputProcessingObj.get_calc_tags(avgInput, stdInput, currRow) 
                    avgInput = calcBlockOut['inputData']
                    stdInput = calcBlockOut['stdInput']
                    calcTagsDF = calcBlockOut['calcOutput']

                #getting nan tags
                if avgInput.loc[currRow, 'isError'] != 1:                   
                    exceptionTags = featuresDF['nanException'].dropna().tolist()
                    modelTags = featuresDF['modelTags_'+str(avgInput.loc[currRow, 'onlineDrum'])].dropna().tolist()
                    tagsToCheck = [tags for tags in modelTags if tags not in exceptionTags]
                    avgInput = self.inputProcessingObj.get_nan_tags(avgInput, tagsToCheck, currRow)

                    errTags = [tags for tags in avgInput.loc[currRow, 'nanTagsList'] if tags]
                    if len(errTags) != 0:
                        avgInput.loc[currRow, 'isError'] = 1
                        for tags in errTags:
                            if tags + '_nanInputs' in calcTagsDF.columns.tolist():
                                for badTag in [tags for tags in calcTagsDF.loc[currRow, tags + '_nanInputs'] if tags]:
                                    errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':badTag,'ErrorCode':nanErrCode}, ignore_index=True)
                            else:
                                errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':tags,'ErrorCode':nanErrCode}, ignore_index=True)
                            continue

                #getting stuck tags                
                if avgInput.loc[currRow, 'isError'] != 1:
                    exceptionTags = featuresDF['StdDevException'].tolist()                   
                    modelTags = featuresDF['modelTags_'+str(avgInput.loc[currRow, 'onlineDrum'])].dropna().tolist()
                    tagsToCheck = [tags for tags in modelTags if tags not in exceptionTags]
                    avgInput = self.inputProcessingObj.get_stuck_tags(avgInput, stdInput, tagsToCheck, currRow)
                    
                    currRowStuckTag = []

                    errTags = [tags for tags in avgInput.loc[currRow, 'stuckTagsList'] if tags]
                    if len(errTags) != 0:
                        avgInput.loc[currRow, 'isError'] = 1
                        for tags in errTags:
                            if tags + '_stuckInputs' in calcTagsDF.columns.tolist():
                                for badTag in [tags for tags in calcTagsDF.loc[currRow, tags + '_stuckInputs'] if tags]:
                                    if badTag in currRowStuckTag:
                                        pass
                                    else:
                                        currRowStuckTag.append(str(badTag))
                                        errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':badTag,'ErrorCode':stuckErrCode}, ignore_index=True)
                            else:
                                if tags in currRowStuckTag:
                                    pass
                                else:
                                    currRowStuckTag.append(str(tags))                                    
                                    errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':tags,'ErrorCode':stuckErrCode}, ignore_index=True)
                            continue

                #getting out of limit tags
                if avgInput.loc[currRow, 'isError'] != 1:            
                    breachDF = self.inputProcessingObj.isLimitBreach(inputDF = avgInput, currRow = currRow, filter=str(avgInput.loc[currRow,'onlineDrum']))
                    if breachDF['breachStatus'].sum() > 0:
                        avgInput.loc[currRow, 'isError'] = 1
                        for tags in breachDF['tag']:
                            if breachDF['breachStatus'].values[breachDF['tag'].tolist().index(tags)]:
                                errList = errList.append({'Time':avgInput.loc[currRow,'Time'],'Tag':tags,'ErrorCode':limitBreachErrCode}, ignore_index=True)
         
                errorFlag = True
                if avgInput.loc[currRow, 'isError'] != 1 :
                    errorFlag = False
                
                
                if self.hgiPredictions:
                    hgiOutput = self.calcObj.HgiPred(avgInput, resultDF, currRow, errorFlag)
                    resultDF = hgiOutput['resultOut']                 
                    avgInput = hgiOutput['inputOut']
                    

                

                if self.recoPredictions:
                    alarmRecoOutput = self.calcObj.alarms_reco(avgInput, resultDF, currRow, errorFlag)
                    resultDF = alarmRecoOutput['resultOut']                 
                    avgInput = alarmRecoOutput['inputOut']
                    
                if self.crudeSlatePrediction:
                    crudeSlateOutput = self.calcObj.crude_slate(avgInput, resultDF, currRow, errorFlag)
                    resultDF = crudeSlateOutput['resultOut']                 
                    avgInput = crudeSlateOutput['inputOut']
                    
                    

                if not hgiDFAvailable:
                    lastHgiDF = resultDF.loc[[currRow], lastHgiDFTags]
                elif hgiDFAvailable and toUseHgiDF:
                    lastHgiDF = pd.concat([resultDF.loc[[currRow], lastHgiDFTags], lastHgiDF], ignore_index = True, sort=False)
                    lastHgiDF = lastHgiDF.loc[0:lastPredPoint, :]
                else:
                    pass

                lastHgiDF['Time'] = lastHgiDF['Time'].dt.strftime(dateFormat)
                lastHgiDF.to_csv(self.lastHgiFileName, index = None, header = True, mode="w")

                if toUseHgiDF:
                    getDrumHgi = False
                    if lastHgiDF.shape[0] > lastPredPoint:
                        if lastHgiDF['onlineDrum'].values[0] == 'Offline' or lastHgiDF['onlineDrum'].values[0] != lastHgiDF['onlineDrum'].values[1] :
                            getDrumHgi = True
                            iOnlineDrum = lastHgiDF['onlineDrum'].values[1]
                            iCycleTime = lastHgiDF['cycleTime_hrs'].values[1]
                            for iIndex in range(2,lastHgiDF.shape[0]):
                                if lastHgiDF['onlineDrum'].values[iIndex] != iOnlineDrum or lastHgiDF['cycleTime_hrs'].values[iIndex] > iCycleTime :
                                    getDrumHgi = False
                                    break
                        if getDrumHgi:
                            for iIndex in range(2,lastHgiDF.shape[0]):
                                hgi_pred_prev_cycle = lastHgiDF.loc[iIndex, 'Hgi_Pred_hour0']
                                prevcycle_drum = lastHgiDF.loc[iIndex, 'onlineDrum']
                                hgi_pred_prev_cycle = pd.to_numeric(hgi_pred_prev_cycle, errors='coerce')
                                if str(hgi_pred_prev_cycle) != 'nan' and hgi_pred_prev_cycle != 0 and str(prevcycle_drum) != 'Offline':
                                    break
                                
                                
                            for drums in self.featuresDF['drumNames'].dropna().tolist():
                                resultDF.loc[currRow, 'Prev_Hgi_Pred'+'_'+str(drums)] = ""
                                
                            resultDF.loc[currRow, 'Prev_Hgi_Pred'] = hgi_pred_prev_cycle
                            resultDF.loc[currRow, 'Prev_Hgi_Pred'+'_'+str(prevcycle_drum)] = hgi_pred_prev_cycle


                                
                                
                
                # finalResultDF = finalResultDF.append(resultDF, ignore_index = True, sort = False) 
                finalResultDF = pd.concat([finalResultDF, resultDF], ignore_index=True, sort=False)

            

            finalResultDF = finalResultDF.iloc[::-1]

            newOutTag = []
            oldTag = []
            for outTags in finalResultDF.columns:
                if outTags in outTagMapDF['modelTag'].values:
                    newOutTag.append(outTagMapDF[outTagMapDF['modelTag'].str.match(outTags)].values[0][1])
                    oldTag.append(outTags)
                    try:
                        finalResultDF[outTags] = finalResultDF[outTags].round(outTagMapDF[outTagMapDF['modelTag'].str.match(outTags)].values[0][2])
                    except TypeError:
                        pass

            finalResultDF = finalResultDF.loc[:, oldTag]            
            finalResultDF.columns = newOutTag
            
            
            # finalResultDF.loc[0, 'Furnace Charge'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Inlet Temperature'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Inlet Pressure'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Outlet Temperature'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'LCGO Quech Flow'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Crude Sulpur'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Crude API'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Crude Nickel'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Crude Vanadium'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Resid API'] = 10 # Only added as dummy for UI purpose
            # finalResultDF.loc[0, 'Outlet Pressure'] = 10 # Only added as dummy for UI purpose
            
            finalResultDF.to_csv(self.outputFileName, index = None, header = True, mode="w")
            self.logData("results written in "+ self.outputFileName)

            errList.to_csv(self.errorOutputFileName, index = None, header = True, mode="w")
            self.logData("Errors if any, reported in "+ self.errorOutputFileName)
            



            self.stopScript()

        except Exception as e:
            if hasattr(e, 'message'):
                self.logData("Error: Exception raised " + str(e.message))
                error_msg = {'error': e.message}
            else:
                self.logData("Error: Exception raised " + str(type(e).__name__))
                error_msg = {'error': type(e).__name__}
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.logData("Exception occured in line " + str(exc_tb.tb_lineno))
            traceback.print_exc(file=sys.stdout)
            return json.dumps(error_msg)
            self.stopScript()

# module_hgi_Obj = module_hgi()
# module_hgi_Obj.run_hgi()
