
import numpy as np
import pandas as pd
import math

import os
import sys
os.system('cls' if os.name == 'nt' else 'clear')

folder = os.path.dirname(os.path.dirname(__file__) )


class HgiInputProcessing():

    def __init__(self, **kwargs):
        self.errorCodeDF = kwargs.get('errorCodeDF')
        self.featuresDF = kwargs.get('featuresDF')
        self.configDF = kwargs.get('configDF')
        self.formulaTagsDF = kwargs.get('formulaTagsDF')


    def isLimitBreach(self, inputDF, currRow, tagsToCheck=[], filter='Common'):
        
        inputData = inputDF.copy(deep=True)

        breachDF = pd.DataFrame(columns=['tag','breachStatus'])
        columns = ['filter', 'tags', 'min_val', 'max_val'] 
        limitTable = self.featuresDF.loc[:,columns] 
        limitTable = limitTable[limitTable['filter'] == filter]

        if len(tagsToCheck) == 0:
            tagsToCheck = limitTable['tags'].dropna().tolist()

        for tags in tagsToCheck:
            tempTable = limitTable[limitTable['tags'] == tags]
            if float(tempTable['min_val'].values[0]) <= float(inputData.loc[currRow,tags]) <= float(tempTable['max_val'].values[0]):
                # breachDF = breachDF.append({'tag':tags,'breachStatus':False}, ignore_index = True)
                
                breachDF = pd.concat([breachDF, pd.DataFrame([{'tag': tags, 'breachStatus': False}])], ignore_index=True)
            else:
                # breachDF = breachDF.append({'tag':tags,'breachStatus':True}, ignore_index = True)
                breachDF = pd.concat([breachDF, pd.DataFrame([{'tag': tags, 'breachStatus': True}])], ignore_index=True)
        
        if len(tagsToCheck) == 1:
            return breachDF['breachStatus'].values[0]
        else:
            return breachDF

    def get_shutdown_status(self, inputDF):
        inputData = inputDF.copy(deep=True)
        try:
            featuresDF = self.featuresDF
            newColumns = ['shutdownStatus', 'shutdownTags']
            shutdownTable = featuresDF[['shutdown_Tag','shutdown_Min','shutdown_Max']].dropna() #getting online determining tag and its limit
            shutdownTable.columns = ['Tag', 'MinVal', 'MaxVal'] #renaming the headers for convenience
            tempDF = pd.DataFrame()
            tempBadTagDF = pd.DataFrame()
            for idx in shutdownTable.index:
                tagName = shutdownTable.loc[idx, 'Tag']
                tagMinVal = shutdownTable.loc[idx, 'MinVal']
                tagMaxVal = shutdownTable.loc[idx, 'MaxVal']
                inputTagSeries = inputData[tagName]
                inputTagSeries = pd.to_numeric(inputTagSeries, errors='coerce') #converting to numeric incase the dtype has changed to object type
                tempDF[str(idx)] = np.where(np.logical_and(~np.isnan(inputTagSeries), (tagMinVal>inputTagSeries) | (inputTagSeries>tagMaxVal)), 1, 0)
                tempBadTagDF[str(idx)] = np.where(tempDF[str(idx)] == 1, tagName , None)               
            inputData['shutdownStatus'] = tempDF.sum(axis=1)
            inputData['shutdownTags'] = tempBadTagDF.values.tolist()
            del tempDF
            return inputData

        except KeyError as e: #incase any column is not available, it is reported and script is stopped.
            raise Exception('Column not found in features.csv: ' + str(e))

    def get_spall_status(self, inputDF):
        
        inputData = inputDF.copy(deep=True)
        featuresDF = self.featuresDF
        #reading furnace names, numbers and DOl tags from feature.csv
        spallTable = featuresDF[['spallTags','spallTagsMinLimit','spallTagsMaxLimit']].dropna() #getting spall tag and its limit
        spallTable.columns = ['Tag', 'MinVal', 'MaxVal'] #renaming the headers for convenience
        inputData['spallStatus'] = 1
        tempDF = pd.DataFrame()

        for idx in spallTable.index:
            tagName = spallTable.loc[idx, 'Tag']
            tagMinVal = spallTable.loc[idx, 'MinVal']
            tagMaxVal = spallTable.loc[idx, 'MaxVal']
            inputTagSeries = inputData[tagName]
            inputTagSeries = pd.to_numeric(inputTagSeries, errors='coerce') #converting to numeric incase the dtype has changed to object type
            tempDF[str(idx)] = np.where((tagMinVal < inputTagSeries) & (inputTagSeries < tagMaxVal), 1 ,0)   

        inputData['spallStatus'] = tempDF.product(axis=1)
        del tempDF
       
        return inputData

    def get_online_status(self, inputDF):

        inputData = inputDF.copy(deep=True)

        drumNames = self.featuresDF['drumNames'].dropna()
        statusTags = self.featuresDF['drumStatusTags'].dropna()
        statusTagsValues = self.featuresDF['drumStatusTagsValue'].dropna()

        onlineTable = self.featuresDF[['drumNames','drumStatusTags','drumStatusTagsValue','drumStatusTagsMin','drumStatusTagsMax']].dropna() #getting online determining tag and its limit
        onlineTable.columns = ['Names', 'Tag', 'MatchVal', 'MinVal', 'MaxVal'] #renaming the headers for convenience
        
        tempOnlineDF = pd.DataFrame()
        tempBadDF = pd.DataFrame()
        for drums in drumNames:
            currDrumStatusTable = onlineTable.loc[onlineTable['Names'] == drums]
            tempDF = pd.DataFrame()
            for idx in currDrumStatusTable.index:
                tagName = currDrumStatusTable.loc[idx, 'Tag']
                tagMatchVal = currDrumStatusTable.loc[idx, 'MatchVal']
                tagMinVal = currDrumStatusTable.loc[idx, 'MinVal']
                tagMaxVal = currDrumStatusTable.loc[idx, 'MaxVal']
                inputTagSeries = inputData[tagName]
                inputTagSeries = pd.to_numeric(inputTagSeries, errors='coerce')
                tempBadDF[str(drums) + str(idx)] = np.where(np.logical_or(str(inputTagSeries)=='nan' ,np.logical_and(tagMinVal <= inputTagSeries, inputTagSeries <= tagMaxVal)), None, tagName)
                tempDF[str(idx)] = np.where(inputTagSeries == tagMatchVal, 1, 0)
            tempOnlineDF[str(drums)+'Online'] = tempDF.product(axis=1) # .product is used as all the tags must satisfy the condition. If any one tag to satisfy then use '.sum(axis=1)'

        tempOnlineDF.index = inputData.index
        tempBadDF.index = inputData.index

        inputData['onlineDrum'] = np.where(np.logical_or(np.logical_or(tempOnlineDF.sum(axis=1) == 0, tempOnlineDF.sum(axis=1) == len(drumNames.tolist())), inputData['shutdownStatus'] == 1), 'Offline', 'Online')
        for drums in drumNames:
            inputData[str(drums)+'DrumStatus'] = np.where(np.logical_and(inputData['onlineDrum'] == 'Online', tempOnlineDF[str(drums)+'Online']==1), 'Online', 'Offline')
            inputData['onlineDrum'] = np.where(inputData[str(drums)+'DrumStatus'] == 'Online', drums, inputData['onlineDrum'])
        inputData['StatusBadTag'] = tempBadDF.values.tolist()
        
        spallCounterDF = inputData.copy(deep=True)
        spallCounterDF = spallCounterDF[::-1]
        spallCounterDF['block'] = (spallCounterDF['onlineDrum'] != spallCounterDF['onlineDrum'].shift(1)).astype(int).cumsum()
        spallCounterDF['spallCount'] = spallCounterDF.groupby('block')['spallStatus'].cumsum()
        spallCounterDF = spallCounterDF[::-1]
        
        inputData['spallStatus'] = np.where(inputData['spallStatus']==1, 1, np.where(spallCounterDF['spallCount'] >= self.featuresDF['SpallCounterThreshold'].values[0], 1, 0))


        del tempOnlineDF, tempBadDF, tempDF, spallCounterDF

        return inputData

    def get_cycle_time(self, inputDF):  

        inputData = inputDF.copy(deep=True)
        featuresDF = self.featuresDF

        cycleTimeName = 'cycleTime_hrs'

        tempDF = inputData.copy(deep=True)
        tempDF[cycleTimeName] = -1
        tempDF['shiftedOnlineDrum'] = tempDF['onlineDrum'].shift(-1)
        tempDF['changed'] = (tempDF['onlineDrum'] == tempDF['shiftedOnlineDrum']).astype(int)

        startDatesSeries =  tempDF.loc[np.logical_and(tempDF['changed'] == 0, tempDF['onlineDrum'] != 'Offline')]
        for timeIndex in startDatesSeries.index:
            sTime = startDatesSeries.loc[timeIndex,'Time']
            tempDF[cycleTimeName] = np.where(np.logical_and(tempDF[cycleTimeName] == -1, tempDF.index <= timeIndex), (tempDF['Time'] - sTime).dt.total_seconds()/3600 , tempDF[cycleTimeName])
            
        tempDF[cycleTimeName] = np.where(tempDF['onlineDrum'] =='Offline', 0, tempDF[cycleTimeName])
        
        inputData[cycleTimeName] = tempDF[cycleTimeName]
        inputData[cycleTimeName] = round(inputData[cycleTimeName],2)
        del tempDF

        return inputData
 
    def get_special_tags(self, inputDF):
        inputData = inputDF.copy(deep=True)
        for names in self.featuresDF['resultTag'].dropna().unique():
            table = self.featuresDF.loc[self.featuresDF['resultTag'] == names]
            table.reset_index(drop=True, inplace = True)                
            outDF = pd.DataFrame(columns=[names])
            for iRow in inputData.index:
                #self.progressBar("Getting drum data", iRow+1, avgInput.shape[0])
                for idx in range(0, table.shape[0]): 
                    tag = str(table.loc[idx,'inputTag'])
                    condition = str(table.loc[idx,'condition'])
                    try:              
                        if eval("inputData.loc[iRow,tag] " + condition):
                            outDF.at[iRow,names] = table.loc[idx, 'status']
                    except (TypeError, NameError, KeyError, IndexError, ValueError):
                        outDF.at[iRow,names] = 0
            inputData[names] = outDF[names]
            del(outDF)
        
        return inputData

    def get_nan_tags(self, inputDF, tagsToCheck=[], currRow = -1):
        inputData = inputDF.copy(deep=True)
        if currRow == -1:
            rows = inputData.index.tolist()
        else:
            rows = [currRow]
        if len(tagsToCheck) == 0:
            tagsToCheck = inputData.columns().tolist()

        tempNan = pd.DataFrame()
        for tags in tagsToCheck:
            inputTagSeries = pd.to_numeric(inputData.loc[rows,tags], errors='coerce')
            tempNan[tags] = np.where(np.isnan(inputTagSeries) , tags, None) 
        
        tempNan.index = rows
        nanTags = tempNan
        
        nanTags['nanTagsList'] = tempNan.values.tolist()

        inputData = pd.concat([inputData, nanTags['nanTagsList']], axis=1, sort=False)
        del tempNan, nanTags

        return inputData

    def get_stuck_tags(self, inputDF, stdInputDF, tagsToCheck=[], currRow =-1):
        inputData = inputDF.copy(deep=True)
        stdDevData = stdInputDF.copy(deep=True)
        if currRow == -1:
            rows = inputData.index.tolist()
        else:
            rows = [currRow]
        if len(tagsToCheck) == 0:
            tagsToCheck = stdDevData.columns().tolist()
        
        tempStuck = pd.DataFrame()
        for tags in tagsToCheck:
            if tags not in stdDevData:
                continue
            inputTagSeries = pd.to_numeric(stdDevData.loc[rows,tags], errors='coerce')
            tempStuck[tags] = np.where(np.logical_or(np.isnan(inputTagSeries), inputTagSeries == 0) , tags, None)
        
        tempStuck.index = rows
        stuckTags = tempStuck
        stuckTags['stuckTagsList'] = tempStuck.values.tolist()
        inputData = pd.concat([inputData, stuckTags['stuckTagsList']], axis=1, sort=False)

        del tempStuck, stuckTags
        return inputData
    
    

    def get_calc_tags(self, inputDF, stdInputDF, currRow = -1):
        
        inputData = inputDF.copy(deep=True)
        stdDevData = stdInputDF.copy(deep=True)

        calcTagsDF = pd.DataFrame(index=inputData.index)

        for calcTags, formulas in self.formulaTagsDF.items():
            tagsInFormula = []                
            currFormula = formulas.values[0]
            tempFormula = currFormula

            while tempFormula.find("{") != -1:
                tagsInFormula.append(tempFormula[tempFormula.find("{")+len("{"):tempFormula.find("}")])
                tempFormula = tempFormula[tempFormula.find("}")+1:] 

            tempStdDF = pd.DataFrame(index=inputData.index)
            tempStdBadTagDF = pd.DataFrame(index=inputData.index)

            tempAvgDF = pd.DataFrame(index=inputData.index)
            tempAvgBadTagDF = pd.DataFrame(index=inputData.index)

            for inputTags in tagsInFormula:

                if inputTags in stdDevData.columns.tolist():
                    inputTagStdSeries = pd.to_numeric(stdDevData[inputTags], errors='coerce')
                    tempStdDF[inputTags] = np.where(np.logical_or(np.isnan(inputTagStdSeries), inputTagStdSeries == 0), 0, inputTagStdSeries)
                    tempStdBadTagDF[inputTags] = np.where(np.logical_or(np.isnan(inputTagStdSeries), inputTagStdSeries == 0), inputTags, None)


                inputData[inputTags] = pd.to_numeric(inputData[inputTags], errors='coerce')
                inputTagSeries = inputData[inputTags]
                tempAvgBadTagDF[inputTags] = np.where(np.isnan(inputTagSeries), inputTags, None)
                currFormula = currFormula.replace(str("{"+inputTags+"}"),str("inputData['" + str(inputTags) + "']"))

            stdDevData[calcTags] = tempStdDF.product(axis=1)
            calcTagsDF[calcTags + '_stuckInputs'] = tempStdBadTagDF.values.tolist()

            try:
                inputData[calcTags] = eval(currFormula)
            except (TypeError, KeyError):
                inputData[calcTags] = None
            calcTagsDF[calcTags + '_nanInputs'] = tempAvgBadTagDF.values.tolist()

            del tempStdDF, tempStdBadTagDF, tempAvgBadTagDF

        output = dict()
        output = {'inputData': inputData, 'stdInput': stdDevData, 'calcOutput': calcTagsDF }

        return output

class HgiCalculations():

    def get_model_output(self, varX_DF, param):

        if param=='HgiPred':
            algoModels=self.algoModels
            modelToUse = self.algoModels[str(param)]
            varX = varX_DF[modelToUse['modelX']]
            scaledX = modelToUse['scalerX'].transform(varX)
            pred_ScaledY = modelToUse['algo_ML'].predict(scaledX)
            pred_Y = modelToUse['scalerY'].inverse_transform(pred_ScaledY.reshape(-1,1)).ravel()
            
            return pred_Y



    def __init__(self, **kwargs):
        self.errorCodeDF = kwargs.get('errorCodeDF')
        self.featuresDF = kwargs.get('featuresDF')
        self.configDF = kwargs.get('configDF')
        self.algoModels = kwargs.get('algoModel')
        self.desiredHGI = kwargs.get('desiredHGI')
        self.dynamic_tagDF = kwargs.get('dynamic_tagDF')
        
        
    def tag_correction(self, avgInput, correctedTags, currRow):
        
        input_data_df = avgInput
        featuresDF = self.featuresDF
        dynamic_tagDF = self.dynamic_tagDF
        
        if 1:

            runtime_tag_df = pd.DataFrame()
            tag_with_value_zero_list = []

            for each_column, each_value in dynamic_tagDF.iteritems():
                calc_value_list = []
                tag_creation_eq = each_value.values[0]
#                self.logger.info('equation which need to be eval is %s for column %s', tag_creation_eq, each_column)
                if "'" in tag_creation_eq:
                    tag_creation_eq = tag_creation_eq.replace("'", "")
                tag_creation_list = tag_creation_eq.replace('(', '@').replace(')', '@').replace('+', '@').replace(
                    '-', '@').replace('*', '@').replace('/', '@').replace('[', '@').replace(']', '@').replace(',',
                                                                                                              '@').split(
                    '@')
                # removing blank if exist

                while "" in tag_creation_list:
                    tag_creation_list.remove("")

                # converting if the number is float to integer
                tag_creation_list = [x for x in tag_creation_list if not x.isalnum()]
                # removing number from list
                tag_creation_list = [x for x in tag_creation_list if not x.isnumeric()]
                # checking if the column exist in input data frame
                column_name_list = []
                tag_missing_list = []
                for i in tag_creation_list:
                    if i in input_data_df.columns:
                        column_name_list.append(i)
                    else:
                        # finding if the list doesnt have numbers
                        found_alphabet = re.search('[a-zA-Z]', i)
                        if found_alphabet:
                            tag_missing_list.append(i)


                matched_input_data_df = input_data_df.loc[[currRow], column_name_list]

                list_of_dict_matched_input_data = matched_input_data_df.to_dict('records')
                for each_row in list_of_dict_matched_input_data:
                    temp_eq = tag_creation_eq
                    faulty_calc_tag_list = []
                    for k, v in each_row.items():

                        if k in temp_eq:
                            temp_eq = temp_eq.replace(k, str(v))
                    try:
                        calc_value_list.append(eval(temp_eq))
                    except Exception as e:
                        # creation of calc tag csv
                        temp_list = list(each_row.keys())
                        tag_with_value_zero_list.extend(temp_list)

                        if hasattr(e, 'message'):
                            self.logger.error("Exception rasied %s for equation %s", e.message, temp_eq)
                        else:
                            self.logger.error("Exception rasied %s for equation %s", type(e).__name__, temp_eq)
                        calc_value_list.append(0)

                runtime_tag_df[each_column] = calc_value_list
            tag_with_value_zero_list = list(set(tag_with_value_zero_list))


            return runtime_tag_df


    def HgiPred(self, inputDF, resultDF, currRow, errorFlag):
 
        avgInput = inputDF.copy(deep=True)
        resultOut = resultDF.copy(deep=True)
        currRow = currRow
        featuresDF = self.featuresDF
        config = self.configDF
        dynamic_tagDF = self.dynamic_tagDF


        if not errorFlag:
            unknownErrCode = self.errorCodeDF[self.errorCodeDF['Errors'].str.match('unknown_error')].values[0][1]
            
            avgTags = featuresDF['avgTags'].dropna().tolist()
            correctedTags = featuresDF['specialTags'].dropna().tolist()
#            correctedFormula = featuresDF['specialFormula'].dropna().tolist()
            
            currCycleTime = avgInput.loc[currRow, 'cycleTime_hrs']

            for tags in avgTags:
                avgInput[str(tags)] = pd.to_numeric(avgInput[str(tags)], errors='coerce')
                avgInput.loc[currRow, str(tags) + "_AVG"] = avgInput[str(tags)].mean()
                

            if config['tagCorrection'].values[0]:        
                corrected = self.tag_correction(avgInput, correctedTags, currRow)        
                for tags in correctedTags:
                    avgInput.loc[currRow, str(tags) + "_AVG"]=corrected[str(tags)+ "_AVG"].values[0]


            varX = avgInput.loc[[currRow],:]
            hgi = self.get_model_output(varX, param='HgiPred')
                

            resultOut.loc[currRow, 'Hgi_Pred_hour0'] = hgi



        output = {'inputOut': avgInput, 'resultOut': resultOut}    
        return output


    def alarms_reco(self, inputDF, resultDF, currRow, errorFlag):
        
        resultOut = resultDF.copy(deep=True)
        avgInput = inputDF.copy(deep=True)

        featuresDF = self.featuresDF     
        configDF = self.configDF
        desiredHGI = self.desiredHGI
                
        recoNum = int(featuresDF['alarmReco_number'].values[0])
        recoType = featuresDF['alarmRecoType'].dropna()
        reco = featuresDF['alarmReco'].dropna()
        alarmCond = featuresDF['alarmCond'].dropna()
        HgiLess = featuresDF['HgiLess'].dropna()
        HgiMore = featuresDF['HgiMore'].dropna()
        HgiInBand = featuresDF['HgiInBand'].dropna()
        
        targetHgi = int(desiredHGI['targetHgi'].values[0])
        hgidelta = int(desiredHGI['HgiDelta'].values[0])
        hgipredDelta = int(featuresDF['HgiPredDelta'].values[0])
        
        targetHgiUpper = targetHgi + hgidelta
        targetHgiLower = targetHgi - hgidelta
        
        resultOut.loc[currRow, 'targetHgiUpper'] = targetHgiUpper
        resultOut.loc[currRow, 'targetHgiLower'] = targetHgiLower
        
#        resultOut.loc[currRow, 'HgiPredUpper'] = resultOut['Hgi_Pred_hour0'].values[0] + hgipredDelta
#        resultOut.loc[currRow, 'HgiPredLower'] = resultOut['Hgi_Pred_hour0'].values[0] - hgipredDelta

        if errorFlag:
            for i in range(0, recoType.shape[0]):
                resultOut.loc[currRow, 'recoFlag'+str(recoType.values[i])] = ""
        else:
            resultOut.loc[currRow, 'HgiPredUpper'] = resultOut['Hgi_Pred_hour0'].values[0] + hgipredDelta
            resultOut.loc[currRow, 'HgiPredLower'] = resultOut['Hgi_Pred_hour0'].values[0] - hgipredDelta            
            if resultOut['Hgi_Pred_hour0'].values[0] < targetHgiLower:
                for i in range(0, recoType.shape[0]):                    
                    resultOut.loc[currRow, 'recoFlag'+str(recoType.values[i])] = HgiLess.values[i]    
                    

            elif resultOut['Hgi_Pred_hour0'].values[0] > targetHgiUpper:
                for i in range(0, recoType.shape[0]):
                    resultOut.loc[currRow, 'recoFlag'+str(recoType.values[i])] = HgiMore.values[i]
            else:
                for i in range(0, recoType.shape[0]):
                    resultOut.loc[currRow, 'recoFlag'+str(recoType.values[i])] = HgiInBand.values[i]
                    
                    

        
        output = {'inputOut': avgInput, 'resultOut': resultOut}    
        return output
    
    def crude_slate(self, inputDF, resultDF, currRow, errorFlag):
        
        resultOut = resultDF.copy(deep=True)
        avgInput = inputDF.copy(deep=True)
        test_df=avgInput.head(1)
        

        featuresDF = self.featuresDF      
        recoNum = int(featuresDF['alarmReco_number'].values[0])

        if errorFlag:
            resultOut.loc[currRow, 'HeavyWtPercent'] = ""
            resultOut.loc[currRow, 'HeavyWtPercent'] = ""
            resultOut.loc[currRow, 'HeavyWtPercent'] = ""
            
                
        else:
            resultOut.loc[currRow, 'HeavyWtPercent'] = (avgInput["Heavy_TPD"].values[0]*100)/(avgInput["Heavy_TPD"].values[0]+avgInput["Medium_TPD"].values[0]+avgInput["Light_TPD"].values[0])
            resultOut.loc[currRow, 'MediumWtPercent'] = (avgInput["Medium_TPD"].values[0]*100)/(avgInput["Heavy_TPD"].values[0]+avgInput["Medium_TPD"].values[0]+avgInput["Light_TPD"].values[0])
            resultOut.loc[currRow, 'LightWtPercent'] = (avgInput["Light_TPD"].values[0]*100)/(avgInput["Heavy_TPD"].values[0]+avgInput["Medium_TPD"].values[0]+avgInput["Light_TPD"].values[0])
            
        #resultOut = pd.concat([resultOut.reset_index(drop=True), test_df.reset_index(drop=True)], axis=1)

        key = "Time"
        resultOut = resultOut.merge(
            test_df.drop(columns=test_df.columns.intersection(resultOut.columns).difference([key])),
            how="left",
            on=key
            )
        
        output = {'inputOut': avgInput, 'resultOut': resultOut}    
        return output




