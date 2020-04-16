#!/usr/bin/env python3

"""
Generic Machine Learning API
Web access to the Cloud functions using mongodb
@author Shreyas Gokhale, Orhan Can Görür
"""
import os
import re
import threading
import sys
import joblib

import csv
import requests
import web
import json
import urllib
import pickle
import pandas as pd
from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import jsonpickle
import seaborn as sns

from GeneticAlgorithmControl import GeneticAlgorithmControl
from ChariotCloudAPI import WebServer

from PMmodels.ocsvm import OCSVM
from PMmodels.knn import KNN
from PMmodels.kpca import KPCA

pm_result_list = []

server = WebServer()
link = 'http://' + server.weblink


class GenericTrain(object):
    """
    Generic ML Training: train a model and save the model and related intermediate variables to mysql database.

    # When Training req sent, data should either be given or should be already in the table
    @author: Orhan Can Görür, Shreyas Gokhale
    @contact: orhan-can.goeruer@dai-labor.de
    """

    def POST(self):
        """
        POST this is the POST function of the GenericTrain.

        Example of request: (* fields are optional)
        {
            "algorithm" : "OCSVM",      # Algorithm you want to use to train the data
            "deviceID" : c3cedae8-6143-4421-84aa-32e527c6b04e, # Device ID.
            "para" :[0.01,0.5],         # Parameters for ML algorithm
            "properties" : ["velocity","power_in"],   # What columns from the data you want for training
        *   "database" : "predictive_maintenance" # If you want to save table in a specific database. It is "Generic" by default.
        }
        # TODO: Combining multiple IDs for one training.

        @rtype: string
        @return: error type or successful reminder
        """
        print("-----------------------------------")
        print("GenericTrain Started")

        try:
            result = json.loads(web.data().decode('utf-8'))
            algorithm = result["algorithm"]
            deviceID = str(result["deviceID"])

            if "database" in result:
                database = result["database"]
            else:
                database = "predictive_maintenance"

            properties = result["properties"]
            # para1 = result["para"][0]  # n_neighbors
            # para2 = result["para"][1]  # Leaf Size
            para_arr = result["para"]

            # Table Name
            property_id = ""
            for property_key in properties:
                property_id = property_id + str(property_key)
            table = "raw" + "_" + property_id +  "_" + deviceID
            #table = result["table"]

            if "resulttable" in result:
                resulttable = result["resulttable"]
            else:
                resulttable = table

            if __debug__:
                print(result)

        except Exception as e:
            print("Error occurred when reading data from post: " + str(e))

        # Get Data from the columns mentioned in the request
        try:
            # Creating query in python dict
            req = {
                'table': table,
                'database': database,
                'deviceID': deviceID
            }

            # This is commented out as csv has all the IDs 0. So for now, just get the data using table name.
            # data_response = requests.post(link + '/cloud/db/getTableDataByID/', data=json.dumps(req))

            data_response = requests.post(link + '/cloud/db/getTableData/', data=json.dumps(req))
            data_response = data_response.json()  # We get data in string, convert in json/dict.
            data_response = pd.DataFrame(data_response)  # Convert to pandas df
            data = data_response[result["properties"]]  # Select columns matching the request

        except Exception as e:
            print(e)
            return "Error in fetching data from the database"

        # Train model
        try:
            if algorithm == 'OCSVM' or algorithm == 'KNN' or algorithm == 'KPCA':
                model = getattr(sys.modules[__name__], algorithm)()
            else:
                raise Exception("Input algorithm not supported (Only supports OCSVM, KNN or KPCA).")

            pickledDict = {}
            if algorithm == 'OCSVM':
                clf = model.fit(data, para_arr[0], para_arr[1])
            else:
                if algorithm == 'KNN':
                    clf, out_var = model.fit(data, para_arr[0], para_arr[1])
                elif algorithm == 'KPCA':
                    clf, scaler, out_var = model.fit(data, para_arr[0], para_arr[1])
                    pickledDict['scaler'] = scaler

                pickledDict['out_var'] = out_var

            pickledDict['clf'] = clf

        except Exception as e:
            print("Error occurred during model training: " + str(e))
            return e

        try:
            print("Model saving")

            payload = {}
            payload["table"] = str(algorithm) + "_model_" + property_id + "_" + str(para_arr[0]) + "_" + str(para_arr[1]) + "_" + deviceID
            payload["database"] = database
            payload["data"] = {
                "clf": str(pickle.dumps(clf)),
                "algorithm": algorithm,
                "table": table,
                "database":database,
                "model": jsonpickle.encode(pickledDict),
                "deviceID": deviceID
            }

            payload["id"] = deviceID

            data = requests.post(link + '/cloud/db/writeOneRowInTable/', data=json.dumps(payload))

            print('%s model trained.' % algorithm)
            return '%s model trained.' % algorithm

        except Exception as e:
            print("Error occurred during model saving: " + str(e))
            return e

        finally:
            print("GenericTrain function finished!")
            print("-----------------------------------")


class GenericPredict(object):
    """
    Predictive maintenance: prediction for any generic data using the previously saved model in MongoDB database

    Functionalities:
    1.  User has a data stored in the database table:
    2. User has given data manually -
        1. Predict and save it in a database. If it does not exist: Create a new database.
        2. Just return the result of input given by user

    Example request:
    {
    !   "algorithm" : "OCSVM",              # Algorithm you want to use to train the data.
    !    "properties" : ["velocity","torque"],           # The columns you trained the data for.
    !a   "table": "real_motor",             # EITHER the Name of the table where your prediction data is stored.
    !b   "input": [ 31.00 , 5],
    !   "para": [0.01, 0.5],          # OR the values to predict itself.
    *   "deviceID": "c3cedae8-6143-4421-84aa-32e527c6b04e", # What do you call your ID. If not provided, it is "ID" by default.
    *   "database": "predictive_maintenance"          # If you want to save table in a specific database. It is "ChariotCloud" by default.
    *   "optional_resulttable" : "nosave"   # Table Name to save results. If = "nosave", don't save - just return the results to the user.
    }

    @author: Orhan Can Görür, Shreyas Gokhale
    @contact: orhan-can.goeruer@dai-labor.de
    """

    def __init__(self):
        global pm_result_list
        self.moving_aver_size = 20
        self.pm_weight = 0.4

    def POST(self):
        """
        POST this is the POST function of the GenericPredict
        @rtype: string
        @return: error type or successful reminder
        """
        predictAllFlag = False  # By default

        print("-----------------------------------\n")
        print("This is the start of GenericPredict function!")
        print("-----------------------------------")

        # Step 1: Understand the request
        try:
            result = json.loads(web.data().decode('utf-8'))
            algorithm = result["algorithm"]  # Algorithm that you want to use
            deviceID = result["deviceID"]  # ID of your device
            num = len(result["properties"])  # Data Column Names

            cols = result["properties"]
            if "database" in result:
                database = result["database"]
            else:
                database = "predictive_maintenance"

            if "idname" in result:
                idname = result["idname"]
            else:
                idname = "deviceID"

            if "para" in result:
                para_arr = result["para"]
            else:
                para_arr = [0.01, 0.5]

            # Table name for the raw data
            property_id = ""
            for property_key in cols:
                property_id = property_id + str(property_key)
            dataTable = "raw" + "_" + property_id + "_" + str(deviceID) # every raw data is saved with the device ID

            if "optional_resulttable" in result:  # User wants to save in table with this specific name
                resultTable = str(result["optional_resulttable"])  # IF resulttable is also nosave, no data will be saved in DB
            else:
                resultTable = "results_" + property_id + "_" + str(deviceID)

            if "input" in result:  # User has given data values to predict. After Prediction, store values and data in saved table.
                inputs = result["input"]
                realTimeFlag = True  # We inplicitly know that user is sending realtime data

                if num == 2:
                    val1 = result["input"][0]
                    val2 = result["input"][1]
                elif num == 3:
                    val1 = result["input"][0]
                    val2 = result["input"][1]
                    val3 = result["input"][2]
                else:
                    raise NameError('More than 3 inputs are not supported currently')
            else:
                realTimeFlag = False

            if resultTable == dataTable:
                sameTableUpdateFlag = True
            else:
                sameTableUpdateFlag = False

            model_table = str(algorithm) + "_model_" + property_id + "_" + str(para_arr[0]) + "_" + str(para_arr[1]) + "_" + str(deviceID)

            if __debug__:
                print(result)


        except Exception as e:
            print("Error occurred when reading data from post: " + str(e))

        # Step 2: Fetch the data to be predicted
        try:
            # read in data
            try:
                if realTimeFlag == False:  # Read the data from the table
                    try:  # For now, this reads all the table data
                        #
                        req = {
                            'database': database,
                            'table': dataTable,
                            'deviceID': deviceID
                        }

                        data_response = requests.post(link + '/cloud/db/getTableData/', data=json.dumps(req))
                        data_response = data_response.json()  # We get data in string, convert in json/dict.
                        data_response = pd.DataFrame(data_response)  # Convert to pandas df
                        data = data_response[result["properties"]]  # Select columns matching the request

                    except Exception as e:
                        err = "An error occurred when reading input data from database: " + str(e)
                        print(err)
                        return err
                else:
                    try:  # Read the given data
                        #data = np.array(inputs).reshape(1, -1)
                        if num == 2:
                            data = np.array([val1, val2]).reshape(1, -1)
                        elif num == 3:
                            data = np.array([val1, val2, val3]).reshape(1, -1)
                    except Exception as e:
                        err = "An error occurred when reading input data from request: " + str(e)
                        print(err)
                        return err

            except Exception as e:
                err = "Error occurred when reading data: " + str(e)
                print(err)
                return err

        except Exception as e:
            err = "Data Reading Exception: " + str(e)
            print(err)
            return err

        print("Data reading success")

        try:

            try:  # Get pretrained model from database for prediction
                payload = {
                    'table': model_table,
                    'database': database,
                    'id': str(deviceID),
                    'idname': str(idname)
                }

                pickledDict = {}

                print("Requesting Model")
                all_model = requests.post(link + '/cloud/db/getTableDataByID/', data=json.dumps(payload))
                all_model = all_model.json()
                if "modelID" in result:
                    modelID = result["modelID"]
                    all_model = all_model[int(modelID)]
                else:
                    index = len(all_model)
                    all_model = all_model[index-1] # if not provided, take the latest saved model

                if all_model is False:
                    return "Matching Model Not Found, Train your data first!"

                pickledDict = jsonpickle.decode(all_model['model'])

            except Exception as e:
                err = "Error occurred when loading trained model and intermediate variables: " + str(e)
                print(err)
                return err

        except Exception as e:
            err = "An error occured when establishing connection with the database: " + str(e)
            print(err)
            return err

        # Step 4: Predict the data
        try:
            # model predicting
            model = getattr(sys.modules[__name__], algorithm)()

            if algorithm == 'OCSVM':
                result = model.predict(pickledDict['clf'], data)
            else:
                if algorithm == 'KNN':
                    result = model.predict(pickledDict['clf'], pickledDict['in_var'], data)
                elif algorithm == 'KPCA':
                    result = model.predict(pickledDict['clf'], pickledDict['scaler'], pickledDict['in_var'], data)

            # Result Modifications
            pm_result = float(result[0])
            pm_needed = self.moving_average_pm(pm_result)

        except Exception as e:
            err = "Error occurred while predicting: " + str(e)
            print(err)
            return err

        print("Prediction Done.....")

        try:
            if resultTable != "nosave":  # Result has to be saved in some table
                if realTimeFlag == True:
                    payload = {}
                    payload["table"] = resultTable
                    rlt = {}
                    i = 0
                    for c in cols:
                        rlt[c] = inputs[i]
                        i += 1
                    rlt['result'] = int(pm_needed)
                    rlt[str(idname)] = deviceID
                    payload["data"] = rlt
                    payload["id"] = deviceID
                    payload["database"] = database

                    res = requests.post(link + '/cloud/db/writeOneRowInTable/', data=json.dumps(payload))

                    print('Result Saved in %s , PM is: %s' % (payload["table"], pm_needed))
                    result_load = {}
                    result_load["pm_needed"] = int(pm_needed)
                    result_load["pm_result"] = pm_result
                    #result_load = [pm_needed, pm_result]
                    json_result = json.dumps(result_load)
                    return json_result

                else:
                    resultdf = pd.DataFrame(result, columns=["result"])

                    resulttablename = resultTable + "_onRawData_"
                    save_response = pd.concat([data_response, resultdf], axis=1)

                    payload["table"] = resulttablename
                    payload["id"] = deviceID

                    payload["data"] = save_response.to_dict(orient='records')

                    delete_payload = {
                        'table': resulttablename

                    }
                    ret = requests.post(link + '/cloud/db/deleteTable/', data=json.dumps(delete_payload))

                    ret = requests.post(link + '/cloud/db/writeInTable/', data=json.dumps(payload))

                    print('Result Saved in %s ' % (payload["table"]))
                    return ('Result Saved in %s' % (payload["table"]))

            else:
                result_load = {}
                result_load = [pm_needed, pm_result]
                json_result = json.dumps(result_load)
                return json_result

        except Exception as e:
            err = "Error occurred during model saving: " + str(e)
            print(err)
            return err

    def moving_average_pm(self, curr_result):
        global pm_result_list
        if curr_result == 1.0:
            # self.result_list.append(1)
            pm_result_list.append(1)
        else:
            # self.result_list.append(0)
            pm_result_list.append(0)

        # calculate the last 10 result for a moving average
        if len(pm_result_list) == self.moving_aver_size:
            # temp = np.array(self.result_list)
            temp = np.array(pm_result_list)
            result_avg = np.mean(temp)
            # self.result_list.pop(0)
            pm_result_list.pop(0)
            if result_avg < self.pm_weight:
                pm_needed = True
            else:
                pm_needed = False
        else:
            pm_needed = False

        return pm_needed


class GenericPlot(object):
    """
    Generic ML Plot

    # When req sent, data should be already in the table
    @author: Shreyas Gokhale, Orhan Can Görür
    @contact: orhan-can.goeruer@dai-labor.de
    """

    def POST(self):
        """
        POST this is the POST function of the GenericTrain.

        Example of request: (* fields are optional)
        {
            "deviceID" : "c3cedae8-6143-4421-84aa-32e527c6b04e",                   # Device ID. This is NOT a serial number of the data.
            "table" : "raw_velocitypower_in_c3cedae8-6143-4421-84aa-32e527c6b04e",    # Name of the table/collection where your data is stored
            "properties" : ["velocity","power_in"],   # What columns from the data you want for plotting
        *   "database" : "predictive_maintenance" # If you have saved the table in a specific database. It is "Generic" by default.
        }

        @rtype: string
        @return: error type or successful reminder
        """
        print("-----------------------------------")
        print("GenericPlot Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            if "idname" in result:
                idname = result["idname"]
            else:
                idname = "deviceID"

            properties = result["properties"]

            if "database" in result:
                database = result["database"]
            else:
                database = "Generic"

            table = result["table"]  # Input table where test train data is stored

        except Exception as e:
            print("Error occurred when reading data from post: " + str(e))

        # Get Data from the columns mentioned in the request
        try:
            # Creating query in python dict
            req = {
                'table': table,
                'database': database,
            }

            # This is commented out as csv has all the IDs 0. So for now, just get the data using table name.
            # data_response = requests.post(link + '/cloud/db/getTableDataByID/', data=json.dumps(req))


            data_response = requests.post(link + '/cloud/db/getTableData/', data=json.dumps(req))
            data_response = data_response.json()  # We get data in string, convert in json/dict.
            data_response = pd.DataFrame(data_response)  # Convert to pandas df
            # data = data_response[result["properties"]]  # Select columns matching the request
            data_response = data_response.dropna()
            print(data_response)
        except Exception as e:
            print(e)
            return "Error in fetching data from the database"

        try:
            # Pair-wise Scatter Plots
            pp = sns.pairplot(data_response[properties], height=1.8, aspect=1.8,
                              plot_kws=dict(edgecolor="k", linewidth=0.5),
                              diag_kind="kde", diag_kws=dict(shade=True))

            fig = pp.fig
            fig.subplots_adjust(top=0.93, wspace=0.3)
            t = fig.suptitle('Pairwise Plots', fontsize=14)
            plt.show(fig)


        except Exception as e:
            print(e)
            return "Error in plotting data from the database"

        finally:
            print("GenericPlot function finished!")
            print("-----------------------------------")
            return "GenericPlot Success"


class GenericGeneticAlgorithmControl(object):

    def POST(self):
        """

        :return:
        """
        print("-----------------------------------\n")
        print("This is the start of GeneticAlgorithmControl function!")
        print("-----------------------------------")

        try:
            print(web.data())
            result = json.loads(web.data().decode('utf-8'))

            print(result)

            # result should include speed, torque, min_speed, max_speed, int_speed, alpha, beta and ng
            power_type = int(result["power_type"])
            if "ID" in result:
                agent_id = int(result["ID"])
            power = float(result["power"])
            speed = float(result["speed"])
            torque = float(result["torque"])
            min_speed = float(result["min_speed"])
            max_speed = float(result["max_speed"])
            int_speed = float(result["int_speed"])
            alpha = float(result["alpha"])
            beta = float(result["beta"])
            ng = int(result["ng"])
            error = float(result["error"])

        except Exception as e:
            print("An error has occured in reading genetic algorithm parameters " + str(e))
            return e

        try:
            geneticAlgorithmControl = GeneticAlgorithmControl(power_type=power_type, error=error, \
                                                              power=power, speed=speed, torque=torque, \
                                                              min_speed=min_speed, max_speed=max_speed, \
                                                              int_speed=int_speed, alpha=alpha, beta=beta, ng=ng)
            output = geneticAlgorithmControl.run()
            return output

        except Exception as e:
            print("An error has occured when performing genetic algorithm: " + str(e))
            return e

        finally:
            print("-----------------------------------\n")
            print("This is the end of GeneticAlgorithmControl function!")
            print("-----------------------------------")
