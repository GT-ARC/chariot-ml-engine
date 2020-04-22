#!/usr/bin/env python3

"""
CloudAPI using MongoDB for ChariotCloud

@author:  Shreyas Gokhale, Orhan Can Görür
@contact: orhan-can.goeruer@dai-labor.de
"""

import web
import requests
import json
from argparse import ArgumentParser
import GenericMachineLearningAPI
import MongoDBDriver
import sys
import pandas as pd
import os

# Global variables and arugument passing

parser = ArgumentParser()
parser.add_argument("-u", "--user", dest="user", nargs='?', const='Invalid',\
    help="defines the username for MongoDB access", metavar="USERNAME")
parser.add_argument("-p", "--password", dest="password", nargs='?', const='Invalid',\
    help="defines the password for MongoDB access", metavar="PASSWORD")
parser.add_argument("-H", "--host", dest="host", nargs='?', const='Invalid',\
    help="Host for MongoDB", metavar="HOST")
parser.add_argument("-l", "--http-link", dest="weblink", nargs='?', const='Invalid',\
    help="defines the link for http access", metavar="LINK")
parser.add_argument("-db", "--authentication-database", dest="authdb", nargs='?', const='Invalid',\
    help="authentication database")
parser.add_argument("-port", "--mongo-db-port", dest="mongodbport", nargs='?', const='Invalid',\
    help="MongoDB port")

args = parser.parse_args()

user = args.user
password = args.password
weblink = args.weblink

if weblink is None:
    weblink = "0.0.0.0:8080"

host = args.host
if host is None:
    host = "localhost"

authdb = args.authdb
if authdb is None:
    authdb = "ml"

monogdbport = args.mongodbport
if monogdbport is None:
    monogdbport = "27017"


weblink_list = weblink.split(":")
webhost  = weblink_list[0]
webport  = weblink_list[1]


link = "mongodb://" + host + ":" + monogdbport
print (link)

class WebServer():
    """
    Webserver to server links

    @author: Shreyas Gokhale, Orhan Can Görür
    @contact: orhan-can.goeruer@dai-labor.de
    """
    def __init__(self):
        self.weblink = weblink
        self.urls = (
            # Barebone APIs
            '/cloud/db/getRecords/', 'MDBGetRecords',
            '/cloud/db/setRecords/', 'MDBSetRecords',

            # DB APIs
            '/cloud/db/getTableData/', 'GetTableData',
            '/cloud/db/getTableDataByID/', 'GetTableDataByID',
            '/cloud/db/getLatestData/', 'GetLatestData',
            '/cloud/db/updateOneInTable/', 'UpdateOneInTable',          # Rename from updateTableById
            '/cloud/db/copyCSVInTable/', 'CopyCSVInTable',              # Rename from StoreDataInTable
            '/cloud/db/CopyCSVToMLDB/', 'CopyCSVToMLDB',
            '/cloud/db/writeInTable/', 'WriteInTable',                  # Rename from writeInTable
            '/cloud/db/writeOneRowInTable/', 'WriteOneRowInTable',      # Rename from writeRowInTable
            '/cloud/db/deleteDatabase/', 'DeleteDatabase',
            '/cloud/db/deleteTable/', 'DeleteTable',                    # Rename DeleteTableEntries

            # # KMS Interface APIs
            # '/cloud/db/kms/copyKMSToMLDB/', 'KMSInterfaceAPI.CopyKMSToMLDB',    # Copy KMS DB History Data To ML DB
            # '/cloud/db/kms/copyMLDBToKMS/', 'KMSInterfaceAPI.CopyMLDBToKMS',    # TODO: not implemented. Copy ML DB table data to KMS DB
            # '/cloud/db/kms/copyCSVToKMS/', 'KMSInterfaceAPI.CopyCSVToKMS',      # Copy Data from CSV to MLDB
            # '/cloud/db/kms/registerDeviceModel/','KMSInterfaceAPI.RegisterDeviceModel',
            # '/cloud/db/kms/addOneRowToKMS/', 'KMSInterfaceAPI.AddOneRowToKMS',
            # '/cloud/db/kms/getLatestKMSData/', 'KMSInterfaceAPI.getLatestKMSData',
            # '/cloud/db/kms/setDeviceID/', 'KMSInterfaceAPI.SetDeviceID',
            # '/cloud/db/kms/getDeviceID/', 'KMSInterfaceAPI.GetDeviceID',

            # ML APIs
            '/cloud/ml/ga/GAControl/', 'GenericMachineLearningAPI.GenericGeneticAlgorithmControl',
            '/cloud/ml/generic/train/', 'GenericMachineLearningAPI.GenericTrain',
            '/cloud/ml/generic/predict/', 'GenericMachineLearningAPI.GenericPredict',
            '/cloud/ml/generic/plot/', 'GenericMachineLearningAPI.GenericPlot'

        )

    def startApp(self):
        """
        Starts application
        :return:
        """
        print("Starting ChariotCloud API")
        app = web.application(self.urls, globals())
        # set the ip and port for the API
        app2 = web.httpserver.runsimple(app.wsgifunc(), (webhost, int(webport)))
        return app2



# Function to make MongoDBAPI object
def makeObject(link=link, dbName = "ChariotCloudDB", dbCollection="ChariotCloudTable"):
    mdbobject = MongoDBDriver.MongoDBAPI(link, user, password, authdb)
    mdbobject.defineDB(dbName)
    mdbobject.defineCollection(dbCollection)
    return mdbobject

class GetTableData(object):
    """
    Gets data from a table / collection. Returns data if found. Otherwise, returns an error.

    WARNING: Data can be non consistent if your saving method is wrong!
    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """

    def POST(self):
        """
        Get Table / Collection from given table name

        GET function
        @return: All information about the querry
        """

        print("-----------------------------------")
        print("GetTableData Started!")
        result = json.loads(web.data().decode('utf-8'))

        # Table Name
        table = str(result["table"])
        # If user has provided with database name in the request
        if "database" in result:
            database = str(result["database"])
        else:
            database = "predictive_maintenance"

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)

            if mdbobject.checkCollectionExists() == False:
                print("Invalid/ Nonexistent Collection")
                return "Invalid/ Nonexistent Collection"

            maskquery = {'_id' :0}                          # Don't return Default _IDs

            # Returns a python list of all the collection
            res = mdbobject.findRecord(maskingquery=maskquery)

            jsonData = json.dumps(res)
            return jsonData

        except Exception as e:
            print("Error in getTableData: " + str(e))
            return e

        finally:
            print("GetTableData Finished")
            print("-----------------------------------")


class GetTableDataByID(object):
    """
    Gets data from a table / collection. By ID. Returns data if found, otherwise, none.

    WARNING: Data can be non consistent if your saving method is wrong!
    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """

    def POST(self):
        """
        Get Table / Collection from given table name

        GET function
        @type web: string
        @param web: Tablename / Collection Name
        @rtype: web response
        @return: All information about the querry
        """

        print("-----------------------------------")
        print("GetTableDataByID Started!")

        result = json.loads(web.data().decode('utf-8'))

        # Table Name
        table = str(result["table"])
        # If user has provided with database name in the request
        if "database" in result:
            database = str(result["database"])
        else:
            database = "predictive_maintenance"

        # If user has a specific naming system for ID.
        if "idname" in result:
            idname = result["idname"]
        else:
            idname = "ID"

        id = result["id"]

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)

            if mdbobject.checkCollectionExists() == False:
                print("Invalid/ Nonexistent Collection")
                return "Invalid/ Nonexistent Collection"

            query = {}
            query[idname] = id
            maskquery = {'_id' :0}

            # Returns a python list of all entries where id is given
            res = mdbobject.findRecord(query,maskquery)

            jsonData = json.dumps(res)
            return jsonData

        except Exception as e:
            print("Error in getTableDataByID: " + str(e))
            return e

        finally:
            print("GetTableDataByID Finished")
            print("-----------------------------------")


class GetLatestData(object):
    """
    Gets latest data in the table

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """

    def POST(self):
        """
        GET Function
        @type web: string
        @param web: The tablename and id
        @rtype: web response
        @return: All information about the querry
        """
        print("-----------------------------------")
        print("GetLatestData Started!")

        result = json.loads(web.data().decode('utf-8'))
        # Table Name
        table = str(result["table"])
        # If user has provided with database name in the request
        if "database" in result:
            database = str(result["database"])
        else:
            database = "predictive_maintenance"

        # If user has a specific naming system for ID.
        if "idname" in result:
            idname = result["idname"]
        else:
            idname = "ID"

        id = result["id"]

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)

            if mdbobject.checkCollectionExists() == False:
                return "Unknown / empty Collection"

            query = {}
            query[idname] = int(id)

            # Returns a list of all entries where id is given
            res = mdbobject.findLatestRecord(query)

            if(res == None):
                return "No Data found"
            jsonData = json.dumps(res)
            return jsonData

        except Exception as e:
            print(" Error in GetLatestData " + str(e))
            return e

        finally:
            print("GetLatestTableData Finished")
            print("-----------------------------------")


class CopyCSVInTable(object):
    """
    Load data from a text (csv) file and store it into the collection.

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """
    def POST(self):
        """
        POST this is the POST function of the CopyCSVInTable
        You either have to define header names in the csv, or give column_names in the request.
        """

        print("-----------------------------------")
        print("CopyCSVInTable Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            # Table Name
            table = str(result["table"])
            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "ChariotCloud"

            path = result["path"]

            # If list of columns is given
            if "properties" in result:
                col_list = result["properties"]
                nameFlag =  True
            else:
                nameFlag = False

            # If you want to explicitly add index column (with given index name) in the table
            if "index_name" in result:
                index_name = str(result["index_name"])
                addIndexFlag = True
            else:
                addIndexFlag = False


            # If you want to add static columns to every data field. (Given by column and value JSON)
            if "static_col" in result:
                static_col = result["static_col"]
                staticColFlag = True
            else:
                staticColFlag = False

        except Exception as e:
            print("Cannot read table name and text file path. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)

            # IF table already exists, drop it.
            # If we comment this, the data will get combined. (Might be helpful for some functionality)

            if(mdbobject.checkCollectionExists()== True):
                mdbobject.dropCollection()

        except Exception as e:
            print("An error occured: " + str(e))
            return e

        try:
            with open(path, 'r') as f:
                if (nameFlag == True):
                    df = pd.read_csv(f, names=col_list)
                else:
                    df = pd.read_csv(f)

                if staticColFlag is True:
                    try:
                        # df.assign(**static_col)           # This cool function does not work below python 3.6  So we resort to other methods
                        for key in static_col:
                            df[key] = static_col[key]
                    except Exception as e:
                        print("Error in assigning static columns: " + str(e))

                if addIndexFlag is True:
                    try:
                        df[index_name] = df.index
                    except Exception as e:
                        print("Error in assigning index " + str(e))

                data_json = json.loads(df.to_json(orient='records'))

                success = mdbobject.insertRecords(data_json)

            if success == True:
                return "Store data from %s in table %s successful." % (path, table)
            else:
                return "Store data from %s in table %s unsuccessful." % (path, table)


        except Exception as e:
            print ("An error occured in reading csv file or query execution: " + str(e))
            return e

        finally:
            print("CopyCSVInTable finished!")
            print("-----------------------------------")


class CopyCSVToMLDB(object):
    """
    Copy Data from CSV to KMS DB

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """
    global deviceID

    def POST(self):
        """
        You either have to define header names in the csv, or give column_names in the request.
        """

        print("-----------------------------------")
        print("CopyCSVToMLDB Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            path = result["path"]
            deviceID = str(result["deviceID"])
            if "timestamp" in result:
                timestampname = result["timestamp"]
            else:
                timestampname = "timestamp"
            # If list of columns is given
            if "properties" in result:
                col_list = result["properties"]
                nameFlag =  True
            else:
                nameFlag = False

            # Table Names to save under MLDB
            property_id = ""
            for property_key in col_list:
                property_id = property_id + str(property_key)
            table = "raw" + "_" + property_id +  "_" + deviceID

            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "predictive_maintenance"

            if "idname" in result:
                idname = result["idname"]
            else:
                idname = "ID"

        except Exception as e:
            print("Cannot read csv file path. Error: "+ str(e))
            return e

        # reading from the CSV and writing to df object
        try:
            with open(path, 'r') as f:
                df = pd.read_csv(f)
                df = df[1:]
                # Set Timestamp as index
                #df.drop_duplicates(inplace = True)
                if timestampname in df.columns:
                    df.set_index(timestampname, inplace= True)
                # Iterate over all the colunmns in df
                for col_name in df.columns:
                    if nameFlag and not (col_name in col_list):
                        del df[col_name]

        except Exception as e:
            err = "Error reading and serializing CSV data: "+ str(e)
            print(err)
            return err

        df = df.dropna()

        # Creating a table under MLDB
        try:
            # Create Object for target database
            mdbobject = makeObject(dbName = database,dbCollection = table)

            if(mdbobject.checkCollectionExists()== True):
                mdbobject.dropCollection()

        except Exception as e:
            print("Error in creating new collection: "+ str(e))
            return e

        # writing the df object into the MLDB table
        try:
            data_json = json.loads(df.to_json(orient='records'))
            success = mdbobject.insertRecords(data_json)
            if success == True:
                return "Store data from %s in table %s successful." % (path, table)
            else:
                return "Store data from %s in table %s unsuccessful." % (path, table)

        except Exception as e:
            print("Error in CopyCSVToMLDB: " + str(e))
            return e

        finally:
            print("CopyCSVToMLDB finished!")
            print("-----------------------------------")
            return success


class WriteInTable(object):
    """
    Load data from request and store it into the collection.

    The data must be expressed as list of entries to be written, even if its a single record.

    For example:

    "data" :
        [
            {
                "name":"Blue","age":0

            },
            {
                "name":"Green","age":2

            }
        ]

    If you want to directly add an entry without list, use WriteOneRowInTable API

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """
    def POST(self):
        """
        POST this is the POST function of the WriteInTable
        """

        print("-----------------------------------")
        print("WriteInTable Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            # Table Name
            table = str(result["table"])
            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "ChariotCloud"

            data  =  result["data"]

        except Exception as e:
            print("Cannot read table name and data. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)
            mdbobject.checkCollectionExists()

        except Exception as e:
            print("An error occured when creating object: " + str(e))
            return e

        try:
            success = mdbobject.insertRecords(data)

            if success == True:
                return "Store data in table %s successful."%( table)
            else:
                return "Store data in table %s unsuccessful."%( table)

        except Exception as e:
            print ("An error occured in query execution: " + str(e))
            return e

        finally:
            print("WriteInTable finished!")
            print("-----------------------------------")


class WriteOneRowInTable(object):
    """
    Load data from request and store it into the collection.

    Example for Data:
    "data" :
        {
            "name":"Red","age":6
        }

    OR

    "indices" = ["name","age"]
    "values" = ["Red",6]

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """
    def POST(self):
        """
        POST this is the POST function of the WriteOneRowInTable
        """

        print("-----------------------------------")
        print("WriteOneRowInTable Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            # Table Name
            table = str(result["table"])
            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "predictive_maintenance"

            if "indices" in result:
                indices = result["indices"]
                data = {}
                if "values" in result:
                    values = result["values"]
                    for i in indices:
                        data[i] = values[i]

                else:
                    print("No values found, please enter values of the indices")
                    return "No values found, please enter values of the indices"

            else:
                data  =  result["data"]


        except Exception as e:
            print("Cannot read table name and data. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)
            mdbobject.checkCollectionExists()

        except Exception as e:
            print("An error occured when creating object: " + str(e))
            return e

        try:
            success = mdbobject.insertOneRecord(data)

            if success == True:
                return "Store data in table %s successful." % (table)
            else:
                return "Store data in table %s unsuccessful." % (table)

        except Exception as e:
            print ("An error occured in query execution: " + str(e))
            return e

        finally:
            print("writeRowInTable finished!")
            print("-----------------------------------")




class UpdateOneInTable(object):
    """
    Update the first instance of given key (or keys) to given value(s)

    Example for Data:

    "key" :
    {
        "speed" : 1.1
        }
    "value" : 23

    @author: Shreyas Gokhale
    @contact: s.gokhale@campus.tu-berlin.de
    """
    def POST(self):
        """
        POST this is the POST function of the WriteOneRowInTable
        """

        print("-----------------------------------")
        print("UpdateOneInTable Started")

        try:
            result = json.loads(web.data().decode('utf-8'))

            # Table Name
            table = str(result["table"])
            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "ChariotCloud"

            key  =  result["key"]
            value = result["value"]

        except Exception as e:
            print("Cannot read table name and data. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)
            mdbobject.checkCollectionExists()

        except Exception as e:
            print("An error occured when creating object: " + str(e))
            return e

        try:
            success = mdbobject.updateOneRecord(key,value)

            if success == True:
                return "Update data in table %s successful." % (table)
            else:
                return "Update data in table %s unsuccessful." % (table)

        except Exception as e:
            print ("An error occured in query execution: " + str(e))
            return e

        finally:
            print("UpdateOneInTable finished!")
            print("-----------------------------------")


class DeleteTable(object):
    def POST(self):
        """
        Deletes given table (collection) from the database
        :return:
        """
        print("-----------------------------------")
        print("DeleteTable Started")
        try:
            result = json.loads(web.data().decode('utf-8'))

            # Table Name
            table = str(result["table"])
            # If user has provided with database name in the request
            if "database" in result:
                database = str(result["database"])
            else:
                database = "ChariotCloud"

        except Exception as e:
            print("Cannot read table name and text file path. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName = database,dbCollection = table)
            if (mdbobject.dropCollection()==True):
                return "Table Drop operation succesful"
            else:
                return "Table does not exist"

        except Exception as e:
            print("ERROR:" + str(e))

        finally:
            print("DeleteTable finished!")
            print("-----------------------------------")


class DeleteDatabase(object):
    """
    Deletes Given Database
    """
    def POST(self):
        print("Begin DeleteDatabase:")
        try:
            result = json.loads(web.data().decode('utf-8'))
            database = result["database"]

        except Exception as e:
            print("Cannot read database name and text file path. Error: " + str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(dbName=database)
            if (mdbobject.dropDatabase() == True):
                return "Database Drop operation succesful"
            else:
                return "Database does not exist"

        except Exception as e:
            print("ERROR:" + str(e))

        finally:
            print("DeleteDatabase finished!")
            print("-----------------------------------")


class MDBSetRecords(object):
    def POST(self):
        try:
            result = json.loads(web.data().decode('utf-8'))
            dbName =  result["database"]
            dbCollection = result ["table"]
            query = result ["query"]


        except Exception as e:
            print("Cannot read request. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(link,dbName,dbCollection)
            res = mdbobject.insertOneRecord(query)
            return res

        except Exception as e:
            print("ERROR:" + str(e))


class MDBGetRecords(object):
    def GET(self):
        try:
            result = json.loads(web.data().decode('utf-8'))
            dbName =  result["database"]
            dbCollection = result ["table"]
            query = result["query"]

        except Exception as e:
            print("Cannot read request. Error: "+ str(e))
            return e

        try:
            # Create Object
            mdbobject = makeObject(link,dbName,dbCollection)
            res = mdbobject.findRecord(query)
            return res

        except Exception as e:
            print("ERROR:" + str(e))


if __name__ == '__main__':
    try:
        server = WebServer()
        app =server.startApp()
        # api_doc(server, config_path='./config/test.yaml', url_prefix='/api/doc', title='API doc')


    except Exception as e:
        print("Cannot start DataBaseAPIServer: " + str(e))
