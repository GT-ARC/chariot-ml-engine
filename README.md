Machine Learning Component with API for Generic ML Algorithms
============
Version 1.0, uploaded on 08.04.2020 as an open source project
- Orhan Can Görür (orhan-can.goeruer@dai-labor.de)
- Shreyas Gokhale (shreyas6gokhale@gmail.com)
- Xin Yu (marvelous.islander@gmail.com)

Below are the instructions to successfully install and run the ML component and its API both for accessing generic ML functionalities provided for CHARIOT and accessing the MongoDB database.
This API supports various MongoDB operations and several machine learning based algorithms (mostly for generic predictive maintenance operation for the project). We would like to also acknowledge the contributions of the students in this project: Peng Qian (qian peng1994@campus.tu-berlin.de), Stefan Klemencic (stefan.klemencic@campus.tu-berlin.de), Simon Torka, Antonia Düker.


## Installation and Running
Below needs to be available on the server running the project. However, we made a docker available that automatically installs all the dependencies and runs the API.

### Docker Install and Run
Run ``` deploy.sh```
* Note that for this you need to have `docker-compose` installed. We need to install latest stable version to run compose files with version `3`. Please follow the steps here: https://docs.docker.com/compose/install/

This will deploy `docker-compose.yaml` file with the services below:
- mongo: with the credentials as given under docker-compose.yaml file
- chariot-cloud: under `ml-db-api`. Note that it requires a Mongo DB to run. Update the credentials according to your mongo image and credentials.
- swagger: under `swagger`. Note that it uses `mlcloud.json` for the documentation. It is also possible to deploy swagger separately: run `deploy_swaggerOnly.sh`.

### Without Docker
Python >= 3.5 should be installed. Python 2.x is not supported due to the usage of the open-source library Gaft.
#### Additional Libraries
Please see the `requirements.txt` file. Below are the list of some packages that are crucial for the project:
* web (run sudo apt install python-webpy, and `pip3 install git+https://github.com/webpy/webpy#egg=web.py` see http://webpy.org/) for database API
* requests (`pip3 install --user requests`)
* mongodb (https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)
* pymongo (`pip3 install --user pymongo` See https://api.mongodb.com/python/current/installation.html),
* json (`pip3 install --user json`)
* sklearn (Scikitlearn) (http://scikit-learn.org/stable/install.html)
* Matplotlib (`sudo apt-get install python3-matplotlib`)
* seaborn-0.9.1 (`pip3 install --user seaborn`)

#### Running:
- First, you need to initialize and run a mongo DB on your PC. If an example credentials would be: `username: mldb`, `pass: chariot`, `authentication database: admin`, `port: 27017`.

- In order to run the API, only ChariotCloudAPI.py is needed. To run:
```
# specify the hostname for MongoDB access using -H option     
# specify the link for RESTFul http access using -l option
# by default (without any options), api runs on localhost
# an example given that Mongo DB is initialized as above:
python3 ChariotCloudAPI.py -port 27017 -u mldb -p chariot -db admin -H localhost
```

## Supported functionality:
Below are the list of services provided. They are also documented via Swagger. To deploy, please refer to the

#### ML APIs
A Postman collection is also provided for an easy call interface for the services: `mlAPI_postman_collection.json`

A simple example of operation would be to first call `/cloud/db/CopyCSVToMLDB/` to save a dataset to the ML DB, then train `/cloud/ml/generic/train/` by specifying the dataset columns to work on and to predict `/cloud/ml/generic/predict/` again by specifying the the trained model algorithm parameters and a data point. Example file and a training is provided for a predictive maintenance application on a servo motor. We have provided One Class Support Vector Machine (OCSVM), k-Nearest Neighbor (k-NN), and a kernel Principle Component Analysis (KPCA) for the anomaly detection toward Predictive maintenance.

* Generic Train

```
 '/cloud/ml/generic/train/'

{
    "algorithm" : "OCSVM",              # Algorithm you want to use to train the data. Alternatively: KNN, KPCA
    "deviceID" : "123456",                           # Device ID. Any string
*   "idname": "deviceID"                # What do you call your ID. If not provided, it is "deviceID" by default.
    "para" :[0.01,0.5],                 # Parameters for ML algorithm
    "properties" : ["velocity","power_in"],           # What variables from the data you want for training
*   "database" : "predictive_maintenance"         # If you want to save table in a specific database.
}

Returns status.

```

* Generic Predict

```
'/cloud/ml/generic/predict/'

{
    "algorithm" : "OCSVM",              # Algorithm you want to use to train the data.  Alternatively: KNN, KPCA
    "para" :[0.01,0.5],                 # Parameters used for training model that want to be used.
    "properties" : ["velocity","power_in"],           # The variables you already trained the data for.
    "value": [ 31.00 , 5]               # the data point for the variables given above.
*   "idname": "deviceID"                # What do you call your ID. If not provided, it is "deviceID" by default.
*   "resulttable" : "results"   # Table Name to save results. If = "nosave", don't save - just return the results to the user. Or it saves to a table by default
*   "database": "predictive_maintenance"          # If you want to save the results in the database.
}

Returns status and the prediction result (true or false) on the given value.

```

#### ML Database APIs
A Postman collection is also provided for an easy call interface for the services: `mongoDB_postman_collection.json`

* CopyCSVToMLDB

```
'/cloud/db/CopyCSVToMLDB/'

{
    "deviceID": "123456",
    "path" : "<path_to_CSV_file_holding_source_data>",  # Make sure the first row of CSV file holds the variable names. An example file provided under /ml-mdb-api/app-docker/datafiles/real_motor_data_0320.csv
    "properties" : ["<variable-1>", "<variable-2>"],    # Make sure the variable names are the ones in CSV file. Use "velocity" and "power_in" for the example above.
  "database": "predictive_maintenance"    # Any database name put will be created!
}

Returns process status
```

* GetTableData

```
'/cloud/db/getTableData/'
{
    "table": "GenericData",             # Name of the table where you want the data to get.
*   "database": "ChariotCloud"          # If you want to get from a specific database. It is "ChariotCloud" by default.

}

Returns Table in JSON format.
```

* GetTableByID

```
'/cloud/db/getTableDataByID/'

{
    "table": "GenericData",             # Name of the table where you want the data to get.
    "id" : 0,                           # Device ID. This is NOT a serial number of the data.
*   "idname": "deviceID"                # What do you call your ID. If not provided, it is "ID" by default.
*   "database": "ChariotCloud"          # If you want to get from a specific database. It is "ChariotCloud" by default.

}

Returns Table in JSON format.
```

* GetLatestData

```
'/cloud/db/getLatestData/'
{
    "table": "GenericData",             # Name of the table where you want the data to get.
    "id" : 0,                           # Device ID. This is NOT a serial number of the data.
*   "idname": "deviceID"                # What do you call your ID. If not provided, it is "ID" by default.
*   "database": "ChariotCloud"          # If you want to get from a specific database. It is "ChariotCloud" by default.
}

Returns latest entry of the table in JSON format.
```

* CopyCSVInTable

```
'/cloud/db/copyCSVInTable/'
{
    "table": "GenericData",              # Name of the table where you want the data to be stored.
    "path" : "real_motor_data_0320.csv", # CSV file path
    # NOTE: You either have to define header names in the csv, or give column_names in the request.
    "col" : ["deviceID" , "speed", "velocity",  "torque"],
*   "database": "ChariotCloud"           # If you want to get from a specific database. It is "ChariotCloud" by default.
}

Returns result.
```

* WriteInTable

```
'/cloud/db/writeInTable/'
{
    "table": "GenericData",              # Name of the table where you want the data to be stored.
*   "database": "ChariotCloud"           # If you want to get from a specific database. It is "ChariotCloud" by default.
    "data" : [                           # JSON array of records
                {
                    "name":"Blue","age":0

                },
                {
                    "name":"Green","age":2

                }
            ]
    # The data must be expressed as list of entries to be written, even if its a single record.
    # If you want to directly add an entry without list, use WriteOneRowInTable API

}
Returns result.
```

* WriteOneRowInTable

```
'/cloud/db/writeOneRowInTable/'
{
    "table": "GenericData",              # Name of the table where you want the data to be stored.
*   "database": "ChariotCloud"           # If you want to get from a specific database. It is "ChariotCloud" by default.
    "data":                              # Single JSON record of enrty to be added
            {
                "deviceID": 0,
                "torque" : 0,
                "speed" : 34.2325,
                "dist" : 1.1515
            }
}
Returns result.
```


* UpdateOneInTable

```
'/cloud/db/uodateOneInTable/'
{
    "table": "GenericData",              # Name of the table where you want the data to be updated.
*   "database": "ChariotCloud"           # If you want to update from a specific database. It is "ChariotCloud" by default.
    "key":                               # You want to update entry having this values. If multiple key exist, first matching key will be updated.
            {
                "deviceID": 0,
                "speed" : 34.2325,
                "dist" : 1.1515
            },
    "value":                             # Whatever part of the record you want to change
            {
                "torque" : 60,
                "dist" : 13513
            }    
}
Returns result.
```

* Delete Database

```
'/cloud/db/deleteDatabase/'
{
   "database": "ChariotCloud"           # Name of the database to be deleted.
}
Returns result.
```

* Delete Table

```
'/cloud/db/deleteTable/'
{
   "table": "GenericData"               # Name of the table to be deleted.
*  "database": "ChariotCloud"           # If you want to delete from a specific database. It is "ChariotCloud" by default.

}
Returns result.
```





# TODOs


##### TODO API Documentation for python requests


```
# getTableDataByID
payload = {}
payload["table"] = "MotorMaintenanceList_1"
payload["idname"] = "id"
payload["id"] = "1"
r = requests.get('http://0.0.0.0:8080/cloud/db/getTableDataByID/', params = payload)
# http://0.0.0.0:8080/cloud/db/getTableDataByID/?table=MotorMaintenanceList_1&idname=id&id=1
```
```
# getTableData
payload = {}
payload["table"] = "MotorMaintenanceList_1"
r = requests.get('http://0.0.0.0:8080/cloud/db/getTableData/', params = payload)
# http://0.0.0.0:8080/cloud/db/getTableData/?table=MotorMaintenanceList_1
```
```
# getLatestData
payload = {}
payload["table"] = "MotorMaintenanceList_1"
payload["log_list_id"] = "1"
r = requests.get('http://0.0.0.0:8080/cloud/db/getLatestData/', params = payload)
# http://0.0.0.0:8080/cloud/db/getLatestData/?table=MotorMaintenanceList_1&log_list_id=1
# http://10.0.2.83:8080/cloud/db/getLatestData/?table=api_motorlogdata&log_list_id=0
```
```
# checkTableEntryNum
payload = {}
payload["table"] = "MotorMaintenanceList_1"
r = requests.get('http://0.0.0.0:8080/cloud/db/checkTableEntryNum/', params = payload)
# http://0.0.0.0:8080/cloud/db/checkTableEntryNum/?table=MotorMaintenanceList_1
```
```
# updateTableDataByID
load = {}
load["table"] = "MotorMaintenanceList_1"
load["indices"] = ["log_list_id","speed","power_in","result"]
load["values"] = ["2", "30", "2.5", "0"]
load["idname"] = "id"
load["id"] = "1"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/updateTableDataByID/', data=json_load)
```
```
# writeRowInTable
load = {}
load["table"] = "MotorMaintenanceList_1"
load["indices"] = ["log_list_id","speed","power_in","result"]
load["values"] = ["1", "50", "3.5", "1"]
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/writeRowInTable/', data=json_load)
```
```
# storeDataInTable
load = {}
load["table"] = "TrainingList_1"
load["path"] = "output_0.5_conveyor_new_setting_combined.csv"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/storeDataInTable/', data=json_load)
```
```
load = {}
load["table"] = "Test"
load["path"] = "testset.csv"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/storeDataInTable/', data=json_load)
```
```
# deleteDatabase
load = {}
load["database"] = "cloud"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/deleteDatabase/', data=json_load)
```
```
# deleteDatabase
load = {}
load["database"] = "cloud"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/deleteDatabase/', data=json_load)
```
```
# createDatabase
load = {}
load["database"] = "new_test"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/createDatabase/', data=json_load)
```
```
# deleteTableDataByID
load = {}
load["table"] = "MotorMaintenanceList_1"
load["idname"] = "id"
load["id"] = "1"
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/db/deleteTableDataByID/', data=json_load)
```
```
# deleteTableEntries
payload = {}
payload["table"] = "MotorMaintenanceList_1"
r = requests.get('http://0.0.0.0:8080/cloud/db/deleteTableEntries/', params = payload)
# http://0.0.0.0:8080/cloud/db/deleteTableEntries/?table=MotorMaintenanceList_1
```

```
# GenericPredict
# This is to do prediction on history data (historical or real time) and saves on another table
load = {}
load["algorithm"] = "OCSVM"				# KNN / OCSCM / KPCA
load["ID"] = 1 							# Unique ID of your agent (or device/application)
load["col"] = ["speed","power_in"] 		# Data Names
#Either
load["table"] = "TrainingList_1" 		# DB table where the data is saved
#Or
load["value"] = "[20,0.9]" 				# Real Time value for predictive maintenance
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/ml/generic/predict/', data=json_load)
# If realtime value is given, it returns if PM is needed or not.
```
```
# MotorFingerprintTrain
load = {}
load["algorithm"] = "OCSVM"
load["motorID"] = 1
load["para"] = [0.01,1.5]
load["table"] = "TrainingList_1"
load["col"] = ["speed","power_in"]
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/ml/m/train/', data=json_load)
```
```
# MotorFingerprintPredict
# This is to do prediction on history data (not real time) and saves on another table
load = {}
load["algorithm"] = "OCSVM"
load["num"] = 2
load["motorID"] = 1
load["para"] = [0.01,1.5]
load["table"] = "Test"
load["col"] = ["speed","power_in"]
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/ml/m/predict/', data=json_load)
```
```
# MotorMaintenanceRealtimePredict
# Real time prediction. ALGORITHM: "OCSVM", "KPCA", "KNN". NUM: "2" (power-speed), "3" (power-speed-torque / doesnt work)
load = {}
load["algorithm"] = "OCSVM"
load["motorID"] = 1
load["num"] = 2
load["col"] = ["speed","power_in"]
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/ml/m/realtimePredict/', data=json_load)
# Ex. json_load= {"motorID": 1, "num": 2, "col": ["speed", "power_in"], "algorithm": "OCSVM"}
```
```
# MotorMaintenanceVisualization
load = {}
load["algorithm"] = "OCSVM"
load["motorID"] = 1
load["col"] = ["speed","power_in"]
load["table"] = ["TrainingList_1","Test_OCSVM_predicted_1"]
json_load = json.dumps(load)
r = requests.post('http://0.0.0.0:8080/cloud/ml/m/visualization/', data=json_load)
```
```
# MotorGeneticAlgorithmControl
payload = {}
payload["speed"] = 20
payload["torque"] =  0.1    
payload["max_speed"] = 23
payload["int_speed"] = 2
payload["alpha"] = 0.3
payload["beta"] = 1
payload["ng"] = 50
r = requests.get('http://0.0.0.0:8080/cloud/ml/ga/GAControl/', params = payload)
# http://0.0.0.0:8080/cloud/ml/ga/GAControl/?torque=0.1&ng=50&beta=1&max_speed=23&log_list_id=1&int_speed=2&alpha=0.3&table=MotorMaintenanceList_1&speed=20
```

### TODO support for applications

#### Supported Applications
You can also run this API with other applications, such as predictive_maintenance_app.py and adaptive_control_app.py. They are under dobot_conveyor_unit project. To run predictive_maintenance_app.py:
```
# If ChariotCloudAPI.py is already running on chariot cloud vm
# On your local machine, run the following.
# specify the link that provides http accesses on cloud after '-l'. e.g. 10.0.2.83:8080
# Specify motor ID after '-m': It is integer "0" for real motor, and integer "1" or others for simulation motor
# Indicate storing new data or not using '-s'; if not indicated, the default value is 'False'
# Indicate training new model or not using '-t'; if not indicated, the default value is 'False'
source <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/devel/setup.bash
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/src/predictive_maintenance_app/
python predictive_maintenance_app_cloud.py -l 10.0.2.83:8080 -m MOTOR_ID -s -t

# If ChariotCloudAPI.py is already running on local
# On local, run the following.
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/src/predictive_maintenance_app/
python predictive_maintenance_app.py -l 10.0.2.83:8080 -m MOTOR_ID -s -t
```
To run adaptive_control_app.py:
```
# If ChariotCloudAPI.py is already running on chariot cloud vm
# On your local machine, run the following.
# specify the link that provides http accesses on cloud after '-l'. e.g. 10.0.2.83:8080
# Specify motor ID after '-m': It is integer "0" for real motor, and integer "1" or others for simulation motor
# Specify POWER_TYPE: power calculation type after '-p'; 0: real motor, 1: sim motor with noises, 2: sim motor without noises
# Indicate running rule-based mechanism on top of genetic algorithm by '-r'; if not indicated, the default value is 'False'
source <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/devel/setup.bash
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/src/adaptive_control_app/
python adaptive_control_app_cloud.py -l 10.0.2.83:8080 -m MOTOR_ID -p POWER_TYPE -r

# If ChariotCloudAPI.py is running on the same PC that runs this app
# On local, run the following.
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/src/adaptive_control_app/
python adaptive_control_app.py -l 10.0.2.83:8080 -m MOTOR_ID -p POWER_TYPE -r
# To start the application service:
rosservice call /adaptive_control_app/continuous_adaptive_control "{isEnabled: true, motorID: 1, param1: 1.2, param2: 1.0, intSpeed: 0.5, Kp: 2.0, Ki: 5.0, Kd: 1.0}"

# In case of rule based mechanism causing the motor to stop
# manually set a speed for the motor in range [6,25]. for example, for simulation motor:
rosservice call /conveyor_belt_agent_1/setMotorSpeedSimMid "isEnabled: true
speed: 25.0"
# re-enable the continuous_adaptive_control service
rosservice call /adaptive_control_app/continuous_adaptive_control "{isEnabled: true, motorID: 1, param1: 1.2, param2: 1.0, intSpeed: 0.5, Kp: 2.0, Ki: 5.0, Kd: 1.0}"
```
To run predictive_maintenance_app.py and adaptive_control_app.py at the same time:
```
# -l, -p, -s, -t, -r, -m remain the same usage as described above
# -c option defines whether the applications are running based on the chariot cloud vm
# If ChariotCloudAPI.py is already running on chariot cloud vm
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/launch/
python pm_ac.py -c -l LINK -p POWER_TYPE -r -m MOTOR_ID -s -t

# If ChariotCloudAPI.py is already running on local
cd <YOUR_PATH>/dobot-conveyor-unit_v2/CHARIOT/src/dobot/launch/
python pm_ac.py -l LINK -p POWER_TYPE -r -m MOTOR_ID -s -t
rosservice call /adaptive_control_app/continuous_adaptive_control "{isEnabled: true, motorID: 1, param1: 1.2, param2: 1.0, intSpeed: 0.5, Kp: 2.0, Ki: 5.0, Kd: 1.0}"
```


## Others
```
```

## Documentation
```
```

## Historical Versions
1. Version 0.20, Updated 24 Mar 2019, Current Version
* Adding more details to the README.md file.
* Includes the newest polynomial regression based power calculation models for real motor, simulation motor with noises and simulation motor without noises.
2. Version 0.11, Updated 11 Dec 2018
* Specifying IP address and port through command line is now supported.
* Working with predictive_maintenance_app.py and motor_plot_2D on localhost and cloud tested.
3. Version 0.10, Updated 8 Dec 2018
* The first working cloud API. Working with predictive_maintenance_app.py and motor_plot_2D not tested.

## Existing Issues
1. The new ChariotCloudAPI.py and MachineLearningAPI.py might cause the malfunction of some files, for example, quality_prediction.py, etc.
