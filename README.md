Machine Learning Engine for IoT
============
Version 1.0, uploaded on 16.04.2020 as an open source project
- Orhan Can Görür (orhan-can.goeruer@dai-labor.de)
- Shreyas Gokhale (shreyas6gokhale@gmail.com)
- Xin Yu (marvelous.islander@gmail.com)
- Fikret Sivrikaya (fikret.sivrikaya@gt-arc.com)

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
- swagger: under `swagger`. Note that it uses `mlcloud.json` for the documentation. It is also possible to deploy swagger separately: run `deploy_swaggerOnly.sh`. Swagger interface can be reached from http://localhost:4400/swagger/
- The ChariotCloudAPI runs by default on localhost:5000. If you would like to change the host, `docker-compose.yaml` file line 6 has the port selection, and Dockerfile line 7 `ChariotCloudAPI` has the link parameter.

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
- First, you need to initialize and run a mongo DB on your PC. An example credentials would be: `username: mldb`, `pass: chariot`, `authentication database: admin`, `port: 27017`.

- In order to run the API, only ChariotCloudAPI.py is needed. To run:
```
# specify the hostname for MongoDB access using -H option     
# specify the link for RESTFul http access using -l option
# by default (without any options), api runs on localhost
# an example given that Mongo DB is initialized as above:
python3 ChariotCloudAPI.py -port 27017 -u mldb -p chariot -db admin -l 0.0.0.0:5000
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
    "deviceID" : "123456",              # Device ID. Any string. Make sure to comply with the same ID used in Database operations below
    "algorithm" : "OCSVM",              # Algorithm you want to use to train the data. Alternatively: KNN, KPCA
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
    "deviceID" : "123456",              # Device ID. Any string. Make sure to comply with the same ID used in Database operations below
    "algorithm" : "OCSVM",              # Algorithm you want to use to train the data.  Alternatively: KNN, KPCA
    "para" :[0.01,0.5],                 # Parameters used for training model that want to be used.
    "properties" : ["velocity","power_in"],           # The variables you already trained the data for.
    "value": [ 31.00 , 5]               # the data point for the variables given above.
*   "idname": "deviceID"                # What do you call your ID. If not provided, it is "deviceID" by default.
*   "resulttable" : "results"   # Table Name to save results. If = "nosave", don't save - just return the results to the user. Or it saves to a table by default
*   "database": "predictive_maintenance"          # If you want to save the results in the database.
}

Returns status and the prediction result (true or false) on the given value. Prediction result:
{
      "anomaly_detected" : True / False
      "ml_result"        : 1.0 / -1.0 (not needed as it holds the same info as anomaly_detected)
}

```

#### ML Database APIs
A Postman collection is also provided for an easy call interface for the services: `mongoDB_postman_collection.json`

* CopyCSVToMLDB

```
'/cloud/db/CopyCSVToMLDB/'

{
    "deviceID": "123456",
    "path" : "<path_to_CSV_file_holding_source_data>",  # Make sure the first row of CSV file holds the variable names. An example file provided. /app/datafiles/real_motor_data_0320.csv
    "properties" : ["<variable-1>", "<variable-2>"],    # Make sure the variable names are the ones in CSV file. Use "velocity" and "power_in" for the example above.
  "database": "predictive_maintenance"    # Any database name put will be created!
}

Returns process status
```

* GetTableData

```
'/cloud/db/getTableData/'
{
    "table": "raw_velocitypower_in_123456",             # Name of the table where you want the data to get.
*   "database": "predictive_maintenance"          # If you want to get from a specific database. It is "predictive_maintenance" by default.

}

Returns Table in JSON format.
```

* GetLatestData

```
'/cloud/db/getLatestData/'
{
    "table": "raw_velocitypower_in_123456",             # Name of the table where you want the data to get.
    "idname"  : deviceID                          # ID, if used in the name of the table (deviceID in our case)
    "deviceID" : 123456,                           # Device ID. This is NOT a serial number of the data.
*   "database": "predictive_maintenance"          # If you want to get from a specific database. It is "ChariotCloud" by default.
}

Returns latest entry of the table in JSON format.
```

* CopyCSVInTable

```
'/cloud/db/copyCSVInTable/'
{
    "table": "raw_velocitypower_in_123456",              # Name of the table where you want the data to be stored.
    "path" : "./datafiles/real_motor_data_0320.csv", # CSV file path
    # NOTE: You either have to define header names in the csv, or give column_names in the request.
    "col" : ["deviceID" , "speed", "velocity",  "torque"],
*   "database": "predictive_maintenance"           # If you want to get from a specific database. It is "predictive_maintenance" by default.
}

Returns result.
```

* WriteOneRowInTable

```
'/cloud/db/writeOneRowInTable/'
{
    "table": "raw_velocitypower_in_123456",              # Name of the table where you want the data to be stored.
*   "database": "predictive_maintenance"           # If you want to get from a specific database. It is "predictive_maintenance" by default.
    "data":                              # Single JSON record of enrty to be added
            {
                "velocity" : 34.2325,
                "power_in" : 1.1515
            }
}
Returns result.
```


* UpdateOneInTable

```
'/cloud/db/updateOneInTable/'
{
    "table": "raw_velocitypower_in_123456",              # Name of the table where you want the data to be updated.
*   "database": "predictive_maintenance"           # If you want to update from a specific database. It is "ChariotCloud" by default.
    "key":                               # You want to update entry having this values. If multiple key exist, first matching key will be updated.
            {
                "properties": "velocity"
            },
    "value":                             # Whatever part of the record you want to change
            {
                "velocity" : 60
            }    
}
Returns result.
```

* Delete Database

```
'/cloud/db/deleteDatabase/'
{
   "database": "predictive_maintenance"           # Name of the database to be deleted.
}
Returns result.
```

* Delete Table

```
'/cloud/db/deleteTable/'
{
   "table": "raw_velocitypower_in_123456"               # Name of the table to be deleted.
*  "database": "predictive_maintenance"           # If you want to delete from a specific database. It is "predictive_maintenance" by default.

}
Returns result.
```

## Historical Versions
1. Version 1.0, Updated 16.04.2020, Current version
* The project was made open source
* CHARIOT project interfaces were removed and redirected to ML DB local interface
* Documentation is updated accordingly
1. Version 0.20, Updated 24 Mar 2019
* Adding more details to the README.md file.
* Includes the newest polynomial regression based power calculation models for real motor, simulation motor with noises and simulation motor without noises.
2. Version 0.11, Updated 11 Dec 2018
* Specifying IP address and port through command line is now supported.
* Working with predictive_maintenance_app.py and motor_plot_2D on localhost and cloud tested.
3. Version 0.10, Updated 8 Dec 2018
* The first working cloud API. Working with predictive_maintenance_app.py and motor_plot_2D not tested.

## Existing Issues
1. The new ChariotCloudAPI.py and MachineLearningAPI.py might cause the malfunction of some files, for example, quality_prediction.py, etc.

## References
In any use of this code, please let the author know and please cite the article below:
- Görür, O.C.; Yu, X.; Sivrikaya, F. Integrating Predictive Maintenance in Adaptive Process Scheduling for a Safe and Efficient Industrial Process. Appl. Sci. 2021, 11, 5042. https://doi.org/10.3390/app11115042


