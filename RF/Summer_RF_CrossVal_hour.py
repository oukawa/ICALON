import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

answer = input("Run for day or night? (d/n): ")

loops = int(input("How many loops? (integers only): ")) 

cores = 8

if answer == "d": 
    # building RF model for daytime
    R2_list = []
    RMSE_list = []

    for i in range(loops):
        # load dataset
        df = pd.read_csv('summer_X_all_daytime_hour_LCZ.csv')
        df = df.drop(columns=["datenum", "year", "month", "day", "hour", "minute", "second", 
            "Talt", "RH", "RH_HOBO", "TPLT", "LOC"])
        df.dropna(inplace=True)
        df.head()

        X = df.iloc[:,0:df.shape[1]-1]
        Y = df.iloc[:,(df.shape[1]-1)]

        # split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # build model
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, oob_score = True, min_samples_leaf = 5, verbose=1, n_jobs=cores)
        rf.fit(X_train, Y_train)

        # predictions
        Ypred = rf.predict(X_test)
        #Yoob = rf.oob_prediction_

        errors = abs(Ypred - Y_test)
        #errorsoob = abs(Yoob - Y_test)

        # metrics
        R2 = r2_score(Y_test, Ypred)
        RMSE = np.sqrt(mean_squared_error(Y_test, Ypred))

        print('RMSE:', round(RMSE,3))
        print('R2:', round(R2,3))

        R2_list.append(R2)
        RMSE_list.append(RMSE)

        print('Completed loop:', i+1)

    valid = {'R2': R2_list, 'RMSE': RMSE_list}
    valid = pd.DataFrame(valid)
    valid.head()
    valid.to_csv('summer_RF_daytime_metrics.csv', index=False)

elif answer == "n": 
    # building RF model for nighttime
    R2_list = []
    RMSE_list = []
    
    for i in range(loops):
        # load dataset
        df = pd.read_csv('summer_X_all_nighttime_hour_LCZ.csv')
        df = df.drop(columns=["datenum", "year", "month", "day", "hour", "minute", "second", 
            "Talt", "RH", "RH_HOBO", "TPLT", "LOC"])
        df.dropna(inplace=True)
        df.head()

        X = df.iloc[:,0:df.shape[1]-1]
        Y = df.iloc[:,(df.shape[1]-1)]

        # split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # build model
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, oob_score = True, min_samples_leaf = 5, verbose=1, n_jobs=cores)
        rf.fit(X_train, Y_train)

        # predictions
        Ypred = rf.predict(X_test)
        #Yoob = rf.oob_prediction_

        errors = abs(Ypred - Y_test)
        #errorsoob = abs(Yoob - Y_test)

        # metrics
        R2 = r2_score(Y_test, Ypred)
        RMSE = np.sqrt(mean_squared_error(Y_test, Ypred))

        print('RMSE:', round(RMSE,3))
        print('R2:', round(R2,3))

        R2_list.append(R2)
        RMSE_list.append(RMSE)

        print('Completed loop:', i+1)
        
    valid = {'R2': R2_list, 'RMSE': RMSE_list}
    valid = pd.DataFrame(valid)
    valid.head()
    valid.to_csv('summer_RF_nighttime_metrics.csv', index=False)

else: 
    print("Invalid option") 