import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

answer = input("Run for day or night? (d/n): ")

variables = ["date", "T_station", "RH_HOBO", "LOC"]

if answer == "d":
# building model for daytime
    print("Selected: daytime")

    answer = input("Run with UHII as resp. variable? (y/n): ")

    # load dataset
    if answer == "y":
        df = pd.read_csv('summer_X_all_daytime_hour_LCZ_UHII.csv')
        df.dropna(inplace=True)
        print(df.head())

        shap_fileName = 'summer_RF_day_SHAP_values_LCZ_UHII.csv'
        shap_corr_fileName = 'summer_RF_day_SHAP_corr_LCZ_UHII.csv'
        shap_swarmPlot = 'summer_summary_plot_day_LCZ_UHII.png'
        metrics_fileName = 'summer_RF_day_metrics_LCZ_UHII.csv'

    elif answer == "n":
        df = pd.read_csv('summer_X_all_daytime_hour_LCZ.csv')
        df.dropna(inplace=True)
        print(df.head())

        shap_fileName = 'summer_RF_day_SHAP_values_LCZ.csv'
        shap_corr_fileName = 'summer_RF_day_SHAP_corr_LCZ.csv'
        shap_swarmPlot = 'summer_summary_plot_day_LCZ.png'
        metrics_fileName = 'summer_RF_day_metrics_LCZ.csv'

    else:
        print("Invalid option")

elif answer == "n":
# building model for nighttime
    print("Selected: nighttime")

    answer = input("Run with UHII as resp. variable? (y/n): ")

    # load dataset
    if answer == "y":
        df = pd.read_csv('summer_X_all_nighttime_hour_LCZ_UHII.csv')
        df.dropna(inplace=True)
        print(df.head())

        shap_fileName = 'summer_RF_night_SHAP_values_LCZ_UHII.csv'
        shap_corr_fileName = 'summer_RF_night_SHAP_corr_LCZ_UHII.csv'
        shap_swarmPlot = 'summer_summary_plot_night_LCZ_UHII.png'
        metrics_fileName = 'summer_RF_night_metrics_LCZ_UHII.csv'

    elif answer == "n":
        df = pd.read_csv('summer_X_all_nighttime_hour_LCZ.csv')
        df.dropna(inplace=True)
        print(df.head())

        shap_fileName = 'summer_RF_night_SHAP_values_LCZ.csv'
        shap_corr_fileName = 'summer_RF_night_SHAP_corr_LCZ.csv'
        shap_swarmPlot = 'summer_summary_plot_night_LCZ.png'
        metrics_fileName = 'summer_RF_night_metrics_LCZ.csv'

    else:
        print("Invalid option")

else:
    print("Invalid option")

df = df.drop(columns=variables)

# train/test using standardized data (mean=0, sd=1)
scaler = preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)

X = df_scaled.iloc[:,0:df_scaled.shape[1]-1]
Y = df_scaled.iloc[:,(df_scaled.shape[1]-1)]

# train/test using unstandardized data
X = df.iloc[:,0:df.shape[1]-1]
Y = df.iloc[:,(df.shape[1]-1)]

# train/test model within a loop
range_int = int(input("Set number of loops (Integers only): "))

R2 = []
RMSE = []
MAE = []

for i in range(range_int):

     # train/test split (80/20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # build the model
    mdl = RandomForestRegressor(n_estimators = 1000, random_state = 42, oob_score = True, 
                            min_samples_leaf = 1, min_samples_split = 3, bootstrap = True,
                            n_jobs=8, verbose=1)
    
    # train
    mdl.fit(X_train, Y_train)

    # predict
    Y_pred = mdl.predict(X_test)
    Y_oob = mdl.oob_prediction_

    # calculate metrics
    RMSE_i = np.sqrt(mean_squared_error(Y_pred, Y_test))
    RMSE.append(RMSE_i)

    MAE_i = np.mean(np.abs(Y_pred - Y_test))
    MAE.append(MAE_i)

    R2_i = r2_score(Y_test, Y_pred)
    R2_i_OOB= r2_score(Y_train, Y_oob)
    R2.append(R2_i)

    # print metrics
    print('RMSE:', np.round(RMSE,4))
    print('MAE:', np.round(MAE,4))
    print('R2:', np.round(R2,4))
    print('R2 (OOB):', np.round(R2_i_OOB,4))

    # Y_df = pd.DataFrame(Y_oob, columns=['T_predicted'])
    # Y_df = pd.concat([Y, Y_df], axis=1)

# store metrics in a .csv file
valid = {'R2': R2, 'RMSE': RMSE, 'MAE': MAE}
valid = pd.DataFrame(valid)
valid.head()
valid.to_csv(metrics_fileName, index=False)

# SHAP feature importance
answer = input("Include SHAP values? (y/n): ")

if answer == "y":

    shap_values = shap.TreeExplainer(mdl).shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

    f = plt.figure()
    shap.summary_plot(shap_values, X)
    f.savefig(shap_swarmPlot, bbox_inches='tight', dpi=600)

    df_shap_values = pd.DataFrame(shap_values, index=X.index, columns=X.columns)
    df_shap_values.to_csv(shap_fileName)

    print(Y)

    shap_copy = pd.DataFrame(shap_values)
    feature_list = X.columns
    shap_copy.columns = feature_list
    df_copy = X.copy().reset_index().drop('index',axis=1)

    corr_list = list()
    for i in feature_list:
            b = np.corrcoef(shap_copy[i],df_copy[i])[1][0]
            corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    
    # make a dataframe -- culumn 1 is the feature and column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')

    # save values to .csv
    corr_df.to_csv(shap_corr_fileName)

    print(corr_df)

elif answer == "n":
    print("Done!")

else:
    print("Invalid option")