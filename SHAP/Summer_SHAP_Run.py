import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load

os.chdir("D:/OneDrive/Research/Project_ICALON/MATLAB/Matrices/Processed_Data")

answer = input("Run for day or night? (d/n): ")

if answer == "d":
# load model for daytime
    print("Selected: daytime")

    answer = input("Run with UHII as resp. variable? (y/n): ")

    # load dataset
    if answer == "y":

        mdl = load('summer_RF_UHII_daytime.joblib')

        X = pd.read_csv('summer_X_all_daytime_hour_LCZ_UHII.csv')

        shap_fileName = 'summer_RF_day_SHAP_values_LCZ_UHII.csv'
        shap_corr_fileName = 'summer_RF_day_SHAP_corr_LCZ_UHII.csv'
        shap_swarmPlot = 'summer_summary_plot_day_LCZ_UHII.png'

    elif answer == "n":

        mdl = load('summer_RF_AirTemp_daytime.joblib')

        X = pd.read_csv('summer_X_all_daytime_hour_LCZ.csv')

        shap_fileName = 'summer_RF_day_SHAP_values_LCZ.csv'
        shap_corr_fileName = 'summer_RF_day_SHAP_corr_LCZ.csv'
        shap_swarmPlot = 'summer_summary_plot_day_LCZ.png'

    else:
        print("Invalid option")

elif answer == "n":
# load model for nighttime
    print("Selected: nighttime")

    answer = input("Run with UHII as resp. variable? (y/n): ")

    # load dataset
    if answer == "y":

        mdl = load('summer_RF_UHII_nighttime.joblib')

        X = pd.read_csv('summer_X_all_nighttime_hour_LCZ_UHII.csv')

        shap_fileName = 'summer_RF_night_SHAP_values_LCZ_UHII.csv'
        shap_corr_fileName = 'summer_RF_night_SHAP_corr_LCZ_UHII.csv'
        shap_swarmPlot = 'summer_summary_plot_night_LCZ_UHII.png'

    elif answer == "n":

        mdl = load('summer_RF_AirTemp_nighttime.joblib')

        X = pd.read_csv('summer_X_all_nighttime_hour_LCZ.csv')

        shap_fileName = 'summer_RF_night_SHAP_values_LCZ.csv'
        shap_corr_fileName = 'summer_RF_night_SHAP_corr_LCZ.csv'
        shap_swarmPlot = 'summer_summary_plot_night_LCZ.png'

    else:
        print("Invalid option")

else:
    print("Invalid option")

shap_values = shap.TreeExplainer(mdl).shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")

f = plt.figure()
shap.summary_plot(shap_values, X)
f.savefig(shap_swarmPlot, bbox_inches='tight', dpi=600)

df_shap_values = pd.DataFrame(shap_values, index=X.index, columns=X.columns)
df_shap_values.to_csv(shap_fileName)

shap_copy = pd.DataFrame(shap_values)
feature_list = X.columns
shap_copy.columns = feature_list
df_copy = X.copy().reset_index().drop('index',axis=1)

corr_list = list()
for i in feature_list:
        b = np.corrcoef(shap_copy[i],df_copy[i])[1][0]
        corr_list.append(b)
corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)

corr_df.columns  = ['Variable','Corr']
corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')

corr_df.to_csv(shap_corr_fileName)

print(corr_df)