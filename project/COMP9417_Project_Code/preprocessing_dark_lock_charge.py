# Date:11/13/2019
# Author:Group work
#
# This file is to extract information from the input files: dark, phonecharge, phonelock
# It will caculate the average duration of dark, phonecharge, phonelock per day of each student and save them 
# to a csv file and plot them.

# read dataset
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def get_df_by_id(id, cur_input_dir):
    # input string id
    # output dataframe list
    subject = 'u' + id
    basedir = './Inputs/sensing/' + cur_input_dir + '/'
    with os.scandir(basedir) as entries:
        for entry in entries:
            if subject in entry.name:
                # print(basedir + entry.name)
                rawdf = pd.read_csv(basedir + entry.name, header = 0)

                out_df  = pd.DataFrame()
                # df.iloc[:, index] = pd.to_datetime(df.iloc[:, index],unit='s')
                out_df['date'] = pd.to_datetime(rawdf.iloc[:, 0],unit='s').dt.date # only show date
                out_df['duration'] = rawdf.iloc[:, 1] - rawdf.iloc[:, 0]
                out_df['duration'] = out_df['duration'].apply(lambda s: s/3600) # convert to hours
                # print(out_df.head())
                pd_sum = out_df.groupby(['date']).agg(day_duration = pd.NamedAgg(column = 'duration', aggfunc = sum))

                return pd_sum



def normalize(series):
    series_n = (series - series.min()) / (series.max() - series.min())
    return series_n

def get_data(workdirs = ['dark', 'phonecharge', 'phonelock']):
    # collect  data
    result_dict = dict()
    for cur_dir in workdirs:
        mean_dict = dict()
        for id in range(60):
            str_id = f'{id:02}'
            df = get_df_by_id(str_id, cur_dir)
            # print(df.head())
            # print(df.mean())
            if df is not None:
                mean_dict['u' + str_id] = df.mean().iloc[0]
        s = pd.Series(mean_dict, name = cur_dir + '_mean_duration')
        result_dict[cur_dir] = s
        # print(s.head()) #debug
    return result_dict

def get_series(name):
    series = get_data([name])[name]
    return series

if __name__  == '__main__':

    result_dict = get_data()
    result_dict['dark'].to_csv('./Pre/mean_dark.csv', header = 'mean dark duration per day', index_label = 'uid')
    result_dict['phonecharge'].to_csv('./Pre/mean_phonecharge.csv', header = 'mean phonecharge duration per day', index_label = 'uid')
    result_dict['phonelock'].to_csv('./Pre/mean_phonelock.csv', header = 'mean phonecharge duration per day', index_label = 'uid')

    # plot
    x = np.arange(len(result_dict['dark'].to_numpy()))
    y1 = result_dict['dark'].to_numpy()
    y2 = result_dict['phonecharge'].to_numpy()
    y3 = result_dict['phonelock'].to_numpy()
    h1 = plt.bar(x, y1, color='cornflowerblue', label='dark')
    h2 = plt.bar(x, y2, bottom=y1, color='lime', label='phonecharge')
    h3 = plt.bar(x, y3, bottom=y1+y2, color='darkorange', label='phonelock')
    plt.legend(handles = [h1,h2,h3])
    plt.title('average hours of each activity for every student ')
    plt.show()

    # plot normalize
    x = np.arange(len(result_dict['dark'].to_numpy()))
    y1 = normalize(result_dict['dark']).to_numpy()
    y2 = normalize(result_dict['phonecharge']).to_numpy()
    y3 = normalize(result_dict['phonelock']).to_numpy()
    h1 = plt.bar(x, y1, color='cornflowerblue', label='dark')
    h2 = plt.bar(x, y2, bottom=y1, color='lime', label='phonecharge')
    h3 = plt.bar(x, y3, bottom=y1+y2, color='darkorange', label='phonelock')
    plt.legend(handles = [h1,h2,h3])
    plt.title('average hours(normalized) of each activity for every student ')
    plt.show()

    
