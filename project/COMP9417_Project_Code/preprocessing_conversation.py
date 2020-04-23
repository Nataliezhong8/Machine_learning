'''
Date:11/13/2019
Author:Group work

Group project:
communication preprocessing
1. get the data from csv file and transform it to pd.dataframe
2. calculate the total conversation duration and conversation count grouped by date for each person
3. calculate the average conversation duration and conversation count for each person
4. collect the data in 'conversation_mean_duration.csv' file and 'conversation_mean_count.csv' file with the index 'uid'
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_df_by_id(id, cur_input_dir):
    # input string id
    # output dataframe list
    subject = 'u' + id
    basedir = './Inputs/sensing/' + cur_input_dir + '/'
    
    #get an iterator of os.DirEntry objects corresponding to the entries in the diretories
    with os.scandir(basedir) as entries:
        for entry in entries:
            if subject in entry.name:
                print(basedir + entry.name)
                rawdf = pd.read_csv(basedir + entry.name, header = 0) #header=0, indicates that the first line is the column name

                out_df  = pd.DataFrame()
               
                out_df['date'] = pd.to_datetime(rawdf.iloc[:, 0],unit='s').dt.date # only show date
                out_df['duration'] = rawdf.iloc[:, 1] - rawdf.iloc[:, 0]
                out_df['duration'] = out_df['duration'].apply(lambda s: s/3600) # convert to hours
               
                pd_sum = out_df.groupby(['date']).sum()
                pd_count = out_df.groupby(['date']).count()
              
                return pd_sum,pd_count


#get data, calculate the mean value for each studnet and then collect data in a csv file
def get_data(cur_dir):
    
    mean_dict = dict()
    count_dict = dict()
    for id in range(60):
        str_id = f'{id:02}'
        df = get_df_by_id(str_id, cur_dir)
        if df is not None:
            mean_dict['u' + str_id] = df[0].mean().iloc[0]
            count_dict['u' + str_id] = df[1].mean().iloc[0]
    
    s1 = pd.Series(mean_dict, name = 'mean_duration')
    s2 = pd.Series(count_dict, name = 'mean_count')
    s1.to_csv('.\pre\conversation_mean_duration.csv',header = 'mean_duration', index_label = 'uid')
    s2.to_csv('.\pre\conversation_mean_count.csv',header = 'mean_count', index_label = 'uid')
    #print(s2)
    return [s1, s2]

if __name__ == '__main__':
    conversation = get_data('conversation')
        
    #conversation_mean_duratio
    x = np.arange(len(conversation[0].to_numpy()))
    
    #plot the data 
    y1 = conversation[0].to_numpy()
    h1 = plt.bar(x, y1, color='cornflowerblue', label='conversation_duration')
    plt.legend(handles = [h1])
    plt.title('average hours of conversation for every student ')
    plt.show()
    
    y2 = conversation[1].to_numpy()
    h2 = plt.bar(x, y2, color='cornflowerblue', label='conversation_count')
    plt.legend(handles = [h2])
    plt.title('average times of conversation for every student ')
    plt.show()
    
