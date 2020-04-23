'''
Date:11/13/2019
Author:Group work

Before running this code,we have selected top 9 destinations and  
they are divided into three classes(study,sleep and other).

Then we iterate every student's csv file and calculate three types
of time gaps based on the classified destinations.

Finally,we collect these time gaps ,plot the stacked bar of different
time gaps and save the results in the csv format.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# this function will return the time statistics of specific student
def get_df_by_id(id, cur_input_dir):
    subject = 'u' + id
    basedir = './Inputs/sensing/' + cur_input_dir + '/'
    with os.scandir(basedir) as entries:
        for entry in entries:
            #deal with info of specified student
            if subject in entry.name:
                #these are top 9 classified destinations
                study = ["sudikoff","north-park","kemeny"]
                sleep = ["mclaughlin","ripley","butterfield"]
                other = ["north-park","hopkins","53_commons"]
                #read corresponding csv file
                rawdf = pd.read_csv(basedir + entry.name, header = 0, index_col=False)      
                stat = ['u' + id,0,0,0]
                timeGap = (int(rawdf["time"].iloc[len(rawdf)-1]) - int(rawdf["time"].iloc[0]))/86400
                rawdf1 = pd.DataFrame() 
                rawdf1["cur_time"] = rawdf["time"]
                rawdf1["pas_time"] = rawdf["time"].shift(1)
                rawdf1['location'] = rawdf['location']
                rawdf1.dropna(how='any',inplace=True)
                #calculate the time gap and add these together
                def count_Specific(row):
                    lists1 = row['location'].split("[")
                    if lists1[0] == "in":
                        lists2 = lists1[1].split("]")
                        if lists2[0] in study:
                            stat[1]+=(row["cur_time"] - row["pas_time"])
                        elif lists2[0] in sleep:
                            stat[2]+=(row["cur_time"] - row["pas_time"])
                        elif lists2[0] in other:
                            stat[3]+=(row["cur_time"] - row["pas_time"])
                    return 
                #transform the unit(sec->hour)
                rawdf1.apply(count_Specific, axis=1)
                stat[1] = (stat[1]/timeGap)/3600
                stat[2] = (stat[2]/timeGap)/3600
                stat[3] = (stat[3]/timeGap)/3600
                return stat

#this function will 
def get_data(workdirs = ['wifi_location']):
    result_dict = dict()
    for cur_dir in workdirs:
        mean_dict = dict()
        rows = []
        #iterate every student
        for id in range(60):
            str_id = f'{id:02}'
            row = get_df_by_id(str_id, cur_dir)
            if row is not None:
                rows.append(row)
        #create new dataframe based on the processed data
        df = pd.DataFrame(rows,columns = ["uid","study","dorm","other"])
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df1["uid"] = df["uid"]
        df1["study"] = df["study"]
        df2["uid"] = df["uid"]
        df2["dorm"] = df["dorm"]
        df3["uid"] = df["uid"]
        df3["other"] = df["other"]
        df1.set_index("uid",inplace =True)
        df2.set_index("uid",inplace =True)
        df3.set_index("uid",inplace =True)
        
        #plot the graph includes three time slots everyone spends per day
        # ~ df.set_index("uid",inplace =True)
        # ~ result_dict[cur_dir] = df
        # ~ x = np.arange(len(df["study"].to_numpy()))
        # ~ y1 = df["study"].to_numpy()
        # ~ y2 = df["dorm"].to_numpy()
        # ~ y3 = df["other"].to_numpy()
        # ~ h1 = plt.bar(x, y1, color='cornflowerblue', label="study")
        # ~ h2 = plt.bar(x, y2, bottom=y1, color='lime', label="dorm")
        # ~ h3 = plt.bar(x, y3, bottom=y1+y2, color='darkorange', label="other")
        # ~ plt.legend(handles = [h1,h2,h3])
        # ~ plt.title('average hours of staying in different places for every student ')
        # ~ plt.show()
        # print(s.head()) #debug
        
        #save separate csv files
        df1.to_csv("./Pre/loc_study_duration.csv", header = 'mean time in study area per day', index_label = 'uid')
        df2.to_csv("./Pre/loc_dorm_duration.csv", header = 'mean time in dorm area per day', index_label = 'uid')
        df3.to_csv("./Pre/loc_other_duration.csv", header = 'mean time in other area per day', index_label = 'uid')
              

if __name__ == "__main__":              
    get_data()                

