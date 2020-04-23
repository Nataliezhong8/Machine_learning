'''
Date:11/13/2019
Author:Group work

This code will iterate every student and calculate the encountered
bluetooth devices per day.

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rows = []
file_format = "./Inputs/sensing/bluetooth/bt_u"
for i in range(0,60):
    try:
        filename = file_format+f'{i:02}'+".csv"
        df = pd.read_csv(filename, index_col=False)
        timeGap = (int(df["time"].iloc[len(df)-1]) - int(df["time"].iloc[0]))/86400
        devicesP = float(len(df)/timeGap)
        rows.append([f'u{i:02}',devicesP])
    except FileNotFoundError:
        pass

out_df = pd.DataFrame(rows,columns = ["uid","devices per day"])

#plot the encountered bluetooth devices graph
# ~ x = out_df["id"]
# ~ y = out_df["devices per day"].to_numpy()
# ~ h = plt.bar(x, y)
# ~ plt.xlabel('ID')
# ~ plt.title('average devices per day for every student ')
# ~ plt.show()

out_df = out_df.set_index("uid")
#save the csv file
out_df.to_csv("./Pre/bluetooth_count.csv", header = 'mean number of nearby bluetooth per day', index_label = 'uid')


