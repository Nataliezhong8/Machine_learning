"""
Date:11/13/2019
Author:Group work

Group project:
output data clean
1. read the dataset
2. calculate the sum score of each student
3. collect the data in a csv file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')



flourishing = pd.read_csv('.\Outputs\FlourishingScale.csv')
#shuffle the data, 
#for missing data, if the uid has pre and post section of dataset, fill the missing data with the existing data of that feature(axis = 1)
#if the uid only has one section of dataset(pre or post),fill the missing data with the average value of the whole dataset of this uid(axis = 0)

fl_table = pd.pivot_table(flourishing,index = 'uid', aggfunc = np.mean).sum(axis = 1) #sum the row values
fl_table.to_csv('.\pre\flourishing_sum.csv',header = 'flourishing_sum', index_label = 'uid')
fl_table.plot()

pan_pos = pd.read_csv('.\Outputs\panas_pos.csv')
postive = pd.pivot_table(pan_pos,index = 'uid').sum(axis = 1)
postive.plot()

pan_neg = pd.read_csv('.\Outputs\panas_nega.csv')
negative = pd.pivot_table(pan_neg,index = 'uid').sum(axis = 1)
negative.plot()

