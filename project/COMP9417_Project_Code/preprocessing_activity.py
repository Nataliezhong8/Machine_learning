'''
Date: 11/13/2019 
Author: Group work

This program mainly do the preprocessing of Activity data including cleaning the dataset, handling
missing values and calculate the average duration of running, walking, stationary. For this part, 
we just read the original data set and calculate the average time of each activity.

After we get all the average time of running, walking, stationary then we calculate the sleep time
according to stationary and phone dark duration and finally we output all the features as csv files.

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dark_lock_charge

def getAverageTime(pathName):
    """this function is used to get average time for every student activity per day"""

    dfAcitivity = pd.read_csv(pathName)
    # rename the columns for better useage
    dfAcitivity.columns = ['timestamp', 'activity_inference']
    totalTimeStationary, totalTimeWalking, totalTimeRunning, unknownTime = 0, 0, 0, 0
    # record every record and find the total time for three classes
    preValue = dfAcitivity['activity_inference'].iloc[0]
    preTimePoint = dfAcitivity['timestamp'].iloc[0]
    count = 0
    # calculation time duration of different activities
    for curvalue in dfAcitivity['activity_inference']:
        if curvalue != preValue:
            curTimePoint = dfAcitivity['timestamp'].iloc[count]
            timeInterval = curTimePoint - preTimePoint
            if preValue == 0:
                totalTimeStationary += timeInterval
            elif preValue == 1:
                totalTimeWalking += timeInterval
            elif preValue == 2:
                totalTimeRunning += timeInterval
            elif preValue == 3:
                unknownTime += timeInterval
            preTimePoint, preValue = curTimePoint, curvalue
        count += 1
    totalDay = (max(dfAcitivity['timestamp']) - min(dfAcitivity['timestamp'])) / (3600 * 24)
    # return average activity time per day
    return totalTimeStationary/totalDay, totalTimeWalking/totalDay, totalTimeRunning/totalDay, unknownTime/totalDay

def calculateSleepDuration(averageStationary, averageLight, averagePhoneLock, averagePhoneCharging):
    """ this function is used to calculate the sleep duration from 4 features with different coefficients
        the values of coefficients come from the reference
    """

    # this coefficients come from paper 
    alphaStationary, alphaLight, alphaPhoneLock, alphaPhongCharging = 0.5445, 0.0415, 0.0512, 0.0469
    dictForSleep = {
        'average_sleep_duration': [1, ],
    }

    averageSleepDurationDataFrame = pd.DataFrame(dictForSleep, index=averagePhoneLock.index)

    for uid in list(averageLight.index):
        averageSleepTime = alphaStationary*averageStationary.loc[uid] + alphaLight*averageLight.loc[uid] + \
            alphaPhoneLock*averagePhoneLock.loc[uid] + alphaPhongCharging*averagePhoneCharging.loc[uid]
        averageSleepDurationDataFrame['average_sleep_duration'].loc[uid] = averageSleepTime

    return averageSleepDurationDataFrame

def allStuAverageTime():
    """this function is used to calculate all stu average time for three class"""

    averageStationary, averageWalking, averageRunning, unknown = [], [], [], []
    totalStuNumber = 60
    studentLost = [6, 11, 21, 26, 28, 29, 37, 38, 40, 48, 55]
    for i in range(totalStuNumber):
        path, post = 'Inputs/sensing/activity/activity_u', '.csv'
        if i not in studentLost:
            if i < 10:
                path = path + '0' + str(i) + post
            elif i >= 10:
                path = path + str(i) + post
            averageStationaryTime, averageWalkingTime, averageRunningTime, unknownTime = getAverageTime(path)
            averageStationary.append(averageStationaryTime/3600)
            averageWalking.append(averageWalkingTime/3600)
            averageRunning.append(averageRunningTime/3600)
            unknown.append(unknownTime/3600)

    # build the index using uid of each student
    seriesIndex = []
    for i in range(totalStuNumber):
        if i not in studentLost:
            if i < 10:
                seriesIndex.append('u0'+str(i))
            else:
                seriesIndex.append('u'+str(i))

    # build the dataframe for all activities
    dictForActivity = {
        'average_stationary': averageStationary,
        'average_walking': averageWalking,
        'average_running':averageRunning,
        'unknown': unknown
    }

    # calculate the average sleep duraiton for each student
    dictForSleep = {
        'average_sleep_duration': [1,],
    }

    activityDataFrame = pd.DataFrame(dictForActivity, index=seriesIndex)
    averageSleepDurationDataFrame = pd.DataFrame(dictForSleep, index=seriesIndex)
    averageStationary = activityDataFrame['average_stationary']
    averageDarkTime = dark_lock_charge.get_series('dark')
    averagePhoneLock = dark_lock_charge.get_series('phonelock')
    averagePhoneCharge = dark_lock_charge.get_series('phonecharge')
    averageSleepDurationDataFrame = calculateSleepDuration(averageStationary, averageDarkTime, \
                                                  averagePhoneLock, averagePhoneCharge)
    activityDataFrame = activityDataFrame.merge(averageSleepDurationDataFrame, left_index=True, right_index=True)
    activityDataFrame['average_stationary'] = activityDataFrame['average_stationary'] - \
                                              activityDataFrame['average_sleep_duration']

    return activityDataFrame

def ouputCSV():
    """return series for further use"""

    activityDataFrame = allStuAverageTime()

    activityDataFrame['average_stationary'].to_csv('./Pre/stationary.csv', header='average_stationary', index_label='uid')
    activityDataFrame['average_walking'].to_csv('./Pre/walking.csv', header='average_walking', index_label='uid')
    activityDataFrame['average_running'].to_csv('./Pre/running.csv', header='average_running', index_label='uid')
    activityDataFrame['average_sleep_duration'].to_csv('./Pre/sleep_duration.csv', header='average_sleep_duration', index_label='uid')


def visiualization():
    """visiualize all three calsses for all students"""

    activityDataFrame = allStuAverageTime()
    # x axis means studentID [0-60]
    x = np.arange(len(activityDataFrame['average_stationary'].to_numpy()))
    y1 = activityDataFrame['average_stationary'].to_numpy()
    y2 = activityDataFrame['average_walking'].to_numpy()
    y3 = activityDataFrame['average_running'].to_numpy()
    y4 = activityDataFrame['unknown'].to_numpy()
    y5 = activityDataFrame['average_sleep_duration'].to_numpy()
    h1 = plt.bar(x, y1, color='cornflowerblue', label='stationary')
    h2 = plt.bar(x, y2, bottom=y1, color='lime', label='walking')
    h3 = plt.bar(x, y3, bottom=y1+y2, color='darkorange', label='running')
    h4 = plt.bar(x, y4, bottom=y1+y2+y3, color='black', label='unknown')
    h5 = plt.bar(x, y5, bottom=y1 + y2 + y3 + y4, color='purple', label='sleep')
    plt.legend(handles=[h1, h2, h3, h4, h5])
    plt.title('average hours of each activity for every student ')
    plt.show()

def main():
    visiualization()
    ouputCSV()

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    main()
