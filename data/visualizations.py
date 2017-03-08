import csv
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame, Panel
from scipy.interpolate import spline


def list_str_to_int(input):
    str_hold = "".join(input)
    return int(str_hold)

def time_series_avg_wind_speeds(curr_windset):
    """ avg wind speed (per day) forecast plots for a single wind farm
    """
    time_series = []
    wind_speeds = []
    prev = None
    curr_date_speeds = []
    for row in curr_windset:

        if row[0] != 'date':
            date        = row[0]
            wind_speed  = row[5]

            date_arr = list(date)

            year    = list_str_to_int(date_arr[0:4])
            month   = list_str_to_int(date_arr[4:6])

            time_series_entry = datetime(year, month, 1)
            if wind_speed != 'NA':
                if (time_series_entry != prev) and (prev != None):
                    avg_wind_speed = np.mean(curr_date_speeds)

                    wind_speeds.append(avg_wind_speed)
                    time_series.append(time_series_entry)
                    curr_date_speeds = []

                else:
                    curr_date_speeds.append(float(wind_speed))
                    # print curr_date_speeds
            prev = time_series_entry
    plt.plot(time_series, wind_speeds)
    plt.savefig('plots/'+str(sys.argv[1] + '_avg.pdf'))
    plt.show()

if __name__ == '__main__':
    curr_windset = csv.reader(open(sys.argv[1], 'r'))
    time_series_avg_wind_speeds(curr_windset)
