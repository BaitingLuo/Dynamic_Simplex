#!/usr/bin/python3
import csv
import pandas as pd
import random
import numpy as np

def get_weather(weather_file,num):
    """
    Get weather parameters
    """
    with open(weather_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data[num]

def Manual_Search(path,current_hyperparameters,folder,simulation_run,route_path,y,scene_num,initial_condition,weather_step,traffic_step,pedestrian_step,tracks_num):
    """
    The random sampler takes in the hyperparameters of the current step and returns a new hyperparameter set that is randomly sampled
    """
    weather_file = path + "/" + "random_weather" + "/" + "weather.csv"
    num = simulation_run * tracks_num + scene_num-1
    parameter_distribution = get_weather(weather_file,num)
    new_hyperparameters_list = []
    #choices_array = []
    distributions = []

    for i in range(len(current_hyperparameters)):
        # if current_hyperparameters[i][0] == 'weather':
        #     step = weather_step
        # elif current_hyperparameters[i][0] == 'traffic':
        #     step = traffic_step
        # elif current_hyperparameters[i][0] == 'pedestrian':
        #     step = pedestrian_step
        # else:
        #     step = 5
        # choice_list = np.arange(current_hyperparameters[i][1],current_hyperparameters[i][2],step)
        # choices_array.append(choice_list)
        # parameter_distribution = random.choice(choice_list)
        distributions.append(int(parameter_distribution[i]))
        new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution[i])))

    print(distributions)

    return distributions, new_hyperparameters_list
