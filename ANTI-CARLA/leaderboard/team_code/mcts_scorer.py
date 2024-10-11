#!/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import random

def create_bin(val):
    bin = -1
    if 0 <= val < 25:
        bin = 0
    elif 25 <= val < 50:
        bin = 1
    elif 50 <= val < 75:
        bin = 2
    else:
        bin = 3

    return bin


def read_data(data_path):
    X = []
    Y = []
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
                x = []
                y = []
                x.append(row[0])
                x.append(row[1])
                x.append(row[2])
                x.append(row[3])
                x.append(create_bin(int(row[4])))
                x.append(create_bin(int(row[5])))
                x.append(create_bin(int(row[6])))
                x.append(create_bin(int(row[7])))
                x.append(row[8])
                x.append(row[9])
                x.append(row[10])
                x.append(float(row[11]))
                x.append(float(row[12]))
                x.append(float(row[13]))
                x.append(float(row[14]))
                X.append(x)
    X = np.array(X)
    return X

def search_scene(entry,X_train):
    match = []
    weather = []

    for val in X_train:
        if (float(entry[0]) == val[0]) and (float(entry[1]) == val[1]) and (float(entry[2]) == val[2]) and (float(entry[3]) == val[3]) and (float(entry[8]) == val[8])  and (float(entry[9]) == val[9])  and (float(entry[10]) == val[10]):
            match.append(val)
            weather.append([val[4],val[5],val[6],val[7]])
    if len(match) == 0:
        for val in X_train:
            if (float(entry[0]) == val[0]) and (float(entry[1]) == val[1]) and (float(entry[2]) == val[2]) and (float(entry[3]) == val[3]):
                match.append(val)
                weather.append([val[4], val[5], val[6], val[7]])
    return match,weather

# def search_fault(entry,X_train):
#     match = []
#     weather = []
#     for val in X_train:
#         if entry[8] == val[8] or entry[9] == val[9] or entry[10] == val[10]:
#             match.append(val)
#             weather.append([val[4],val[5],val[6],val[7]])


    # return match,weather

def get_driving_score(x,X_subset,X_weather,knn,controller):
    perf_score = []
    safe_score = []
    curr_weather = [x[4],x[5],x[6],x[7]]
    for val in X_subset:
        if (x[4] == val[4]) and (x[5] == val[5]) and (x[6] == val[6]) and (x[7] == val[7]) and (float(x[8]) == val[8])  and (float(x[9]) == val[9])  and (float(x[10]) == val[10]):
            if controller == "LBC":
                perf_score.append(float(val[11]))
                safe_score.append(float(val[12]))
            else:
                perf_score.append(float(val[13]))
                safe_score.append(float(val[14]))
    if len(perf_score) == 0 and len(safe_score) == 0:
        knn.fit(X_weather)
        curr = np.array(np.array(curr_weather).reshape(1,-1))
        neighbors = knn.kneighbors(curr, return_distance=False)
        for val1 in neighbors[0]:
            if controller == "LBC":
                perf_score.append(float(X_subset[val1][11]))
                safe_score.append(float(X_subset[val1][12]))

            else:
                perf_score.append(float(X_subset[val1][13]))
                safe_score.append(float(X_subset[val1][14]))
    #print(perf_score)
    #print(controller,perf_score)
    return round(sum(perf_score)/len(perf_score),2), round(sum(safe_score)/len(safe_score),2)

def get_mean_velocity(x,X_subset,X_weather,knn,controller):
    perf_score = []
    safe_score = []
    curr_weather = [x[4],x[5],x[6],x[7]]
    for val in X_subset:
        if (x[4] == val[4]) and (x[5] == val[5]) and (x[6] == val[6]) and (x[7] == val[7]):
            if controller == "LBC":
                perf_score.append(float(val[11]))
                safe_score.append(float(val[12]))
            else:
                perf_score.append(float(val[13]))
                safe_score.append(float(val[14]))
    if len(perf_score) == 0 and len(safe_score) == 0:
        knn.fit(X_weather)
        curr = np.array(np.array(curr_weather).reshape(1, -1))
        neighbors = knn.kneighbors(curr, return_distance=False)
        for val1 in neighbors[0]:
            if controller == "LBC":
                perf_score.append(float(X_subset[val1][11]))
                safe_score.append(float(X_subset[val1][12]))

            else:
                perf_score.append(float(X_subset[val1][13]))
                safe_score.append(float(X_subset[val1][14]))
    #print(controller,perf_score)
    return round(sum(perf_score)/len(perf_score),2), round(sum(safe_score)/len(safe_score),2)

def predict(state,train):
    Y_pred = []
    state = np.array(list(state))
    state = state.astype(float)
    path = "/ANTI-CARLA/trained_monitors/prior_data"
    train_dataset = "mcts_lut.csv"
    velocity_dataset = "/ANTI-CARLA/trained_monitors/mcts_lut.csv"
    test_dataset = "mcts_lut_test.csv"
    train_data_path = path + "/" + train_dataset
    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,radius=1.0)
    #X_train = read_data(train_data_path)
    for cluster in train:
        if (float(cluster[0][0]) == state[0]) and (float(cluster[0][1]) == state[1]) and (float(cluster[0][2]) == state[2]) and (float(cluster[0][3]) == state[3]):
            X_train = np.array(cluster)
    #X_train = np.array(train)
    X_train = X_train.astype(float)
    if state[-1] == 0:
        controller = "AP"
    else:
        controller = "LBC"
    #for x in state[0:-2]:
    x = state[0:-1]
    X_subset, X_weather = search_scene(x,X_train)
    #print(len(X_subset),len(X_weather))
    perf_score, safe_score = get_driving_score(x,X_subset,X_weather,knn,controller)
    #print(perf_score, safe_score)

    return perf_score, safe_score

def vpredict(state,train):
    state = np.array(list(state))
    state = state.astype(float)

    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=3, p=2,radius=1.0)
    for cluster in train:
        if (float(cluster[0][0]) == state[0]) and (float(cluster[0][1]) == state[1]) and (float(cluster[0][2]) == state[2]) and (float(cluster[0][3]) == state[3]):
            X_train = np.array(cluster)
    X_train = X_train.astype(float)
    if state[-1] == 0:
        controller = "AP"
    else:
        controller = "LBC"
    x = state[0:-1]
    X_subset, X_weather = search_scene(x,X_train)
    v_lbc, s_lbc = get_mean_velocity(x,X_subset,X_weather,knn,controller)
    return v_lbc, s_lbc

if __name__ == '__main__':
    controller = ["LBC","AP"]

