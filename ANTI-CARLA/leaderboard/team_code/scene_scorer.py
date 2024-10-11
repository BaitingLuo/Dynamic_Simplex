#!/bin/python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import csv
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

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
                x.append(float(row[8]))
                x.append(float(row[9]))
                X.append(x)
    X = np.array(X)
    return X

def search(entry,X_train):
    match = []
    weather = []
    for val in X_train:
        if entry[0] == val[0] and entry[1] == val[1] and entry[2] == val[2] and entry[3] == val[3]:
            match.append(val)
            weather.append([val[4],val[5],val[6],val[7]])


    return match,weather

def get_driving_score(x,X_subset,X_weather,knn):
    lbc_score = []
    ap_score = []
    curr_weather = [x[4],x[5],x[6],x[7]]
    for val in X_subset:
        if (x[4] == val[4]) and (x[5] == val[5]) and (x[6] == val[6]) and (x[7] == val[7]):
            lbc_score.append(float(val[8]))
            ap_score.append(float(val[9]))
        else:
            knn.fit(X_weather)
            curr = np.array(np.array(curr_weather).reshape(1,-1))
            neighbors = knn.kneighbors(curr, return_distance=False)
            for val in neighbors[0]:
                lbc_score.append(float(X_subset[val][8]))
                ap_score.append(float(X_subset[val][9]))

    return round(sum(lbc_score)/len(lbc_score),2), round(sum(ap_score)/len(ap_score),2)


def predict(X_train,X_test,knn):
    Y_pred = []
    for x in X_test:
        X_subset, X_weather = search(x,X_train)
        lbc_score, ap_score = get_driving_score(x,X_subset,X_weather,knn)
        print(lbc_score,ap_score)

    return lbc_score, ap_score



if __name__ == '__main__':
    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=3, p=2,radius=1.0)
    train_dataset = "encoded_prior_belief.csv"
    test_dataset = "encoded_prior_belief_test.csv"
    path = "/isis/Carla/AV-Adaptive-Mitigation/controller_data/"
    test_data_path = path + "prior_data" + "/" + test_dataset
    train_data_path = path + "prior_data" + "/" + train_dataset
    X_train = read_data(train_data_path)
    X_test = read_data(test_data_path)
    lbc_score, ap_score = predict(X_train,X_test,knn)
