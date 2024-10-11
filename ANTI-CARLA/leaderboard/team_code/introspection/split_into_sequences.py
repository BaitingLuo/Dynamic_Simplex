from dataclasses import dataclass
import numpy as np
import csv
import shutil
import random
import os, os.path
from os import listdir
from os.path import isfile, join
from glob import glob

def store_sequences(data_success,data_failure,points_per_seq,target_path):
    #Split data into sequences of given duration
    #Get number of sequences from length of data
    n_success = len(data_success)
    n_failure = len(data_failure)
    n_seq_success = np.floor((n_success)/points_per_seq).astype('int')
    n_seq_failure = np.floor((n_failure)/points_per_seq).astype('int')

    #Ensure target directory exists
    if os.path.isdir(target_path) == False: 
        os.mkdir(target_path)

    #Iterate through data and combine into one csv file, each row representing one state
    data_all = []
    labels_all = []

    if n_success > 0:
        for i in range(n_seq_success):
            seq_i = data_success[points_per_seq*i:points_per_seq*(i+1)]
            if len(seq_i) < points_per_seq:
                raise Exception('Sequence too short!')
            if len(seq_i) > points_per_seq:
                raise Exception('Sequence too long!')            

            #Now, go through sequence and append to data vector
            for s in seq_i:         
                data_all.append(s)
                label_i = 0 # Successes are zeros
                labels_all.append(label_i)

    if n_failure > 0:
        for i in range(n_seq_failure):
            seq_i = data_failure[points_per_seq*i:points_per_seq*(i+1)]
            if len(seq_i) < points_per_seq:
                raise Exception('Sequence too short!')
            if len(seq_i) > points_per_seq:
                raise Exception('Sequence too long!')            

            #Now, go through sequence and append to data vector
            for s in seq_i:              
                data_all.append(s)
                label_i = 1 # Failures are ones
                labels_all.append(label_i)

    if len(data_all)>0:
        #Store data
        f = open(target_path + 'data.csv','a')
        np.savetxt(f, data_all,fmt="%s", delimiter=',')
        #f.write("\n")
        f.close()

        #Store labels
        f = open(target_path + 'labels.csv','a')
        np.savetxt(f, labels_all,fmt="%i", delimiter=',')
        #f.write("\n")
        f.close()

def merge_throttle_brake(state_data):
    state_data_merged = []
    collision_vector = []
    for row in state_data:
        brake = row[5]
        collision = row[-1]
        if brake == 'True':
            acceleration = -1
        else:
            acceleration = row[4]
        row_new = [row[2], row[3], acceleration]
        collision_vector.append(collision)
        state_data_merged.append(row_new)

    return state_data_merged, collision_vector

def find_collision_time(collision_vector):
    time_collision = -1
    for i, row in enumerate(collision_vector):
        if int(row) == 1:
            time_collision = i
            break
    
    return time_collision

def split_into_success_and_failure(state_data,time_collision,points_per_seq):
    #If no collision is present, return right away
    if time_collision == -1:
        return state_data, []
    #Else. take all data before collision minus points_per_seq buffer
    else:
        state_data = state_data[0:time_collision]
        state_data_success = state_data[0:-points_per_seq]
        state_data_failure = state_data[-points_per_seq-1:-1]
        return state_data_success, state_data_failure  

def read_data_from_file(data_path):
    with open(data_path, newline='') as csvfile:
            csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
            state_data = list(csv_data)
    return state_data

def extract_sequences(data_path,target_path,points_per_seq):
    #Read in data from given sequence stored at data_path
    state_data = read_data_from_file(data_path)

    #Discard first row of semantic information
    state_data = state_data[1:]

    #Merge throttle and Boolean brake values into one state and separate collision information
    state_data_merged, collision_vector = merge_throttle_brake(state_data)

    #Look for time of collision
    time_collision = find_collision_time(collision_vector)

    #Get success and failure data from merged state data
    state_data_success, state_data_failure = split_into_success_and_failure(state_data_merged,time_collision,points_per_seq) 

    #Get both success and failure sequences and store together with labels
    store_sequences(state_data_success,state_data_failure,points_per_seq,target_path)

def normalize_data(root_data, downsample_dataset):
    if downsample_dataset == False:
        data = read_data_from_file(root_data + 'data.csv')
    else:
        data = read_data_from_file(root_data + 'data_downsampled.csv')
    
    #First, read in all data, calculate average, and store absolute maximum value for each state
    state_avg = [0, 0, 0]
    state_max = [0, 0, 0]
    n_seq = 0

    for row in data:
        data_i = row[:3]
        for i, state in enumerate(data_i):
            state_avg[i] = state_avg[i] + float(state)
            if abs(float(state))>state_max[i]:
                state_max[i] = abs(float(state))

        n_seq = n_seq + 1

    for i, state in enumerate(state_avg): state_avg[i] = state_avg[i]/n_seq

    #Save average and max values for test normalization
    np.savetxt(root_data  + 'state_average.csv',state_avg,fmt="%f", delimiter=',')
    np.savetxt(root_data  + 'state_max.csv',state_max,fmt="%f", delimiter=',')

    #Then, go through data and normalize
    data_normalized = []
    for i, row in enumerate(data):
        for s in range(3): #3 states
            for k in range(int(len(row)/3)):
                row[3*k+s] = (float(row[3*k+s]) - state_avg[s])/state_max[s]
                row[3*k+s] = 0.001*np.round(1000*row[3*k+s])
        data_normalized.append(row)

    np.savetxt(root_data + 'data_normalized.csv', data_normalized,fmt="%f", delimiter=',')

def downsample_data(root_data,points_per_seq):
    data = read_data_from_file(root_data + 'data.csv')
    labels = read_data_from_file(root_data + 'labels.csv')

    labels_success = []
    data_success = []
    labels_failure = []
    data_failure = []
    for i, l in enumerate(labels):
        if int(l[0])==0: 
            data_success.append(data[i])
            labels_success.append(labels[i])
        else:
            data_failure.append(data[i])
            labels_failure.append(labels[i])

    #Downsample success sequences to 10%
    n_down = 0.1
    random.seed(0)
    i_down = random.sample(range(0, int(len(data_success)/points_per_seq)-1), int(n_down * len(data_success)/points_per_seq))

    #Go through data and downsample
    data_down = data_failure
    labels_down = labels_failure

    for i in i_down:
        data_i = data_success[points_per_seq*i:points_per_seq*(i+1)]
        labels_i = labels_success[points_per_seq*i:points_per_seq*(i+1)]
        d_i = []
        l_i = []
        for d in data_i: data_down.append(d)
        for l in labels_i: labels_down.append(l)

    #Finally, turn into floats/integers
    for i, data in enumerate(data_down):
        d_i = []
        for d in data: d_i.append(float(d))
        data_down[i] = d_i
        l_i = []
        for l in labels_down[i]: l_i.append(int(l))
        labels_down[i] = l_i

    np.savetxt(root_data + 'data_downsampled.csv', data_down,fmt="%f", delimiter=',')
    np.savetxt(root_data + 'labels_downsampled.csv', labels_down,fmt="%i", delimiter=',')

if __name__ == "__main__":
    #Extract failure sequences and success sequences from entire recording
    #Split up recording into success portion and failure portion
    #Failure portion is the specified duration before the failure, success is the rest
    duration = 10
    frequency = 20
    #Number of state points per sequence defined by duration and frequency
    points_per_seq = duration*frequency

    #Decide if data set is built from scratch or if new data is added to existing data set
    create_dataset =  True
    downsample_dataset = True
    #Decide if data set should be normalized at the end
    normalize_dataset = True
    
    path_target = './data_introspection/'

    #Specify path to sequence folder and target paths
    subsets = ['part1','part2']

    for subset in subsets:
        root_path = '../data/' + subset + '/LBC/state/'

        #Find all runs and iterate, then for each run find all simulations and iterate
        run_paths = glob(root_path + "/*/", recursive = False)

        for run in run_paths:
            print(run)
            simulation_paths = glob(run + "/*/", recursive = False)

            for simulation in simulation_paths:
                #Get paths of all raw sequences
                scenario_paths = [f for f in listdir(simulation) if isfile(join(simulation, f))]

                if create_dataset == True:
                    for scenario in scenario_paths:
                        #Extract success and failure sequences from each raw scenario
                        extract_sequences(simulation + scenario, path_target, points_per_seq)

    if downsample_dataset == True:
        downsample_data(path_target,points_per_seq)

    if normalize_dataset == True:
        normalize_data(path_target,downsample_dataset) 