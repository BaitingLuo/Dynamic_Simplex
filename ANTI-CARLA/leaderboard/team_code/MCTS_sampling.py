
import math
from MCTS_random import MCTS
import csv
import numpy as np
from pytorch.NNet import NNetWrapper as LBC_score_nn
from AP_pytorch.NNet import NNetWrapper as AP_score_nn
from collections import Counter
from pytorch_OOD.NNet import NNetWrapper as ood_nn
from torch.multiprocessing import Process, Queue, set_start_method
import torch
import random
import scipy
import scipy.stats
import time
try:
     set_start_method('spawn')
except RuntimeError:
    pass
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def read_dataset(data_path,entry):
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
            #x.append(create_bin(int(row[4])))
            #x.append(create_bin(int(row[5])))
            #x.append(create_bin(int(row[6])))
            #x.append(create_bin(int(row[7])))
            x.append(int(row[4]))
            x.append(int(row[5]))
            x.append(int(row[6]))
            x.append(int(row[7]))
            x.append(row[8])
            x.append(row[9])
            x.append(row[10])
            x.append(float(row[11]))
            x.append(float(row[12]))
            x.append(float(row[13]))
            x.append(float(row[14]))
            x.append(float(row[15]))
            x.append(float(row[16]))
            X.append(x)
        X = np.array(X)

    match,weather,driving_score,safe_perf_scores = search(entry,X)
    return match,weather,driving_score,safe_perf_scores

def read_velocity(data_path,entry):
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
            #x.append(create_bin(int(row[4])))
            #x.append(create_bin(int(row[5])))
            #x.append(create_bin(int(row[6])))
            #x.append(create_bin(int(row[7])))MCTS
            x.append(int(row[4]))
            x.append(int(row[5]))
            x.append(int(row[6]))
            x.append(int(row[7]))
            x.append(row[8])
            x.append(row[9])
            x.append(row[10])
            x.append(float(row[11]))
            x.append(float(row[12]))
            x.append(float(row[13]))
            x.append(float(row[14]))
            X.append(x)
        X = np.array(X)

    match = search_v(entry,X)
    return match

def search_v(entry,X_train):
    match = []
    weather = []
    driving_score = []
    safe_perf = []
    for ent in entry:
        match_1 = []
        weather_1 = []
        safe_perf_1 = []
        for val in X_train:
            if str(ent[0]) == str(val[0]) and str(ent[1]) == str(val[1]) and str(ent[2]) == str(val[2]) and str(ent[3]) == str(val[3]):
                match_1.append(val)
                weather_1.append([val[4],val[5],val[6],val[7]])
                safe_perf_1.append([val[11],val[12],val[13],val[14]])
        match.append(match_1)
        weather.append(weather_1)
        safe_perf.append(safe_perf_1)

    return match


def search(entry,X_train):
    match = []
    weather = []
    driving_score = []
    safe_perf = []
    for ent in entry:
        match_1 = []
        weather_1 = []
        driving_score_1 = []
        safe_perf_1 = []
        for val in X_train:
            if str(ent[0]) == str(val[0]) and str(ent[1]) == str(val[1]) and str(ent[2]) == str(val[2]) and str(ent[3]) == str(val[3]):
                match_1.append(val)
                weather_1.append([val[4],val[5],val[6],val[7]])
                driving_score_1.append([val[15],val[16]])
                safe_perf_1.append([val[11],val[12],val[13],val[14]])
        match.append(match_1)
        weather.append(weather_1)
        driving_score.append(driving_score_1)
        safe_perf.append(safe_perf_1)

    return match,weather,driving_score, safe_perf

def mcts_planner(current_scene, semantic_data, failure_type, distance, LBC_performance, LBC_safe, AP_scorenn, OOD_time_estimation, time_interval, mcts_queue):
    """
    MCTS Planner
    """
    print("I am running")
    alpha_args = dotdict({
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'numMCTSSims': 500,
        'cpuct': math.sqrt(2),
    })
    print(current_scene, semantic_data, failure_type, distance)
    mcts = MCTS(alpha_args, semantic_data, 3, failure_type)
    temp = 1
    #print(self.current_scene)
    #scene_for_tree = self.current_scene.copy()
    #scene_for_tree[8] = self.current_traffic_density

    policy = mcts.getActionProb(current_scene, semantic_data, [0,1],
                                     distance, temp=temp)
    print("############################################")
    #print("current scene:", self.current_scene)
    #print("current semantic data:", self.semantic_data[self.k // 2:5])
    print("policy", policy)
    print("############################################")
    ##print("policy",self.policy)
    # self.mcts_queue.queue.clear()
    mcts_queue.put(policy)



if __name__ == '__main__':
    #action_constraint = int(config['Constraints']['action_delta'])
    project_path = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA"
    train_data_path = project_path + "/trained_monitors/prior_data/mcts_lut.csv"
    #with open(path_lut, newline='') as csvfile:
    #    semantic_data = list(csv.reader(csvfile))
    #semantic_data = [[0,0,0,2],[0,0,0,5],[3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #semantic_data = [[0, 0, 0, 5], [3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #semantic_data =[[1, 0, 0, 1],[0, 2, 0, 1], [0, 0, 0, 3]]
    #semantic_data = [[1, 0, 0, 3], [1, 0, 0, 1], [0, 2, 0, 1], [0, 0, 0, 3]]
    semantic_data =  [[2,1,0,2], [0, 0, 0, 2],[0,0,0,2],[0,0,0,3],[0,1,0,0]]
    matching_lut_entries, matching_lut_weather, matching_lut_driving_score, matching_lut_safe_perf_scores = read_dataset(
        train_data_path, semantic_data)
    velocity_path = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/trained_monitors/mcts_lut.csv"
    matching_lut_entries_v = read_velocity(velocity_path, semantic_data)
    alpha_args = dotdict({
        'tempThreshold': 15,  #
        'updateThreshold': 0.6,
        'numMCTSSims': 500,  # Number of games moves for MCTS to simulate.
        'cpuct': math.sqrt(2),
    })
    y_d = 1
    train = True
    action_set = [0, 1]
    #s[-1] scene, s[-2] controller, s[-3] tree depth, s[-4] distance
    #s = [0, 0, 0, 2, 50, 95, 60, 70, 0, 0, 0, 0.52, 0.0, 0.44, 0, 0, 0, 0, 0]
    #future_scene = [[0,0,0,2],[0,0,0,5],[3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #s = [0, 0, 0, 5, 50, 50, 60, 70, 0, 0, 0, 0.52, 0.0, 0.44, 0,0, 0, 0]
    #future_scene = [[0, 0, 0, 5], [3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #s = [1, 0, 0, 3, 50, 100, 65, 70, 0, 0, 0, 0.71, 0.0, 0.83, 0.0, 0, 0, 0, 0]
    #future_scene = [[1, 0, 0, 3], [1, 0, 0, 1], [0, 2, 0, 1], [0, 0, 0, 3]]
    #s = [2, 1, 0, 2, 70, 10, 45, 1,0, 0, 0.39, 0.0, 0.91, 0.0, 0, 0, 0, 0]
    #s = [0.1, 0, 0, 0.2, 0, 0.15, 0.2, 0, 1, 0, 0, 0, 0, 0]
    #s = [0.1, 0.0, 0.0, 0.2, 0.5, 0.95, 0.6, 0, 0.9, 0, 0, 0, 0, 0]
    #s = [0.4, 0.0, 0.0, 0.1, 0.1, 0.8, 0.45, 0, 0.1, 0, 0, 0, 0, 0]
    #s = [0.4, 0.0, 0.0, 0.4, 0.2, 0.85, 0.9, 0, 0.2, 0, 0, 0, 0, 0]
    s = [0.0, 0.2, 0.0, 0.1, 0.8, 0.1, 0.35, 0, 0.8, 1, 0, 0, 0, 0]
    #s = [0, 2, 0, 1, 40, 85, 90, 70, 0, 0, 0, 0.51, 0.0, 0.54, 0.0, 0, 0, 0, 0]
    #future_scene = [[0.1, 0, 0, 0.2],[0,0,0,0.2],[0,0,0,0.3],[0,0.1,0,0]]
    #s = [0, 0, 0, 2, 50, 100, 70, 70, 0, 0, 0, 0.52, 0.0, 0.52, 0.0, 0, 0, 0, 0]
    #future_scene = [[1,0,0,2],[0,0,0,5],[3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #future_scene = [[1, 0, 0, 2], [0, 0, 0, 5], [3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #future_scene = [[4, 0, 0, 1], [4, 2, 0, 0]]
    #future_scene = [[4,0,0,4], [4,1,0,0], [3,0,1,0]]
    future_scene = [[0,2,0,1], [0,0,0,3]]
    #future_scene = [[0.2, 0.1, 0, 0.2], [0, 0, 0, 0.2], [0, 0, 0, 0.2], [0, 0, 0, 0.3], [0, 0.1, 0, 0]]
    #s = [0, 0, 0, 2,  50, 100, 70, 70, '0', '0', '0', '0.52', '0.0', '0.52', '0.0', 0, 0, 0, 0]
    #future_scene =[[0,0,0,2], [0, 0, 0, 5], [3, 0, 1, 6], [4, 0, 0, 1], [4, 2, 0, 0]]
    #mcts = MCTS(alpha_args, future_scene, 4,1)
    temp = 1
    size = 1000
    x = scipy.arange(1000)
    y = scipy.stats.beta.rvs(6,2,size = size, random_state = 40) * 5000000
    #y2 = scipy.stats.beta.rvs(6, 2, size=size, random_state=40) * 3000
    #print(y)
    dis = getattr(scipy.stats, 'beta')
    param = dis.fit(y)
    args = param[:-2]
    loc = param[-2]
    scale = param[-1]
    #print(scipy.stats.beta.sf(3000000, *args, loc=loc,scale=scale))
    #oodnn = ood_nn(9, 1)
    # NOTE: this is required for the ``fork`` method to work
    #oodnn.load_checkpoint(folder="./OOD_estimation/", filename='temp.pth.tar')
    LBC_performance = LBC_score_nn(10, 1)
    LBC_safe = LBC_score_nn(10, 1)
    AP_scorenn = AP_score_nn(9, 1)
    OOD_time_estimation = ood_nn(9, 1)
    time_interval = ood_nn(9, 1)
    OOD_time_estimation.load_checkpoint(folder="OOD_time", filename='temp.pth.tar')
    time_interval.load_checkpoint(folder="interval_estimation", filename='temp.pth.tar')
    LBC_performance.load_checkpoint(folder="LBC_performance_score", filename='temp.pth.tar')
    LBC_safe.load_checkpoint(folder="LBC_safe_score", filename='temp.pth.tar')
    AP_scorenn.load_checkpoint(folder="AP_score_estimation", filename='temp.pth.tar')
    print(AP_scorenn.predict([0.3, 0.0, 0.1, 0.6, 0.65, 0.0, 0.2, 0, 0.2]))
    print(torch.round(LBC_safe.predict([0.3, 0.0, 0.1, 0.6, 0.65, 0.0, 0.2, 0, 0.2, 0])+1))
    print(LBC_performance.predict([0.3, 0.0, 0.1, 0.6, 0.65, 0.0, 0.2, 0, 0.2, 0]))
    #[51, 130, 231, 308]
    mcts_queue = Queue()
    #for i in range(8):
    #    mcts = MCTS(alpha_args, future_scene, 3,1, LBC_performance, LBC_safe, AP_scorenn, OOD_time_estimation, time_interval)
    #    pi = mcts.getActionProb(s, future_scene, [0,1], [31.123796463012695, 108.38648796081543],temp=temp)
    #    print(pi)
    num_processes = 1
    processes = []
    [94.0076904296875]
    [23.214244842529297, 102.16483688354492, 203.2828598022461, 280.5455513000488]
    [358.00579833984375, 725.2097015380859]
    p = Process(target=mcts_planner, args=(
    s, future_scene, 0, [59.5031681060791], LBC_performance, LBC_safe, AP_scorenn, OOD_time_estimation, time_interval,
    mcts_queue))
    p.start()
    st = time.time()
    for rank in range(num_processes):
        processes.append(p)
    for p in processes:
        p.join()
    result_list = []
    for p in processes:
        print(p.is_alive())
    while not mcts_queue.empty():
        result_list.append(mcts_queue.get())
    action_list = [np.argmax(i) for i in result_list]
    print(action_list)
    action = Counter(action_list)
    print(action.most_common(1)[0][0])
    et = time.time()
    print(et-st)
    for i in range(8):
        mcts = MCTS(alpha_args, future_scene, 3,1)
        st = time.time()
        pi = mcts.getActionProb(s, future_scene, [0,1], [59.5031681060791],temp=temp)
        et = time.time()
        print(et - st)

