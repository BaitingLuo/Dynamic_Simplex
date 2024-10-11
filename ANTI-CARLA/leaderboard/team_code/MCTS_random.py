import logging
import math

import numpy as np
import decimal
import mcts_scorer as ms
EPS = 1e-8
import random
log = logging.getLogger(__name__)
from itertools import *
from scipy.stats import poisson
from reliability.Distributions import Weibull_Distribution
import csv
from pytorch.NNet import NNetWrapper as LBC_score_nn
from AP_pytorch.NNet import NNetWrapper as AP_score_nn

from pytorch_OOD.NNet import NNetWrapper as ood_nn
import torch
import scipy
import scipy.stats
class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, args, fs, random_weather, failure_type):
        self.future_scenes = fs
        self.args = args
        self.failure_type = failure_type
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.length_weather = random_weather
        actions = [0, 1, 2]
        action_set = list(product(actions, repeat=random_weather))
        self.random_space = action_set
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.current_time = 0
        #self.dist = Weibull_Distribution(alpha=1000000, beta=410000)  # this created the distribution object
        size = 1000
        x = scipy.arange(1000)
        y = scipy.stats.beta.rvs(6, 2, size=size, random_state=40) * 5000000
        # y2 = scipy.stats.beta.rvs(6, 2, size=size, random_state=40) * 3000
        dis = getattr(scipy.stats, 'beta')
        param = dis.fit(y)
        self.beta_args = param[:-2]
        self.loc = param[-2]
        self.scale = param[-1]
        self.sensor_failure_time = -100
        #self.sensor_failure_rate = scipy.stats.beta.sf(self.current_time, *args, loc=loc, scale=scale)
        # self.transition_time = np.random.normal(20, 1, 1)[0]
        self.transition_time = 20
        self.general_failure_rate = scipy.stats.beta.sf(self.current_time, *self.beta_args, loc=self.loc, scale=self.scale)
        self.occlusion_path = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/trained_monitors/prior_data/occlusion.csv"
        self.occlusion_table = self.read_occlusion(self.occlusion_path)
        #self.oodnn = ood_nn(9, 1)
        self.LBC_performance = LBC_score_nn(10,1)
        self.LBC_safe = LBC_score_nn(10, 1)
        self.AP_scorenn = AP_score_nn(9, 1)
        #self.oodnn.load_checkpoint(folder="./OOD_estimation/", filename='temp.pth.tar')
        self.LBC_performance.load_checkpoint(folder="./LBC_performance_score/", filename='temp.pth.tar')
        self.LBC_safe.load_checkpoint(folder="./LBC_safe_score/", filename='temp.pth.tar')
        self.AP_scorenn.load_checkpoint(folder="./AP_score_estimation/", filename='temp.pth.tar')
        self.OOD_time_estimation = ood_nn(9, 1)
        self.time_interval = ood_nn(9, 1)
        self.OOD_time_estimation.load_checkpoint(folder="OOD_time", filename='temp.pth.tar')
        self.time_interval.load_checkpoint(folder="interval_estimation", filename='temp.pth.tar')
        #self.LBC_performance, self.LBC_safe, self.AP_scorenn, self.OOD_time_estimation, self.time_interval= LBC_performance, LBC_safe, AP_scorenn, OOD_time_estimation, time_interval

        self.alpha1 = 1 #weight metric for performant score --- 1 indicates increasing the importance of performance to the maximum level, 0 indicates igorance of the importance of performance.
        self.alpha2 = 1 #weight metric for safety score  --- Genrally, don't consider tuning this parameter as any smaller number may compromise safety
        self.alpha3 = 0.5 #weight metric for switch score --- 1 indicates penalize switch at the ultimate level, while 0 indicates no penalty fpr switching
        self.discount = 0.9
    def getActionProb(self, state, future_scene, action_set, distance, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #print(self.AP_scorenn.predict(state[]))
        #print(self.LBC_safe.predict([0, 0, 0, 0.2, 0.1, 0, 0.6, 0, 0.1, 0.1]))
        #print(self.LBC_performance.predict([0, 0, 0, 0.2, 0.1, 0, 0.6, 0, 0.1, 0.1]))
        if len(future_scene) > 3:
            future_scene = future_scene[:3]
        #self.velocity_dataset = matching_lut_entries_v
        #self.train_dataset = matching_lut_entries
        self.distance_to_scene = distance
        #run MCTS iterations for each scene here
        for i in range(self.args.numMCTSSims):
            self.search(state, future_scene, action_set, 0, 0, self.current_time)
        s = list(state[:10])
        s.extend([state[-2]])
        if not isinstance(s,tuple):
            s = tuple(s)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(2)]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            #mean that no valid actions to take now
            if len(bestAs) == len(action_set):
                probs = [0] * len(counts)
                return probs
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            probs = [0] * len(counts)
            return probs
        probs = [x / counts_sum for x in counts]
        return probs

    def sample_occlusion(self, current_state):
        s = current_state
        s = list(s)
        occlusion_likelihood = 0
        for array in self.occlusion_table:
            if (int(array[0]) == (int(s[0]*100))) and (int(array[1]) == int(s[1])*100) and (
                    int(array[2]) == int(s[2])*100) and (int(array[3]) == int(s[3])*100) and (
                    int(array[4]) == int(s[4])*100) and (int(array[5]) == int(s[5])*100):
                occlusion_likelihood = float(array[6])
        chance = random.uniform(0, 1)
        if chance < occlusion_likelihood:
            s[8] = 1
        else:
            s[8] = 0
        return tuple(s)

    def read_occlusion(self,data_path):
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
                x.append(int(row[4]))
                x.append(int(row[5]))
                x.append(float(row[6]))
                X.append(x)
        return X

    def random_sample_state(self, current_state):
        s = current_state
        s = list(s)

        sampling_change = random.choice(self.random_space)
        for i in range(self.length_weather):
            if sampling_change[i] == 1:
                if s[4+i] < 1:
                    s[4+i] += 0.05
                    s[4+i] = round(s[4+i], 2)
            elif sampling_change[i] == 2:
                if s[4+i] > 0:
                    s[4+i] = round(s[4+i], 2)
        return tuple(s)

    def rollout(self, leaf_state, future_scene, sw, time, total_time):
        #CT count the time has passed since lasy weather change
        #-3 tree depth -2 controller -1 scene depth
        CT = time
        elapsed_time = total_time
        current_state = leaf_state
        depth = current_state[-1]
        tree_depth = current_state[-3]
        switch_count = sw
        perf_score_list = []
        safe_score_list = []
        depth_count = 0
        R = 0
        #use a while loop to substitute recursion for rollout
        #print(current_state)
        while (depth < len(future_scene) - 1):
            if current_state[-2] == 0:
                action = np.random.choice([0,1])
                #print(action)
            else:
                action = 0
            if action == 1:
                #print(current_state)
                switch_count += 1
                if not isinstance(current_state, list):
                    current_state = list(current_state)
                current_state[-2] = 1
            #retrieve velocity from surrogate model
            state_for_prediction = list(current_state[:10])
            #ood_v =
            if current_state[-2] == 1:
                perf_score, safe_score = self.LBC_performance.predict(state_for_prediction), self.LBC_safe.predict(state_for_prediction)
                #print(safe_score)
            else:
                perf_score, safe_score = self.AP_scorenn.predict(state_for_prediction[:-1])
            #velocity, _ = ms.vpredict(current_state, self.velocity_dataset)
            velocity = perf_score * 8
            #print(state_for_prediction,velocity)

            #state_for_prediction = list(current_state[:11])+[current_state[-2]]
            #perf_score, safe_score = ms.predict(state_for_prediction, self.train_dataset)
            ap_noise = np.random.normal(2, 1, 1)
            LBC_noise = np.random.normal(4, 1, 1)

            estimate_time = -1
            for distance in self.distance_to_scene:
                if current_state[-4] < distance:
                    #print(distance-current_state[-4])
                    estimate_time = (distance-current_state[-4]) / velocity
                    break
            #estimate time indicates the approximate time may take to arrive in the closest location characterized structural transitions
            if estimate_time == -1:
                #state_for_prediction = list(current_state[:11]) + [current_state[-2]]
                #perf_score, safe_score = ms.predict(state_for_prediction, self.train_dataset)
                state_for_prediction = list(current_state[:10])
                if current_state[-2] == 1:
                    perf_score, safe_score = self.LBC_performance.predict(state_for_prediction), self.LBC_safe.predict(state_for_prediction)
                else:
                    perf_score, safe_score = self.AP_scorenn.predict(state_for_prediction[:-1])
                if float(tree_depth) == 0:
                    deno = 1
                else:
                    deno = float(tree_depth)
                if (switch_count > 0):
                    switch_reward = (switch_count - 1)
                else:
                    switch_reward = 0
                R += (0.9**(depth_count))*(self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno))
                return R
            current_state = list(current_state)
            state_for_prediction = current_state[:9]
            if current_state[9] == 0:
                next_oodtime = self.time_interval.predict(state_for_prediction)
            else:
                ood_period = self.OOD_time_estimation.predict(state_for_prediction)
            #if self.sensor_failure_rate < random.uniform():
            #    current_state[7] = 1
            #print(velocity)
            sensor_failure_time = float()
            if current_state[7] == 0 and self.sensor_failure_time == -100 and self.general_failure_rate < 0.9:
                if (estimate_time + CT) > self.transition_time:
                    temp = self.transition_time
                else:
                    temp = estimate_time
                if torch.is_tensor(elapsed_time):
                    temp_elapsed_time = elapsed_time.cpu()
                else:
                    temp_elapsed_time = elapsed_time
                for i in range(int(temp)):
                    reliability = scipy.stats.beta.sf(i + temp_elapsed_time, *self.beta_args, loc=self.loc, scale=self.scale)
                    if reliability < random.uniform(0,1):
                        self.sensor_failure_time = i + 1
                        break
            if ((estimate_time+CT)< self.transition_time):
                flag = True
                failure_flag = False
                current_state[8] += round(random.uniform(-1, 1), 1)
                current_state[8] = round(current_state[8], 1)
                if current_state[8] < 0.1:
                    current_state[8] = 0.1
                if current_state[8] > 1.2:
                    current_state[8] = 1.2
                if (self.sensor_failure_time != -100):
                    if current_state[9] == 0:
                        if self.sensor_failure_time <= next_oodtime:
                            failure_flag = True
                    elif current_state[9] == 1:
                        if self.sensor_failure_time <= ood_period:
                            failure_flag = True
                    if failure_flag:
                        current_state[7] = 1
                        current_state[-4] += velocity * self.sensor_failure_time
                        CT += self.sensor_failure_time
                        elapsed_time += self.sensor_failure_time
                        self.transition_time -= self.sensor_failure_time
                        flag = False
                        self.sensor_failure_time = -100
                if current_state[9] == 0 and flag:
                    if next_oodtime < (estimate_time+CT) and next_oodtime > 1:
                        current_state[9] = 1
                        current_state[-4] += velocity * next_oodtime
                        CT += next_oodtime
                        elapsed_time += next_oodtime
                        self.transition_time -= next_oodtime
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= next_oodtime
                        flag = False
                elif current_state[9] == 1 and flag:
                    if ood_period < (estimate_time + CT):
                        current_state[9] = 0
                        current_state[-4] += velocity * ood_period
                        CT += ood_period
                        elapsed_time += ood_period
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= ood_period
                        self.transition_time -= ood_period
                        flag = False
                if flag:
                    current_state[0] = float(future_scene[depth+1][0])/10
                    current_state[1] = float(future_scene[depth+1][1])/10
                    current_state[2] = float(future_scene[depth+1][2])/10
                    current_state[3] = float(future_scene[depth+1][3])/10
                    current_state[-4] += float(velocity * estimate_time)
                    CT += estimate_time
                    elapsed_time += estimate_time
                    self.transition_time -= estimate_time
                    if float(self.failure_type) == 2:
                        current_state = self.sample_occlusion(current_state)
                current_state = tuple(current_state)
                depth += 1
            elif (CT+estimate_time)==self.transition_time:
                flag = True
                failure_flag = False
                current_state[8] += round(random.uniform(-1, 1), 1)
                current_state[8] = round(next_s[8], 1)
                if current_state[8] < 0.1:
                    current_state[8] = 0.1
                if current_state[8] > 1.2:
                    current_state[8] = 1.2
                if (self.sensor_failure_time != -100):
                    if current_state[9] == 0:
                        if self.sensor_failure_time <= next_oodtime:
                            failure_flag = True
                    elif current_state[9] == 1:
                        if self.sensor_failure_time <= ood_period:
                            failure_flag = True
                    if failure_flag:
                        current_state[7] = 1
                        current_state[-4] += velocity * self.sensor_failure_time
                        CT += self.sensor_failure_time
                        elapsed_time += self.sensor_failure_time
                        self.transition_time -= self.sensor_failure_time
                        flag = False
                        self.sensor_failure_time = -100
                if current_state[9] == 0 and flag:
                    if next_oodtime < (estimate_time+CT) and next_oodtime > 1:
                        current_state[9] = 1
                        current_state[-4] += velocity * next_oodtime
                        CT += next_oodtime
                        elapsed_time += next_oodtime
                        self.transition_time -= next_oodtime
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= next_oodtime
                        flag = False
                elif current_state[9] == 1 and flag:
                    if ood_period < (estimate_time + CT):
                        current_state[9] = 0
                        current_state[-4] += velocity * ood_period
                        CT += ood_period
                        elapsed_time += ood_period
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= ood_period
                        # self.transition_time = 20
                        self.transition_time -= ood_period
                        flag = False

                if flag:
                    current_state[0] = float(future_scene[depth+1][0])/10
                    current_state[1] = float(future_scene[depth+1][1])/10
                    current_state[2] = float(future_scene[depth+1][2])/10
                    current_state[3] = float(future_scene[depth+1][3])/10
                    current_state[-4] += float(velocity * estimate_time)
                    CT = 0
                    elapsed_time += estimate_time
                    current_state = self.random_sample_state(current_state)
                    self.transition_time = 20
                    if float(self.failure_type) == 2:
                        current_state = self.sample_occlusion(current_state)


                current_state = tuple(current_state)
                #self.transition_time = np.random.normal(20, 1, 1)[0]


                depth += 1
            elif (CT+estimate_time)>self.transition_time:
                flag = True
                failure_flag = False
                current_state[8] += round(random.uniform(-0.5, 0.5), 1)
                current_state[8] = round(current_state[8], 1)
                if current_state[8] < 0.1:
                    current_state[8] = 0.1
                if current_state[8] > 1.2:
                    current_state[8] = 1.2
                if (self.sensor_failure_time != -100):
                    if current_state[9] == 0:
                        if self.sensor_failure_time <= next_oodtime:
                            failure_flag = True
                    elif current_state[9] == 1:
                        if self.sensor_failure_time <= ood_period:
                            failure_flag = True
                    if failure_flag:
                        current_state[7] = 1
                        current_state[-4] += velocity * self.sensor_failure_time
                        CT += self.sensor_failure_time
                        elapsed_time += self.sensor_failure_time
                        self.transition_time -= self.sensor_failure_time
                        flag = False
                        self.sensor_failure_time = -100
                if current_state[9] == 0 and flag:
                    if next_oodtime < (self.transition_time) and next_oodtime > 1:
                        current_state[9] = 1
                        current_state[-4] += velocity * next_oodtime
                        CT += next_oodtime
                        elapsed_time += next_oodtime
                        self.transition_time -= next_oodtime
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= next_oodtime
                        flag = False
                elif current_state[9] == 1 and flag:
                    if ood_period < (self.transition_time):
                        current_state[9] = 0
                        current_state[-4] += velocity * ood_period
                        CT += ood_period
                        if (failure_flag == False) and (self.sensor_failure_time != -100):
                            self.sensor_failure_time -= ood_period
                        elapsed_time += ood_period
                        # self.transition_time = 20

                        self.transition_time -= ood_period
                        flag = False
                if flag:
                    current_state[-4] += velocity * self.transition_time
                    elapsed_time += self.transition_time
                    self.transition_time = 20
                    current_state = self.random_sample_state(current_state)
                    if int(self.failure_type) == 2:
                        current_state = self.sample_occlusion(current_state)
                    CT = 0

                #self.transition_time = np.random.normal(20, 1, 1)[0]

                current_state = tuple(current_state)

            depth_count += 1
            tree_depth += 1
            if float(tree_depth) == 0:
                deno = 1
            else:
                deno = float(tree_depth)
            if (switch_count > 0):
                switch_reward = (switch_count - 1)
            else:
                switch_reward = 0
            R += (self.discount**(depth_count))*(self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno))
        #print(elapsed_time)
        return R

    def search(self, state, future_scene, action_set, c_sw, time, total_time):
        current_s = list(state[:10])
        current_s.extend([state[-2]])
        current_s = tuple(current_s)
        #print(self.sensor_failure_time)
        s = state
        CT = time
        elapsed_time = total_time
        count_switch = c_sw
        if not isinstance(s,tuple):
            s = tuple(state)
        scene_depth = s[-1]
        if scene_depth == len(future_scene)-1:
            self.Es[current_s] = 1 #1 indicate the tree has ended
        else:
            self.Es[current_s] = 0 #0 indicate the tree has not ended yet

        if self.Es[current_s] != 0:
            # terminal node
            T_s = list(state)
            #T_s =T_s
            T_s = tuple(T_s)
            state_for_prediction = list(T_s[:10])
            #perf_score, safe_score = ms.predict(T_s,self.train_dataset)
            if T_s[-2] == 1:
                perf_score, safe_score = self.LBC_performance.predict(state_for_prediction), self.LBC_safe.predict(state_for_prediction)
            else:
                perf_score, safe_score = self.AP_scorenn.predict(state_for_prediction[:-1])
            if (count_switch > 0):
                switch_reward = (count_switch - 1)
            else:
                switch_reward = 0
            if float(s[-3]) == 0:
                deno = 1
            else:
                deno = float(s[-3])

            v = self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno)
            return v
        if current_s not in self.Ns:
            # leaf node
            #rollout happens here
            roll_s = s
            v = self.rollout(roll_s, future_scene, count_switch, CT, elapsed_time)
            self.Ns[current_s] = 0
            return v

        #UCB happens here
        cur_best = -float('inf')
        best_act = -1
        for a in range(2):
            if (current_s, a) in self.Qsa:
                u = self.Qsa[(current_s, a)] + self.args.cpuct * math.sqrt(np.log(self.Ns[current_s]) / (self.Nsa[(current_s, a)]))
            else:
                u = float('inf')
                #print(u)
            if u > cur_best:
                cur_best = u
                best_act = a
        a = best_act
        next_s = list(s)
        T_s = list(s)
        if a == 1:
            count_switch += 1
            #next_s = list(s)
            if next_s[-2] == 0:
                T_s[-2] = 1
                next_s[-2] = 1
            elif next_s[-2] == 1:
                T_s[-2] = 0
                next_s[-2] = 0

        #T_s = T_s[:11] + [T_s[-2]]
        T_s = tuple(T_s)
        estimate_time = -1
        if T_s[-2] == 1:
            perf_score, safe_score = self.LBC_performance.predict(T_s[:10]), self.LBC_safe.predict(T_s[:10])
        else:
            perf_score, safe_score = self.AP_scorenn.predict(T_s[:9])
        #velocity = ms.vpredict(T_s, self.velocity_dataset)
        velocity = perf_score * 8
        #print(T_s[-2], safe_score)
        for distance in self.distance_to_scene:
            if next_s[-4] < distance:
                estimate_time = (distance-float(next_s[-4])) / velocity
                break
        #print(next_s,next_s[-4])
        if estimate_time == -1:
            #perf_score, safe_score = ms.predict(T_s,self.train_dataset)
            if (count_switch > 0):
                switch_reward = (count_switch - 1)
            else:
                switch_reward = 0
            if float(s[-3]) == 0:
                deno = 1
            else:
                deno = float(s[-3])
            #print(self.alpha3*(switch_reward/deno))
            v = self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno)
            return v
        state_for_prediction = T_s[:9]
        if T_s[9] == 0:
            next_oodtime = self.time_interval.predict(state_for_prediction)
        else:
            ood_period = self.OOD_time_estimation.predict(state_for_prediction)
        #state transitions happen here
        if T_s[7] == 0 and self.sensor_failure_time == -100 and self.general_failure_rate < 0.9:
            if (CT+estimate_time) < self.transition_time:
                temp = estimate_time
            else:
                temp = self.transition_time
            if torch.is_tensor(elapsed_time):
                temp_elapsed_time = elapsed_time.cpu()
            else:
                temp_elapsed_time = elapsed_time
            for i in range(int(temp)):
                reliability = scipy.stats.beta.sf(i + temp_elapsed_time, *self.beta_args, loc=self.loc, scale=self.scale)
                if reliability < random.uniform(0,1):
                    self.sensor_failure_time = i + 1
                    break
        if ((CT + estimate_time) < self.transition_time):
            flag = True
            failure_flag = False
            next_s[-3] += 1
            next_s[8] += round(random.uniform(-1, 1), 1)
            next_s[8] = round(next_s[8], 1)
            if next_s[8] < 0.1:
                next_s[8] = 0.1
            if next_s[8] > 1.2:
                next_s[8] = 1.2
            if (self.sensor_failure_time != -100):
                if T_s[9] == 0:
                    if self.sensor_failure_time <= next_oodtime:
                        failure_flag = True
                elif T_s[9] == 1:
                    if self.sensor_failure_time <= ood_period:
                        failure_flag = True
                if failure_flag:
                    next_s[7] = 1
                    next_s[-4] += velocity * self.sensor_failure_time
                    CT += self.sensor_failure_time
                    elapsed_time += self.sensor_failure_time
                    self.transition_time -= self.sensor_failure_time
                    flag = False
                    self.sensor_failure_time = -100
            if T_s[9] == 0 and flag:
                if next_oodtime < (estimate_time + CT) and next_oodtime > 1:
                    next_s[9] = 1
                    next_s[-4] += velocity * next_oodtime
                    CT += next_oodtime
                    elapsed_time += next_oodtime
                    self.transition_time -= next_oodtime
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= next_oodtime
                    flag = False
            elif T_s[9] == 1 and flag:
                if ood_period < (estimate_time + CT):
                    next_s[9] = 0
                    next_s[-4] += velocity * ood_period
                    CT += ood_period
                    elapsed_time += ood_period
                    self.transition_time -= ood_period
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= ood_period
                    flag = False
            if flag:
                next_s[-1] += 1
                next_s[0] = float(future_scene[next_s[-1]][0])/10
                next_s[1] = float(future_scene[next_s[-1]][1])/10
                next_s[2] = float(future_scene[next_s[-1]][2])/10
                next_s[3] = float(future_scene[next_s[-1]][3])/10
                next_s[-4] += float(velocity * estimate_time)
                if int(self.failure_type) == 2:
                    next_s = self.sample_occlusion(next_s)
                CT += estimate_time
                elapsed_time += estimate_time
                self.transition_time -= estimate_time
            next_s = tuple(next_s)
        elif (CT+estimate_time) == self.transition_time:
            flag = True
            failure_flag = False
            next_s[8] += round(random.uniform(-1, 1), 1)
            next_s[8] = round(next_s[8],1)
            if next_s[8] < 0.1:
                next_s[8] = 0.1
            if next_s[8] > 1.2:
                next_s[8] = 1.2
            next_s[-3] += 1
            if (self.sensor_failure_time != -100):
                if T_s[9] == 0:
                    if self.sensor_failure_time <= next_oodtime:
                        failure_flag = True
                elif T_s[9] == 1:
                    if self.sensor_failure_time <= ood_period:
                        failure_flag = True
                if failure_flag:
                    next_s[7] = 1
                    next_s[-4] += velocity * self.sensor_failure_time
                    CT += self.sensor_failure_time
                    elapsed_time += self.sensor_failure_time
                    self.transition_time -= self.sensor_failure_time
                    flag = False
                    self.sensor_failure_time = -100
            if T_s[9] == 0 and flag:
                if next_oodtime < (estimate_time + CT) and next_oodtime > 1:
                    next_s[9] = 1
                    next_s[-4] += velocity * next_oodtime
                    CT += next_oodtime
                    elapsed_time += next_oodtime
                    self.transition_time -= next_oodtime
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= next_oodtime

                    flag = False
            elif T_s[9] == 1 and flag:
                if ood_period < (estimate_time + CT):
                    next_s[9] = 0
                    next_s[-4] += velocity * ood_period
                    CT += ood_period
                    elapsed_time += ood_period
                    self.transition_time -= ood_period
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= ood_period
                    flag = False
            if flag:
                next_s[-1] += 1
                next_s[0] = float(future_scene[next_s[-1]][0])/10
                next_s[1] = float(future_scene[next_s[-1]][1])/10
                next_s[2] = float(future_scene[next_s[-1]][2])/10
                next_s[3] = float(future_scene[next_s[-1]][3])/10
                next_s[-4] += float(velocity*estimate_time)
                next_s = self.random_sample_state(next_s)
                if int(self.failure_type) == 2:
                    next_s = self.sample_occlusion(next_s)
                next_s = tuple(next_s)
                # self.transition_time = np.random.normal(20, 1, 1)[0]
                self.transition_time = 20
                elapsed_time += estimate_time
                CT = 0
            next_s = tuple(next_s)
        elif (CT+estimate_time) > self.transition_time:
            flag = True
            failure_flag = False
            next_s[8] += round(random.uniform(-0.5, 0.5), 1)
            next_s[8] = round(next_s[8], 1)
            if next_s[8] < 0.1:
                next_s[8] = 0.1
            if next_s[8] > 1.2:
                next_s[8] = 1.2
            next_s[-3] += 1
            if (self.sensor_failure_time != -100):
                if T_s[9] == 0:
                    if self.sensor_failure_time <= next_oodtime:
                        failure_flag = True
                elif T_s[9] == 1:
                    if self.sensor_failure_time <= ood_period:
                        failure_flag = True
                if failure_flag:
                    next_s[7] = 1
                    next_s[-4] += velocity * self.sensor_failure_time
                    CT += self.sensor_failure_time
                    elapsed_time += self.sensor_failure_time
                    self.transition_time -= self.sensor_failure_time
                    flag = False
                    self.sensor_failure_time = -100
            if T_s[9] == 0 and flag:
                if next_oodtime < self.transition_time and next_oodtime > 1:
                    next_s[9] = 1
                    next_s[-4] += velocity * next_oodtime
                    CT += next_oodtime
                    elapsed_time += next_oodtime
                    self.transition_time -= next_oodtime
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= next_oodtime
                    flag = False
            elif T_s[9] == 1 and flag:
                if ood_period < self.transition_time:
                    next_s[9] = 0
                    next_s[-4] += velocity * ood_period
                    CT += ood_period
                    elapsed_time += ood_period
                    self.transition_time -= ood_period
                    if (failure_flag == False) and (self.sensor_failure_time != -100):
                        self.sensor_failure_time -= ood_period
                    flag = False
            if flag:
                next_s[-4] += velocity * self.transition_time
                next_s = self.random_sample_state(next_s)
                if int(self.failure_type) == 2:
                    next_s = self.sample_occlusion(next_s)
                elapsed_time += self.transition_time
                # self.transition_time = np.random.normal(20, 1, 1)[0]
                self.transition_time = 20
                CT = 0
            next_s = tuple(next_s)
        if T_s not in self.Vs.keys():
            #perf_score, safe_score = ms.predict(T_s,self.train_dataset)
            if T_s[-2] == 1:
                perf_score, safe_score = self.LBC_performance.predict(T_s[:10]), self.LBC_safe.predict(T_s[:10])
            else:
                perf_score, safe_score = self.AP_scorenn.predict(T_s[:9])
            if (count_switch > 0):
                switch_reward = (count_switch - 1)
            else:
                switch_reward = 0
            if float(next_s[-3]) == 0:
                deno = 1
            else:
                deno = float(next_s[-3])
            if T_s[-2] == 0:
                self.Vs[T_s] = self.alpha1*perf_score - self.alpha2*(safe_score)
                v = self.alpha1*perf_score - self.alpha2*(safe_score)
            else:
                self.Vs[T_s] = self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno)
                v =  self.alpha1*perf_score - self.alpha2*(safe_score) - self.alpha3*(switch_reward/deno)
            #print(self.alpha1*perf_score, self.alpha2*safe_score, self.alpha3*(switch_reward/deno))
            #print(T_s)
            #print(self.alpha1*perf_score,self.alpha2*torch.round(safe_score),self.alpha3*(switch_reward/deno))
            #print(v)
            v += self.discount * self.search(next_s, future_scene, action_set, count_switch, CT, elapsed_time)
        else:
            v = self.Vs[T_s]
            v += self.discount * self.search(next_s, future_scene, action_set, count_switch, CT, elapsed_time)
        #backpropagation happens here
        if (current_s, a) in self.Qsa:
            self.Qsa[(current_s, a)] = (self.Nsa[(current_s, a)] * self.Qsa[(current_s, a)] + v) / (self.Nsa[(current_s, a)] + 1)
            self.Nsa[(current_s, a)] += 1
        else:
            self.Qsa[(current_s, a)] = v
            self.Nsa[(current_s, a)] = 1

        self.Ns[current_s] += 1
        return v

