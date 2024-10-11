import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import cv2
import time
import math
import zmq
from matplotlib import cm
import torch
import torch.nn as nn
import torchvision
import carla
import csv
import yaml
import json
import argparse
import threading
from PIL import Image
from threading import Thread
# from queue import Queue
from PIL import Image, ImageDraw
from collections import deque
from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter
from team_code.general_base_agent import BaseAgent
from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner
from team_code.detectors.anomaly_detector import occlusion_detector, blur_detector
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent
from transfuser.model import TransFuser
from transfuser.config import GlobalConfig
from transfuser.data import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from controller.dm import Selector, Switcher
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights
from team_code.controller.Controller_Selector import Controller_Selector
from team_code.Source.Utils.BaseModelHandler import BaseModelHandler
from team_code.Source.Utils.ConfigParser import BaseConfigParser
from team_code.introspection.Introspective_LSTM import Introspective_LSTM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
#from multiprocessing import Process, Queue
from MCTS_random import MCTS
from team_code.mcts_scorer import read_data
#import multiprocessing as mp
from torch.multiprocessing import Process, Queue, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from itertools import *
import random
from model import VAE, vae_loss
from torchvision import transforms
from dataloaders import AVTDataset
from torch.utils.data import DataLoader
from scipy.integrate import quad
from pytorch.NNet import NNetWrapper as LBC_score_nn
from AP_pytorch.NNet import NNetWrapper as AP_score_nn
from collections import Counter

from pytorch_OOD.NNet import NNetWrapper as ood_nn
random.seed(40)

# from controller.lbc import Learning_by_cheating
from controller.autopilot import autopilot

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0] - 3, target_cam[1] - 3, target_cam[0] + 3, target_cam[1] + 3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

        _draw_rgb.ellipse((x - 2, y - 2, x + 2, y + 2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


# Function that gets in weather from a file
def process_weather_data(weather_file):
    weather = []
    lines = []
    with open(weather_file, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            weather.append(row)

    return weather

def read_occlusion(data_path):
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
            x.append(float(row[6]))
            X.append(x)
        #X = np.array(X)

    return X

def generate_failure(img, failure_type):
    if failure_type == "blur":
        img = cv2.blur(img, (10, 10))
    elif failure_type == "occlusion":
        h, w, _ = img.shape
        img = cv2.rectangle(img, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
    return img


def init_seg_model():
    root = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/leaderboard/team_code/"
    path = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/leaderboard/team_code/Config/"
    with open(path + "dataset.yaml", "r") as thisFile:
        dataset = yaml.load(thisFile, Loader=yaml.BaseLoader)
    with open(path + "model.yaml", "r") as thisFile:
        model = yaml.load(thisFile, Loader=yaml.BaseLoader)
    with open(path + "optimizer.yaml", "r") as thisFile:
        optimizer = yaml.load(thisFile, Loader=yaml.BaseLoader)

    datasets = list(dataset.keys())
    backbones = list(model["DeepLabV3"]["Backbone"].keys())
    optimizers = list(optimizer.keys())

    parser = argparse.ArgumentParser(description="Semantic Segmentation on DeepLabV3+")
    # Required arguments:
    parser.add_argument("--dataset", choices=datasets, default='CARLA', help="Choose dataset")
    parser.add_argument("--loc", default='./data', help="Specify dataset location")

    # Optional arguments:
    parser.add_argument(
        "--backbone", choices=backbones, default=backbones[0], help="Choose backbone"
    )
    parser.add_argument(
        "--optimizer", choices=optimizers, default=optimizers[0], help="Choose optimizer"
    )

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    with open(path + "{}.json".format(args.dataset), "r") as thisFile:
        dataset_json = json.load(thisFile)
        dataset["mean"] = dataset_json["mean"]
        dataset["std"] = dataset_json["std"]
        dataset["class"] = dataset_json["class"]

    config = BaseConfigParser(args, dataset, model, optimizer)
    seg_model = BaseModelHandler(config, root)

    mean = dataset["mean"]
    std = dataset["std"]

    return seg_model, mean, std


def create_bin(val):
    bin = -1
    if 0 <= int(val) < 25:
        bin = 0
    elif 25 <= int(val) < 50:
        bin = 1
    elif 50 <= int(val) < 75:
        bin = 2
    else:
        bin = 3

    return bin


def read_dataset(data_path, entry):
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

    match, weather, driving_score, safe_perf_scores = search(entry, X)
    match_sensor, weather_sensor, driving_score_sensor, safe_perf_scores_sensor = search_sensor(entry, X)
    return match, weather, driving_score, safe_perf_scores, match_sensor, weather_sensor, driving_score_sensor, safe_perf_scores_sensor


def read_velocity(data_path, entry):
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

    match = search_v(entry, X)
    return match


def search(entry, X_train):
    match = []
    weather = []
    driving_score = []
    safe_perf = []
    weather_sensor = []
    for ent in entry:
        match_1 = []
        weather_1 = []
        driving_score_1 = []
        safe_perf_1 = []
        weather_sensor_1 = []
        for val in X_train:
            if str(ent[0]) == str(val[0]) and str(ent[1]) == str(val[1]) and str(ent[2]) == str(val[2]) and str(
                    ent[3]) == str(val[3]) and (val[8] != '1') and (val[9] != '1') and (val[10] != '1'):
                match_1.append(val)
                weather_1.append([val[4], val[5], val[6], val[7]])
                weather_sensor_1.append([val[4], val[5], val[6], val[7], val[8], val[9], val[10]])
                driving_score_1.append([val[15], val[16]])
                safe_perf_1.append([val[11], val[12], val[13], val[14]])
        match.append(match_1)
        weather.append(weather_1)
        weather_sensor.append(weather_sensor_1)
        driving_score.append(driving_score_1)
        safe_perf.append(safe_perf_1)

    return match, weather, driving_score, safe_perf


def search_sensor(entry, X_train):
    match = []
    weather = []
    driving_score = []
    safe_perf = []
    weather_sensor = []
    for ent in entry:
        match_1 = []
        weather_1 = []
        driving_score_1 = []
        safe_perf_1 = []
        for val in X_train:
            #(val)
            if str(ent[0]) == str(val[0]) and str(ent[1]) == str(val[1]) and str(ent[2]) == str(val[2]) and str(
                    ent[3]) == str(val[3]):
                if (str(val[8]) != "0") or (str(val[9]) != "0") or (str(val[10]) != "0"):
                    match_1.append(val)
                    weather_1.append([val[4], val[5], val[6], val[7]])
                    driving_score_1.append([val[15], val[16]])
                    safe_perf_1.append([val[11], val[12], val[13], val[14]])
        match.append(match_1)
        weather.append(weather_1)
        driving_score.append(driving_score_1)
        safe_perf.append(safe_perf_1)

    return match, weather, driving_score, safe_perf


def search_v(entry, X_train):
    match = []
    weather = []
    driving_score = []
    safe_perf = []
    for ent in entry:
        match_1 = []
        weather_1 = []
        safe_perf_1 = []
        for val in X_train:
            if str(ent[0]) == str(val[0]) and str(ent[1]) == str(val[1]) and str(ent[2]) == str(val[2]) and str(
                    ent[3]) == str(val[3]):
                match_1.append(val)
                weather_1.append([val[4], val[5], val[6], val[7]])
                safe_perf_1.append([val[11], val[12], val[13], val[14]])
        match.append(match_1)
        weather.append(weather_1)
        safe_perf.append(safe_perf_1)

    return match


def get_driving_score(k, lut_weather, lut_driving_score, lut_safe_perf_scores, X_test, knn, weather):
    lbc_ds = []
    ap_ds = []
    lbc_col = []
    ap_col = []
    lbc_sp = []
    ap_sp = []
    curr_weather = [weather[0], weather[1], weather[2], weather[3]]
    for count, val in enumerate(lut_weather[k // 2]):
        if (weather[0] == val[0]) and (weather[1] == val[1]) and (weather[2] == val[2]) and (weather[3] == val[3]):
            lbc_ds.append(float(lut_driving_score[k // 2][count][0]))
            ap_ds.append(float(lut_driving_score[k // 2][count][1]))
            lbc_sp.append(float(lut_safe_perf_scores[k // 2][count][0]))
            lbc_col.append(float(lut_safe_perf_scores[k // 2][count][1]))
            ap_sp.append(float(lut_safe_perf_scores[k // 2][count][2]))
            ap_col.append(float(lut_safe_perf_scores[k // 2][count][3]))
        else:
            # print("1")

            knn.fit(lut_weather[k // 2], lut_driving_score[k // 2])
            # print("2")
            curr = np.array(np.array(curr_weather).reshape(1, -1))
            # print("3")
            neighbors = knn.kneighbors(curr, return_distance=False)
            # print("4")
            for val1 in neighbors[0]:
                # print(val1)
                lbc_ds.append(float(lut_driving_score[k // 2][val1][0]))
                ap_ds.append(float(lut_driving_score[k // 2][val1][1]))
                lbc_sp.append(float(lut_safe_perf_scores[k // 2][val1][0]))
                lbc_col.append(float(lut_safe_perf_scores[k // 2][val1][1]))
                ap_sp.append(float(lut_safe_perf_scores[k // 2][val1][2]))
                ap_col.append(float(lut_safe_perf_scores[k // 2][val1][3]))
    # print(lbc_score)
    # print(ap_score)

    return round(sum(lbc_ds) / len(lbc_ds), 2), round(sum(ap_ds) / len(ap_ds), 2), round(sum(lbc_sp) / len(lbc_sp),
                                                                                         2), round(
        sum(lbc_col) / len(lbc_col), 2), round(sum(ap_sp) / len(ap_sp), 2), round(sum(ap_col) / len(ap_col), 2)


def predict(k, lut_weather, lut_driving_score, lut_safe_perf_scores, X_test, knn, weather):
    Y_pred = []
    # print(X_train)
    # X_subset, X_weather, X_driving = search(X_test,X_train)
    # print(X_subset)
    lbc_ds, ap_ds, lbc_sp, lbc_col, ap_sp, ap_col = get_driving_score(k, lut_weather, lut_driving_score,
                                                                      lut_safe_perf_scores, X_test, knn, weather)
    # print(lbc_score,ap_score)

    return lbc_ds, ap_ds, str(lbc_sp), str(lbc_col), str(ap_sp), str(ap_col)

def mcts_planner(current_scene, semantic_data, failure_type, distance, mcts_queue):
    """
    MCTS Planner
    """
    print("I am running")
    print("inside time:", time.time())
    alpha_args = dotdict({
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'numMCTSSims': 500,
        'cpuct': math.sqrt(2),
    })
    print(current_scene, semantic_data, failure_type, distance)
    #mcts = MCTS(alpha_args, semantic_data, 3, failure_type, LBC_performance, LBC_safe, AP_scorenn, OOD_time_estimation, time_interval)
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


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class ImageAgent(BaseAgent):
    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file, data_folder, project_path, route_folder, tracks_num,failure_type,configuration):
        super().setup(path_to_conf_file, data_folder, project_path, route_folder, tracks_num,failure_type,configuration)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()
        if configuration == "AP":
            self.controller = "Autopilot"
            self.selected_controller = "Autopilot"
        else:
            self.controller = "LBC"
            self.selected_controller = "LBC"
        self.ckpt = project_path + "/trained_monitors/controller_selector_best.pth"
        self.scene_scorer_train_dataset = "encoded_prior_belief.csv"
        self.train_data_path = project_path + "/trained_monitors/prior_data/mcts_lut.csv"
        self.mcts_lut = project_path + "/trained_monitors/prior_data/mcts_lut.csv"
        self.path_lut = route_folder + "label.csv"
        self.model = Controller_Selector(2, 128)
        self.device = 'cuda:0'  # get_device()
        self.model.eval()
        self.sliding_window = 60
        self.failure_ckpt = project_path + "/trained_monitors/state_lstm_best.pth"
        self.occlusion_path = project_path + "/trained_monitors/prior_data/occlusion.csv"
        self.failure_model = Introspective_LSTM(2, 64, self.sliding_window)
        self.failure_model.to(self.device)
        self.failure_model.eval()
        self.failure_model_checkpoint = torch.load(self.failure_ckpt, map_location=torch.device('cpu'))
        self.failure_model.load_state_dict(self.failure_model_checkpoint["model_state"])
        self.failure_model = nn.DataParallel(self.failure_model)
        self.failure_model.to(self.device)
        self.failure_cur_itrs = self.failure_model_checkpoint["cur_itrs"]
        self.failure_best_score = self.failure_model_checkpoint["best_score"]
        # Load weights
        self.checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.cur_itrs = self.checkpoint["cur_itrs"]
        self.best_score = self.checkpoint['best_score']
        self.data_folder = data_folder + "temp.csv"
        self.step = -1
        self.blur_queue = Queue(maxsize=1)
        self.occlusion_queue = Queue(maxsize=1)
        self.segmentation_queue = Queue(maxsize=1)
        self.failure_prediction_queue = Queue(maxsize=1)

        self.state = []
        self.monitors = []
        self.weather = []
        self.my_weather = []
        self.state_data = []
        self.collision_likelihood_window = []
        self.collision_likelihood = 0.0
        # self.base_weather = []
        self.blur = [0, 0, 0]
        self.occlusion = [0, 0, 0]
        self.outlier_score = 0.0
        self.failure_prediction_score = 0.0
        self.classifier_score = 0.0
        self.combined_score = 0.0
        self.wait = 0
        self.dm = []
        self.lbc = []
        self.ap = []
        self.seg = []
        # self.x = 0
        # self.y = 0
        self.k = 0
        self.val = 1
        self.weather_file = route_folder + "changing_weather.csv"
        # print(self.weather_file)
        self.tracks_num = tracks_num
        self.semantic_data = []
        self.time = -1
        self.switch = 0
        self.current_scene = []
        self.action_set = [0, 1]
        self.decision_query = 1
        self.policy = []
        self.cam_failure = ['0', '0', '0']
        self.lbc_sp = ""
        self.lbc_col = ""
        self.ap_sp = ""
        self.ap_col = ""
        # self.planning_decision = 0
        self.scene_flag = False
        self.planner_start = False
        self.failure_type = failure_type
        self.configuration = configuration
        self.ood_count = 0
        self.current_traffic_density = 0
        self.ood_flag = False
        self.ood_continue = True
        interpolation_mode = Image.BICUBIC
        self.transfs =transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        self.traffic_level = 0
        saved_net_state = torch.load('VAE_Interpolation_1000_3.pth')
        print(f'Number of epochs we trained this network for = {saved_net_state["epoch"]}')
        #test_dataset = AVTDataset(path='CARLA_test', subsample=1, transforms=transfs)
        #test_loader = DataLoader(test_dataset, batch_size=32)

        device = torch.device("cuda:0")

        self.VAE = VAE().to(device)
        self.VAE.load_state_dict(saved_net_state['model_state_dict'])
        self.VAE.training = False
        self.VAE.eval()
        interpolation_mode = Image.BICUBIC
        trans = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((128, 128), interpolation=interpolation_mode)])
        test_dataset = AVTDataset(path='/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/calibration', subsample=1, transforms=trans)
        test_loader = DataLoader(test_dataset, batch_size=128)
        images, labels = iter(test_loader).next()
        self.all_loss = []
        #self.oodnn = ood_nn(9, 1)
        self.LBC_performance = LBC_score_nn(10, 1)
        self.LBC_safe = LBC_score_nn(10, 1)
        self.AP_scorenn = AP_score_nn(9, 1)
        self.OOD_time_estimation = ood_nn(9, 1)
        self.time_interval = ood_nn(9, 1)
        self.OOD_time_estimation.load_checkpoint(folder="OOD_time", filename='temp.pth.tar')
        self.time_interval.load_checkpoint(folder="interval_estimation", filename='temp.pth.tar')
        self.LBC_performance.load_checkpoint(folder="LBC_performance_score", filename='temp.pth.tar')
        self.LBC_safe.load_checkpoint(folder="LBC_safe_score", filename='temp.pth.tar')
        self.AP_scorenn.load_checkpoint(folder="AP_score_estimation", filename='temp.pth.tar')
        self.processes = []
        self.num_processes = 1
        for img in images:
            with torch.no_grad():
                img = img.to(device)
                re_images, latent_mu, latent_logvar = self.VAE(img.view(1, 3, 128, 128))
                loss = vae_loss(re_images, img.view(1, 3, 128, 128), latent_mu, latent_logvar)
                self.all_loss.append(loss)

        print("--------------------------Initializing Files-------------------------------------")

        print("Training state for the failure prediction model restored from %s" % self.failure_ckpt)
        print("Failure Prediction Model restored from %s" % self.failure_ckpt)

        del self.checkpoint  # free memory
        del self.failure_model_checkpoint  # free memory

    def _init(self):
        # super()._init()
        self.mcts_queue = Queue()
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._command_planner = RoutePlanner(7.5, 25.0, 257, self._global_wp, self.scene_gps, self.trajectory_wp)
        self._command_planner.set_route(self._global_plan, True)
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2, radius=1.0)
        self.weather = process_weather_data(self.weather_file)
        self.velocity = 0
        with open(self.path_lut, newline='') as csvfile:
            self.semantic_data = list(csv.reader(csvfile))
        # self.prior_scene_scores = read_dataset(self.train_data_path)
        self.matching_lut_entries, self.matching_lut_weather, self.matching_lut_driving_score, self.matching_lut_safe_perf_scores, self.sensor_entry, self.sensor_weather, self.sensor_driving_score, self.sensor_safe_perf_scores = read_dataset(
            self.train_data_path, self.semantic_data)
        self.weather_flag = False
        self.weather_trajectory = []
        actions = [0, 1, 2]
        action_set = list(product(actions, repeat=3))
        self.random_space = action_set
        self.process_scene = []
        # print(self.matching_lut_entries[0])
        # self.mcts_scores = mcts_scorer.read_data(self.mcts_lut)
        self.change_count = 0

        velocity_path = "/home/baiting/Desktop/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA/trained_monitors/mcts_lut.csv"
        self.occlusion_table = read_occlusion(self.occlusion_path)
        self.matching_lut_entries_v = read_velocity(velocity_path, self.semantic_data)
        self.initialized = True
        self.sensor_flag = False
        # for stop signs
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False  # if the ego vehicle is influenced by a stop sign
        self.occlusion_likelihood = -1
        self.sensor_failure_happened = False
        f = open("scene_num.txt", "r")
        lines = f.readlines()
        self.seed = int(lines[0])
        self.scene_recorder = -1
        if int(self.failure_type) == 0:
            self.t_sensor_failure = 99999999999
        else:
            random.seed(int(lines[0]))
            #self.t_sensor_failure = random.randint(100, 2000)
            self.t_sensor_failure = 0
        #self.t_sensor_failure = random.randint(100, 2000)
        self.seg_model, self.mean, self.std = init_seg_model()
        self.PlannerProcess = Process(target=self.mcts_planner)
        self.scene_event = False
        self.distance_to_location = []
        self.sensor_flag_continue = True
        self.traffic_level_flag = False
        self.traffic_time_count = 0
        print("----------------------------Done Initialization-------------------------------------")

    # ------------------------------- Runtime Monitors ------------------------------------------------

    def blur_detection(self, result):
        self.blur = []
        fm1, rgb_blur = blur_detector(result['rgb'], threshold=20)
        fm2, rgb_left_blur = blur_detector(result['rgb_left'], threshold=20)
        fm3, rgb_right_blur = blur_detector(result['rgb_right'], threshold=20)
        self.blur.append(rgb_blur)
        self.blur.append(rgb_left_blur)
        self.blur.append(rgb_right_blur)
        # print(self.blur)
        self.blur_queue.put(self.blur)
        # print("done computing")

    def occlusion_detection(self, result):
        self.occlusion = []
        percent1, rgb_occluded = occlusion_detector(result['rgb'], threshold=25)
        percent2, rgb_left_occluded = occlusion_detector(result['rgb_left'], threshold=25)
        percent3, rgb_right_occluded = occlusion_detector(result['rgb_right'], threshold=25)
        self.occlusion.append(rgb_occluded)
        self.occlusion.append(rgb_left_occluded)
        self.occlusion.append(rgb_right_occluded)
        self.occlusion_queue.put(self.occlusion)

    def create_input_data(self, bin_weather):
        """
        Process input data and prepare it for the classifier
        """
        data = []
        our_weather = self.my_weather

        data = [float(i)/10 for i in self.semantic_data[self.k // 2]]

        # for weather in our_weather:
        #    data.append(create_bin(int(weather)))
        for weather in our_weather[:-1]:
            data.append(float(weather)/100)
            # data.append(int(weather))

        return data

    def Scene_Scorer(self, tick_data):
        """
        Scene_Scorer:
        Input: semantic labels, traffic density, weather
        Output: Driving Score of the two controllers
        """
        x = time.time()
        lbc_ds, ap_ds, lbc_sp, lbc_col, ap_sp, ap_col = predict(self.k, self.matching_lut_weather,
                                                                self.matching_lut_driving_score,
                                                                self.matching_lut_safe_perf_scores, self.current_scene,
                                                                self.knn, self.my_weather)
        print(time.time() - x)

        return lbc_ds, ap_ds, lbc_sp, lbc_col, ap_sp, ap_col


    def sensor_score(self, tick_data):
        """
        Scene_Scorer:
        Input: semantic labels, traffic density, weather
        Output: Driving Score of the two controllers
        """
        x = time.time()
        # current_scene = self.create_input_data(bin_weather=1)
        lbc_ds, ap_ds, lbc_sp, lbc_col, ap_sp, ap_col = predict(self.k, self.sensor_weather, self.sensor_driving_score,
                                                                self.sensor_safe_perf_scores, self.current_scene,
                                                                self.knn, self.my_weather)
        print(time.time() - x)

        return lbc_ds, ap_ds, lbc_sp, lbc_col, ap_sp, ap_col

    def Segmentation_Outlier_Detector(self, result):
        """
        Outlier Detector based on segmentation network
        Input: Camera Images
        Target: Outlier Score
        """
        image = result['rgb']
        image = np.array(image) / 255
        image = (image - self.mean) / self.std

        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image, 0)

        self.outlier_score = self.seg_model.getEvaluation(image)

        self.segmentation_queue.put(1 - np.mean(self.outlier_score))

    def predict_failure(self, result):
        # print(self.state_data)
        # Normalize data with values from training set
        state_avg = [2.975678, 0.005583, 0.098343]
        state_max = [7.285361, 0.469301, 1]
        data_normalized = []
        for i, row in enumerate(self.state_data):
            for s in range(3):  # 3 states
                for k in range(int(len(row) / 3)):
                    row[3 * k + s] = (float(row[3 * k + s]) - state_avg[s]) / state_max[s]
                    row[3 * k + s] = 0.001 * np.round(1000 * row[3 * k + s])
            data_normalized.append(row)

        softmax = nn.Softmax(dim=0)
        data = torch.from_numpy(np.array(data_normalized)).float().unsqueeze(0)

        with torch.no_grad():
            data = data.to(self.device)
            output_data = self.failure_model(data)
            probability = softmax(output_data[0]).cpu().numpy()

        self.failure_prediction_score = np.round(1000 * probability[1]) * 0.001
        self.failure_prediction_queue.put(self.failure_prediction_score)


    def mcts_planner(self):
        """
        MCTS Planner
        """
        print("I am running inside MCTS")
        alpha_args = dotdict({
            'tempThreshold': 15,
            'updateThreshold': 0.6,
            'numMCTSSims': 500,
            'cpuct': math.sqrt(2),
        })

        mcts = MCTS(alpha_args, self.semantic_data[self.k // 2:5], 3, self.failure_type, self.LBC_performance, self.LBC_safe, self.AP_scorenn, self.OOD_time_estimation, self.time_interval)
        temp = 1
        #print(self.current_scene)
        scene_for_tree = self.current_scene.copy()
        scene_for_tree[8] = self.current_traffic_density
        print(scene_for_tree)
        print(self.semantic_data[self.k // 2:5])
        print(self.distance_to_location)
        self.policy = mcts.getActionProb(scene_for_tree, self.semantic_data[self.k // 2:5], self.action_set, self.distance_to_location, temp=temp)
        print("############################################")
        print("current scene:", self.current_scene)
        print("current semantic data:", self.semantic_data[self.k // 2:5])
        print("policy", self.policy)
        print("############################################")
        ##print("policy",self.policy)
        # self.mcts_queue.queue.clear()
        #self.mcts_queue.put(self.policy)
        return self.policy

    # ------------------------------Controller Code---------------------------------------------------------------

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))  # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None

    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane

            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_red_light_present = 1 if light is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        self.is_stop_sign_present = 1 if stop_sign is not None else 0

        return any(x is not None for x in [vehicle, light, walker, stop_sign])

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker
        return None

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0] + v[0], p[1] + v[1], z)
        color = carla.Color(*color)
        # self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route, scene_gps, trajectory_wp):
        super().set_global_plan(global_plan_gps, global_plan_world_coord, wp_route, scene_gps, trajectory_wp)
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def write_data(self, steer, throttle, brake, pos, speed, theta):

        stats = []
        stats.append(self.step)
        stats.append(float(self.step / 20))
        stats.append(speed)
        stats.append(steer)
        stats.append(throttle)
        stats.append(brake)
        # stats.append(self.angle)
        stats.append(pos[0])
        stats.append(pos[1])
        stats.append(theta)
        stats.append(self.current_traffic_density)
        stats.append(self.ood_count)
        with open(self.data_folder, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(stats)

    def Learning_by_cheating(self, tick_data, pos, theta, speed):
        """
        Learning By Cheating controller code
        """
        with torch.no_grad():
            img = torchvision.transforms.functional.to_tensor(tick_data['image'])
            img = img[None].cuda()

            target = torch.from_numpy(tick_data['target'])
            target = target[None].cuda()

            points, (target_cam, _) = self.net.forward(img, target)
            points_cam = points.clone().cpu()
            points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
            points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
            points_cam = points_cam.squeeze()
            points_world = self.converter.cam_to_world(points_cam).numpy()

            aim = (points_world[1] + points_world[0]) / 2.0
            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
            steer = self._turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)

        #desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0 + 0.4
        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0 + 0.4
        # desired_speed *= (1 - abs(angle)) ** 2DecisionManagerThread.start()

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        return steer, throttle, brake, desired_speed

    def autopilot(self, target, far_target, tick_data, theta, speed):
        # print("AP")
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        topdown = draw_traffic_lights(tick_data['topdown'], self._vehicle, self._traffic_lights)

        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 3.0 if should_slow else 4.0
        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.4)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed

    def prepare_current_scene_data(self):

        self.current_scene = self.create_input_data(bin_weather=1)
        # lbc_perf,lbc_safe,ap_perf,lbc_perf = self.get_scores()
        # print(self.current_scene)
        self.current_scene.append(int(self.cam_failure[0]))
        #self.current_scene.append(self.cam_failure[1])
        self.current_scene.append(self.traffic_level)
        self.current_scene.append(self.ood_count)
        #self.current_scene.append(self.cam_failure[2])
        #self.current_scene.append(self.lbc_sp)
        #self.current_scene.append(self.lbc_col)
        #self.current_scene.append(self.ap_sp)
        #self.current_scene.append(self.ap_col)
        self.current_scene.append(0)
        self.current_scene.append(0)
        self.current_scene.append(0)
        self.current_scene.append(0)
        # print(self.current_scene)

        return self.current_scene

    def create_threads(self, result):
        OccusionDetectorThread = Thread(target=self.occlusion_detection, args=(result,))
        BlurDetectorThread = Thread(target=self.blur_detection, args=(result,))
        SegmentationModelThread = Thread(target=self.Segmentation_Outlier_Detector, args=(result,))
        FailurePredictionProcess = Thread(target=self.predict_failure, args=(result,))
        FailurePredictionProcess.daemon = True
        BlurDetectorThread.daemon = True
        OccusionDetectorThread.daemon = True
        SegmentationModelThread.daemon = True

        return OccusionDetectorThread, BlurDetectorThread, SegmentationModelThread, FailurePredictionProcess

    def get_control_action(self, tick_data, pos, theta, speed, near_node, far_node):
        """
        Get control actions from the selected controller
        """
        # print(self.controller)
        if self.controller == "LBC":
            steer, throttle, brake, desired_speed = self.Learning_by_cheating(tick_data, pos, theta, speed)
        elif self.controller == "Autopilot":
            steer, throttle, brake, desired_speed = self.autopilot(near_node, far_node, tick_data, theta, speed)

        return steer, throttle, brake, desired_speed

    def change_weather(self):
        """
        Change weather conditions
        """
        self.change_count += 1
        f = open("scene_num.txt", "r")
        lines = f.readlines()
        if int(self.my_weather[3]) >= 90:
            self.my_weather[3] = 90
        else:
            self.my_weather[3] = int(self.my_weather[3])

        random.seed(int(lines[0]) * self.change_count)
        sampling_change = random.choice(self.random_space)
        for i in range(len(self.my_weather) - 1):
            if sampling_change[i] == 1:
                if int(self.my_weather[i]) < 100:
                    self.my_weather[i] = int(self.my_weather[i]) + 5
            elif sampling_change[i] == 2:
                if int(self.my_weather[i]) >= 5:
                    self.my_weather[i] = int(self.my_weather[i]) - 5
            else:
                self.my_weather[i] = int(self.my_weather[i])

        self.weather = [self.my_weather]
        # self.weather_trajectory.append(self.my_weather)
        with open('weather_trajectory.txt', 'a') as f:
            f.write(" ".join([str(i) for i in self.my_weather]))
            f.write("\n")
        with open('transition_record.txt', 'a') as f:
            f.write("\n")
            f.write("weather")
        print("-------------------------------")
        print("\n")
        print("Weather Change", self.my_weather)
        print("Location", self.semantic_data[self.k // 2])
        print("\n")
        print("-------------------------------")
        self.my_weather = [float(i) for i in self.my_weather]
        carla_weather = carla.WeatherParameters(cloudiness=self.my_weather[0], precipitation=self.my_weather[1],
                                                precipitation_deposits=self.my_weather[2],
                                                sun_altitude_angle=self.my_weather[3])
        #carla_weather = carla.WeatherParameters(cloudiness=self.my_weather[0], precipitation=self.my_weather[1],
        #                                        precipitation_deposits=self.my_weather[2])
        self._world.set_weather(carla_weather)

    # --------------------------Simulation Tick-----------------------------------------------------
    def prepare_tree(self):
        return self.current_scene, self.semantic_data[self.k // 2:5], self.failure_type, self.distance_to_location, self.LBC_performance, self.LBC_safe, self.AP_scorenn, self.mcts_queue


    def tick(self, input_data):
        #print("&&&&&&&&7image_size:", rgb.shape)
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        #print("&&&&&&&&7image_size:", rgb.shape)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        topdown = input_data['map'][1][:, :, 2]


        result = {
            'rgb': rgb,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'topdown': topdown
            # 'rgb_rear': rgb_rear,
            # 'lidar': lidar,
        }
        if self.failure_type == 1:
            if self.step >= self.t_sensor_failure:
                self.sensor_failure_happened = True
                rgb = generate_failure(rgb, 'occlusion')
                result['rgb'] = rgb
        elif self.failure_type == 2:
            if len(self.current_scene) != 0:
                for array in self.occlusion_table:
                    #print(len(array),self.current_scene)
                    if (int(array[0]) == int(self.current_scene[0])) and (int(array[1]) == int(self.current_scene[1])) and (int(array[2]) == int(self.current_scene[2])) and (int(array[3]) == int(self.current_scene[3])) and (int(array[4]) == int(self.current_scene[4])) and (int(array[5]) == int(self.current_scene[5])):
                        self.occlusion_likelihood = float(array[6])
                if self.scene_recorder == -1:
                    self.scene_recorder = self.current_scene
                if self.scene_recorder != self.current_scene:
                    self.seed += 1
                    self.scene_recorder = self.current_scene
                #we use random seed to increase reproducibility
                random.seed(self.seed)
                chance = random.uniform(0,1)
                if chance < self.occlusion_likelihood:
                    self.sensor_failure_happened = True
                    rgb = generate_failure(rgb, 'occlusion')
                    result['rgb'] = rgb
                else:
                    self.sensor_failure_happened = False

        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)





        OccusionDetectorThread, BlurDetectorThread, SegmentationModelThread, FailurePredictionProcess = self.create_threads(
            result)

        self.step += 1
        waypoints_left, current_waypoint_global, self.k, total_waypoints = self._command_planner.get_waypoints_remaining(
            gps)
        accumulative_distance = 0
        self.distance_to_location = []

        for i in [2, 4, 6, 8, 10]:
            if self.k < i:
                accumulative_distance += self._vehicle.get_transform().location.distance(self.trajectory_wp[self.k + 1])
                for j in range(self.k + 1, 10):
                    if j % 2 == 0:
                        self.distance_to_location.append(accumulative_distance)

                    accumulative_distance += self.trajectory_wp[j].distance(self.trajectory_wp[j - 1])
                break




        #if conditions for detecting changes in scenes
        if self.k % 2 != 0:
            self.val = 1

        if self.k % 2 == 0 and self.val == 1 and self.step > 5:
            with open('transition_record.txt', 'a') as f:
                f.write("\n")
                f.write("location")
            self.lbc_prior_belief, self.ap_prior_belief, self.lbc_sp, self.lbc_col, self.ap_sp, self.ap_col = self.Scene_Scorer(
                result)
            self.current_scene = self.prepare_current_scene_data()
            #print(self.current_scene)
            if len(self.process_scene) == 0:
                self.process_scene = self.current_scene

            self.val = 0
        # change weather each 20s in game time, current simualtion frequency 20Hz.
        if (self.step % 400) == 0 and self.step > 5:
            self.change_weather()
            self.weather_flag = True
            self.current_scene = self.prepare_current_scene_data()
            #print(self.current_scene)
            if len(self.process_scene) == 0:
                self.process_scene = self.current_scene
        if self.step > 60:
            FailurePredictionProcess.start()
        if self.step != 0 and self.step % 18 == 0:
            BlurDetectorThread.start()
            OccusionDetectorThread.start()
            SegmentationModelThread.start()
        if self.step % 20 == 0:
            p_v = 1
            for i in range(10):
                with torch.no_grad():
                    VAE_in = self.transfs(rgb)
                    images = VAE_in.to(self.device)
                    # print(images.shape)
                    re_images, latent_mu, latent_logvar = self.VAE(images.view(1, 3, 128, 128))
                    loss = vae_loss(re_images, images.view(1, 3, 128, 128), latent_mu, latent_logvar)
                    temp = len([v for v in self.all_loss if v > loss])
                    if temp == 0:
                        temp += 1
                    temp_p = temp/len(self.all_loss)

                    p_v = p_v * temp_p
            I_quad, est_err_quad = \
                quad(lambda x: x ** 10 * p_v ** (x - 1), 0, 1)
            print("ood values:", I_quad)

            if I_quad > 10000:
                self.ood_count = 1

            else:
                self.ood_count = 0
            self.current_scene = self.prepare_current_scene_data()


        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        gps = self._get_position(result)
        far_node, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target

        if self.step > 60:
            FailurePredictionProcess.join()
            self.failure_prediction_score = self.failure_prediction_queue.get()

        self.collision_likelihood_window.append(self.failure_prediction_score)

        if len(self.collision_likelihood_window) >= self.sliding_window:
            self.collision_likelihood_window = self.collision_likelihood_window[-1 * self.sliding_window:]

        self.collision_likelihood = round(sum(self.collision_likelihood_window) / self.sliding_window, 2)

        return result, far_node, theta, speed, OccusionDetectorThread, BlurDetectorThread, SegmentationModelThread

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
            # print(self.weather)
            self.my_weather = self.weather[0]
            with open('weather_trajectory.txt', 'a') as f:
                f.write(" ".join(self.my_weather))
                f.write("\n")
        data = []

        tick_data, far_node, theta, speed, OccusionDetectorThread, BlurDetectorThread, SegmentationModelThread = self.tick(
            input_data)
        pos = self._get_position(tick_data)
        near_node, near_command = self._waypoint_planner.run_step(pos)
        print(self.configuration)
        if (self.k % 2) == 0 and (self.scene_flag == False):
            self.scene_event = True
            self.scene_flag = True
        elif (self.k % 2) == 1 and self.scene_flag == True:
            self.scene_flag = False
        if ((sum(self.blur) >= 1) or (sum(self.occlusion) >= 1)) and self.sensor_flag_continue:
            self.sensor_flag = True
            self.sensor_flag_continue = False
        if self.sensor_flag_continue == False and ((sum(self.blur) == 0) or (sum(self.occlusion) == 0)):
            self.sensor_flag_continue = True
        if (self.ood_count == 1 and self.ood_continue):
            self.ood_flag = True
            self.ood_continue = False
        if (self.ood_count == 0 and self.ood_continue == False):
            self.ood_continue = True
        #print(self.current_scene)

        traffic_number = []
        carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')

        for vehicle in carla_vehicles:
            distance = self._vehicle.get_transform().location.distance(CarlaDataProvider.get_world().get_snapshot().find(vehicle.id).get_transform().location)
            if distance <= 30:
                traffic_number.append(distance)
        self.current_traffic_density = self.current_traffic_density * self.traffic_time_count + len(traffic_number)/10.0
        self.traffic_time_count += 1
        #print(self.current_traffic_density, self.traffic_time_count)
        self.current_traffic_density /= self.traffic_time_count

        traffic_check = self.traffic_level
        temp = round(self.current_traffic_density,1)
        if temp < 0.2:
            self.traffic_level = 0
        elif temp < 0.3:
            self.traffic_level = 1
        elif temp < 0.4:
            self.traffic_level = 2
        elif temp < 0.5:
            self.traffic_level = 3
        elif temp < 0.6:
            self.traffic_level = 4
        elif temp < 0.7:
            self.traffic_level = 5
        elif temp < 0.8:
            self.traffic_level = 6
        elif temp < 0.9:
            self.traffic_level = 7
        else:
            self.traffic_level = 8
        print("density",self.current_traffic_density)
        print(traffic_check,self.traffic_level)
        if traffic_check != self.traffic_level:
            self.traffic_level_flag = True
        print("traffic density:", len(traffic_number))

        if (self.scene_event or self.weather_flag or self.sensor_flag or self.ood_flag or self.traffic_level_flag):
            self.current_traffic_density = len(traffic_number) / 10.0
            self.traffic_time_count = 1
        print(self.scene_event,self.weather_flag,self.sensor_flag,self.ood_flag,self.traffic_level_flag)
        if (self.scene_event or self.weather_flag or self.sensor_flag or self.ood_flag or self.traffic_level_flag) and self.step > 100 and self.controller == "LBC":
            # if (self.k % 2 == 0) and self.step > 9999:
            if (self.step %10 == 0 and self.step != 0 and self.wait == 0):
                self.lbc_prior_belief, self.ap_prior_belief, self.lbc_sp, self.lbc_col, self.ap_sp, self.ap_col = self.Scene_Scorer(
                    tick_data)
                #retrieve the data from cluustered with only sensor failures
                if self.sensor_failure_happened:
                    self.lbc_prior_belief, self.ap_prior_belief, self.lbc_sp, self.lbc_col, self.ap_sp, self.ap_col = self.sensor_score(
                        tick_data)

                #self.lbc_prior_belief = float(self.lbc_sp) - float(self.lbc_col)
                #self.ap_prior_belief = float(self.ap_sp) - float(self.ap_col)
                # print(self.AP_scorenn.predict([0, 0, 0, 0.2, 0.1, 0, 0.6, 0, 1]))
                # print(self.LBC_safe.predict([0, 0, 0, 0.2, 0.1, 0, 0.6, 0, 0.1, 0.1]))
                # print(self.LBC_performance.predict([0, 0, 0, 0.2, 0.1, 0, 0.6, 0, 0.1, 0.1]))
                ap_prediction_state = self.current_scene[:9].copy()
                lbc_prediction_state = self.current_scene[:10].copy()
                lbc_prediction_state[8] = self.current_traffic_density
                ap_prediction_state[8] = self.current_traffic_density
                ap_performance, ap_safety = self.AP_scorenn.predict(ap_prediction_state)
                lbc_safety =self.LBC_safe.predict(lbc_prediction_state)
                lbc_performance = self.LBC_performance.predict(lbc_prediction_state)
                print(ap_performance, ap_safety)
                print(lbc_prediction_state,lbc_safety,lbc_performance)
                #self.lbc_prior_belief = 0.5*lbc_performance - torch.round(lbc_safety)
                #self.ap_prior_belief = 0.5*ap_performance - torch.round(ap_safety)
                self.lbc_prior_belief = lbc_performance - lbc_safety
                self.ap_prior_belief = ap_performance - ap_safety
                selector = Selector(self.failure_prediction_score, self.outlier_score, self.blur, self.occlusion,
                                    self.lbc_prior_belief, self.ap_prior_belief, speed,
                                    self.controller, self.configuration)  # Decision Manager
                if lbc_safety < 0.2:
                    selector = "same"
                if selector != "same":
                    self.selected_controller = selector
                else:
                    self.selected_controller = self.controller
                if self.configuration == "AP":
                    self.selected_controller = "Autopilot"
                elif self.configuration == "LBC":
                    self.selected_controller = "LBC"
                self.weather_flag_foward = False
                self.switch = Switcher(speed, self.controller, self.selected_controller)
                self.val = 0
                self.weather_flag = False
                self.scene_event = False
                self.sensor_flag = False
                self.traffic_level_flag = False
                self.ood_flag = False
        print(self.controller)
        if self.switch == 0:
            print("No switching required")
            self.controller = self.controller
                #self.controller = "Autopilot"
            steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
            self.wait = 0
        #print(timestamp)
        if self.switch == 1:
            print("Reverse switch control transition phase")
            self.wait = 1
            if speed < 10:
                print("Control Given to LBC")
                self.controller = "LBC"
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
                self.switch = 0
                self.wait = 0
                with open('switch_record.txt', 'a') as f:
                    f.write("switch to LBC:")
                    f.write(str(timestamp))
                    f.write("\n")
                with open('transition_record.txt', 'a') as f:
                    f.write("to LBC")
            else:
                #print("Controller Transition Phase During Reverse Switching")
                self.controller = "Autopilot"
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
                if brake !=1:
                    throttle = 0.15

        elif self.switch == 3:
            self.wait = 1
            print("Forward switch required")
            if speed < 10:
                print("Control Given to Autopilot")
                self.controller = "Autopilot"
                with open('switch_record.txt', 'a') as f:
                    f.write("switch to autopilot:")
                    f.write(str(timestamp))
                    f.write("\n")
                with open('transition_record.txt', 'a') as f:
                    f.write("to autopilot")
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
                self.switch = 0
                self.wait = 0
            else:
                print("Controller Transition Phase During Forward Switching")
                self.controller = "LBC"
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
                if brake !=1:
                    throttle = 0.15

        #elif self.controller == "Autopilot" and self.scene_flag != self.k and ((self.k%2 == 0) or (float(self.collision_likelihood) > 0.5) or (sum(self.blur) >=1) or (sum(self.occlusion)>=1) or self.weather_flag):
        elif self.controller == "Autopilot" and (self.scene_event or self.sensor_flag or self.weather_flag or self.ood_flag or self.traffic_level_flag) and self.configuration == "DS":
                    # and (self.lbc_prior_belief >= self.ap_prior_belief)
            print("Reverse switch tested")
            #self.time +=1
            print(self.time)
            self.controller = "Autopilot"
            steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
            if self.process_scene[:11] != self.current_scene[:11]:
                for p in self.processes:
                    if p.is_alive():
                        p.terminate()
                self.planner_start = False
                self.time = 0
                    #self.PlannerProcess.terminate()
                    #self.planner_start = False
                    #self.time = 0
            if self.step !=0 and self.planner_start == False and self.step % 20 == 0: # and self.step >20:
                print("I am starting the planner")
                self.x = self.time
                current_scene = self.current_scene
                semantic_data = self.semantic_data[self.k // 2:5]
                failure_type = self.failure_type
                distance_to_location = self.distance_to_location
                #ood_nn = self.oodnn
                LBC_performance = self.LBC_performance
                LBC_safe = self.LBC_safe
                AP_scorenn = self.AP_scorenn
                mcts_queue = self.mcts_queue
                current_scene = self.current_scene.copy()
                current_scene[8] = self.current_traffic_density


                for rank in range(self.num_processes):
                    #p = Process(target=mcts_planner, args=(current_scene, self.semantic_data[self.k // 2:5], self.failure_type, self.distance_to_location, self.LBC_performance, self.LBC_safe, self.AP_scorenn, self.OOD_time_estimation, self.time_interval, self.mcts_queue))
                    p = Process(target=mcts_planner, args=(
                    current_scene, self.semantic_data[self.k // 2:5], self.failure_type, self.distance_to_location, self.mcts_queue))
                    p.start()
                    print("I have started")
                    self.processes.append(p)
                time.sleep(4)
                #self.PlannerProcess = Process(target=mcts_planner, args=(current_scene, self.semantic_data[self.k // 2:5], self.failure_type, self.distance_to_location, self.LBC_performance, self.LBC_safe, self.AP_scorenn, self.OOD_time_estimation, self.time_interval, self.mcts_queue))
                #self.PlannerProcess.daemon = True
                #print("####################pre_start")
                #self.PlannerProcess.start()
                #print("####################after_start")
                self.planner_start = True
                self.controller = "Autopilot"
                self.process_scene = self.current_scene
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
            #elif self.time == 100 and self.planner_start:
            PlannerProcess_finish = True
            if self.planner_start:
                for p in self.processes:
                    if p.is_alive():
                        PlannerProcess_finish = False

            #elif (not self.PlannerProcess.is_alive()) and self.planner_start:
            if PlannerProcess_finish and self.planner_start:
                #(self.planner_start == True) and (self.PlannerProcess.is_alive() == False):
                for p in self.processes:
                    p.join()
                self.planner_start = False

                print("wait for join")
                #self.PlannerProcess.join()
                result_list = []
                #self.policy = self.mcts_planner()
                #if self.policy[0] < self.policy[1]:
                #    self.controller = "LBC"
                #    print("LBC selected")
                #    self.switch = 1
                #    self.wait = 1
                #elif self.policy[0] > self.policy[1]:
                #    self.controller = "Autopilot"
                #    print("Autopilot selected")
                while not self.mcts_queue.empty():
                    result_list.append(self.mcts_queue.get())
                action_list = [np.argmax(i) for i in result_list]
                action = Counter(action_list)
                action = action.most_common(1)[0][0]
                print("joined!!")
                #self.policy = self.mcts_queue.get()
                self.weather_flag = False
                self.scene_event = False
                self.sensor_flag = False
                self.ood_flag = False
                self.traffic_level_flag = False
                #print('####################')
                #print("\n")
                #print("Policy we have so far:", action)
                #print("\n")
                #print('####################')
                if action == 0:
                    self.controller = "Autopilot"
                    print("Autopilot selected")
                if action == 1:
                    self.controller = "LBC"
                    print("LBC selected")
                    self.switch = 1
                    self.wait = 1
                #self.PlannerProcess.terminate()
                #if self.policy[0] < self.policy[1]:
                #    self.controller = "LBC"
                #    print("LBC selected")
                #    self.switch = 1
                #    self.wait = 1
                #elif self.policy[0] > self.policy[1]:
                #    self.controller = "Autopilot"
                #    print("Autopilot selected")
                self.mcts_queue = Queue()
                self.processes = []
                steer,throttle,brake,desired_speed = self.get_control_action(tick_data,pos,theta,speed,near_node,far_node)
                with open('planning_duration.txt', 'a') as f:
                    f.write("planner duration:")
                    f.write(str((self.time+1)/20.0))
                    f.write("\n")
                self.time = 0
                #self.scene_flag = self.k
                if self.k == 0:
                    self.scene_flag = True
            if self.planner_start == True:
                self.time +=1


        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        print(speed)
        self.velocity = speed
        self.write_data(steer,throttle,brake,pos,speed,theta)

        if brake == 1:
            throttle = -1
        
        data.append(speed)
        data.append(steer)
        data.append(throttle)
        self.state_data.append(data)

        if len(self.state_data) >=self.sliding_window:
            self.state_data = self.state_data[-1*self.sliding_window:]
        

        if self.step != 0 and self.step % 18 == 0:
            #print("stuck here")
            BlurDetectorThread.join()
            OccusionDetectorThread.join()
            SegmentationModelThread.join()
            self.blur = self.blur_queue.get()
            self.occlusion = self.occlusion_queue.get()
            #print(self.blur)
            self.outlier_score = self.segmentation_queue.get()
            print(self.outlier_score)
            if self.blur[0] == 1 or self.occlusion[0] == 1:
                self.cam_failure = ['1','0','0']
            elif self.blur[1] == 1 or self.occlusion[1] == 1:
                self.cam_failure = ['0','1','0']
            if self.blur[2] == 1 or self.occlusion[2] == 1:
                self.cam_failure = ['0','0','1']

        return control
