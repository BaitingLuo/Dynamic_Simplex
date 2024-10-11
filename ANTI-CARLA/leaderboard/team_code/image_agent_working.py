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
from queue import Queue
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
from MCTS import MCTS


#from controller.lbc import Learning_by_cheating
from controller.autopilot import autopilot

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
SAVE_PATH = os.environ.get('SAVE_PATH', None)


CarlaDataProvider
def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144


        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

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

#Function that gets in weather from a file
def process_weather_data(weather_file):
    weather = []
    lines = []
    with open(weather_file, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            weather.append(row)

    return weather

def init_seg_model():
    root = "/ANTI-CARLA/leaderboard/team_code/"
    path = "/ANTI-CARLA/leaderboard/team_code/Config/"
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

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()


    with open(path + "{}.json".format(args.dataset), "r") as thisFile:
        dataset_json = json.load(thisFile)
        dataset["mean"] = dataset_json["mean"]
        dataset["std"] = dataset_json["std"]
        dataset["class"] = dataset_json["class"]

    config = BaseConfigParser(args, dataset, model, optimizer)
    seg_model = BaseModelHandler(config,root)

    mean = dataset["mean"]
    std = dataset["std"]

    return seg_model, mean, std


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
            x.append(int(row[0]))
            x.append(int(row[1]))
            x.append(int(row[2]))
            x.append(int(row[3]))
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
    driving_score = []
    for val in X_train:
        if entry[0] == val[0] and entry[1] == val[1] and entry[2] == val[2] and entry[3] == val[3]:
            match.append(val)
            weather.append([val[4],val[5],val[6],val[7]])
            driving_score.append([val[8],val[9]])


    return match,weather,driving_score

def get_driving_score(x,X_subset,X_weather,X_driving,knn):
    lbc_score = []
    ap_score = []
    curr_weather = [x[4],x[5],x[6],x[7]]
    for val in X_subset:
        if (x[4] == val[4]) and (x[5] == val[5]) and (x[6] == val[6]) and (x[7] == val[7]):
            lbc_score.append(float(val[8]))
            ap_score.append(float(val[9]))
        else:
            knn.fit(X_weather,X_driving)
            curr = np.array(np.array(curr_weather).reshape(1,-1))
            neighbors = knn.kneighbors(curr, return_distance=False)
            for val in neighbors[0]:
                lbc_score.append(float(X_subset[val][8]))
                ap_score.append(float(X_subset[val][9]))
    #print(lbc_score)
    #print(ap_score)

    return round(sum(lbc_score)/len(lbc_score),2), round(sum(ap_score)/len(ap_score),2)

def predict(X_train,X_test,knn):
    Y_pred = []
    #print(X_train)
    X_subset, X_weather, X_driving = search(X_test,X_train)
    #print(X_subset)
    lbc_score, ap_score = get_driving_score(X_test,X_subset,X_weather,X_driving,knn)
    #print(lbc_score,ap_score)

    return lbc_score, ap_score


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]



class ImageAgent(BaseAgent):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self,path_to_conf_file,data_folder,project_path,route_folder,tracks_num):
        super().setup(path_to_conf_file,data_folder,project_path,route_folder,tracks_num)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()
        self.controller = "Autopilot"
        self.preferred_controller = "LBC"
        self.selected_controller = "Autopilot"
        #self.n_classes = 2
        #self.model_size = 256
        self.ckpt = project_path + "/trained_monitors/controller_selector_best.pth"
        print(route_folder)
        self.scene_scorer_train_dataset = "encoded_prior_belief.csv"
        self.train_data_path = project_path + "/trained_monitors/prior_data" + "/" + self.scene_scorer_train_dataset
        self.path_lut = route_folder + "label.csv"
        self.model = Controller_Selector(2,128)
        self.device = 'cuda:0'#get_device()
        #self.model.to(self.device)
        self.model.eval()
        self.sliding_window = 60
        self.failure_ckpt = project_path + "/trained_monitors/state_lstm_best.pth"
        self.failure_model = Introspective_LSTM(2,64,self.sliding_window)
        self.failure_model.to(self.device)
        self.failure_model.eval()
        self.failure_model_checkpoint = torch.load(self.failure_ckpt, map_location=torch.device('cpu'))
        self.failure_model.load_state_dict(self.failure_model_checkpoint["model_state"])
        self.failure_model = nn.DataParallel(self.failure_model)
        self.failure_model.to(self.device)
        self.failure_cur_itrs = self.failure_model_checkpoint["cur_itrs"]
        self.failure_best_score = self.failure_model_checkpoint["best_score"]
        #Load weights
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
        self.dm_queue = Queue(maxsize=1)
        self.lbc_queue = Queue(maxsize=1)
        self.ap_queue = Queue(maxsize=1)
        self.segmentation_queue = Queue(maxsize=1)
        self.weather_queue = Queue(maxsize=1)
        self.failure_prediction_queue = Queue(maxsize=1)
        self.environment_classifier_queue = Queue(maxsize=1)
        self.state = []
        self.monitors = []
        self.weather = []
        self.my_weather = []
        self.state_data = []
        self.collision_likelihood_window = []
        self.collision_likelihood = 0.0
        #self.base_weather = []
        self.blur = [0,0,0]
        self.occlusion = [0,0,0]
        self.outlier_score = 0.0
        self.failure_prediction_score = 0.0
        self.classifier_score = 0.0
        self.combined_score = 0.0
        self.dm = []
        self.lbc = []
        self.ap = []
        self.seg = []
        self.x = 0
        self.y = 0
        self.k = 0
        self.val = 1
        self.weather_file = route_folder + "/changing_weather.csv"
        self.tracks_num = tracks_num
        self.semantic_data = []
        self.time=0
        self.current_scene = []
        self.action_set = [0, 1]


        #print("Environment Classifier Model Training state restored from %s" % self.ckpt)
        #print("Environment Classifier Model restored from %s" % self.ckpt)
        print("Training state for the failure prediction model restored from %s" % self.failure_ckpt)
        print("Failure Prediction Model restored from %s" % self.failure_ckpt)

        del self.checkpoint  #free memory
        del self.failure_model_checkpoint # free memory

        self.weather = process_weather_data(self.weather_file)
        #print(self.weather)

    def _init(self):
        #super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._command_planner = RoutePlanner(7.5, 25.0, 257,self._global_wp,self.scene_gps,self.trajectory_wp)
        self._command_planner.set_route(self._global_plan, True)
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=3, p=2,radius=1.0)
        self.prior_scene_scores = read_data(self.train_data_path)
        with open(self.path_lut, newline='') as csvfile:
            self.semantic_data = list(csv.reader(csvfile))

        self.initialized = True

         # for stop signs
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

        self.seg_model, self.mean, self.std = init_seg_model()


#------------------------------- Runtime Monitors ------------------------------------------------

    def blur_detection(self,result):
        self.blur =[]
        fm1,rgb_blur = blur_detector(result['rgb'], threshold=20)
        fm2,rgb_left_blur = blur_detector(result['rgb_left'], threshold=20)
        fm3,rgb_right_blur = blur_detector(result['rgb_right'], threshold=20)
        self.blur.append(rgb_blur)
        self.blur.append(rgb_left_blur)
        self.blur.append(rgb_right_blur)
        self.blur_queue.put(self.blur)

    def occlusion_detection(self,result):
        self.occlusion = []
        percent1,rgb_occluded = occlusion_detector(result['rgb'], threshold=25)
        percent2,rgb_left_occluded = occlusion_detector(result['rgb_left'], threshold=25)
        percent3,rgb_right_occluded = occlusion_detector(result['rgb_right'], threshold=25)
        self.occlusion.append(rgb_occluded)
        self.occlusion.append(rgb_left_occluded)
        self.occlusion.append(rgb_right_occluded)
        self.occlusion_queue.put(self.occlusion)

    def decision_manager(self,tick_data,failure_prediction,risk,ood_score,blur,occlusion):
        controller = DecisionManager(self.step,tick_data,failure_prediction,risk,ood_score,blur,occlusion)
        self.dm_queue.put(controller)

    def lbc_control_actions(self,tick_data,pos,theta,speed):
        print("1")
        self.lbc = []
        steer,throttle,brake,desired_speed = self.Learning_by_cheating(tick_data,pos,theta, speed)
        self.lbc.append(steer)
        self.lbc.append(throttle)
        self.lbc.append(brake)
        self.lbc.append(desired_speed)
        self.lbc_queue.put(self.lbc)

    def ap_control_actions(self,near_node, far_node, tick_data,theta, speed):
        print("2")
        self.ap = []
        steer,throttle,brake,desired_speed = self.autopilot(near_node, far_node, tick_data,theta, speed)
        self.ap.append(steer)
        self.ap.append(throttle)
        self.ap.append(brake)
        self.ap.append(desired_speed)
        self.ap_queue.put(self.ap)


    def create_input_data(self,bin_weather):
        """
        Process input data and prepare it for the classifier
        """
        k = self.k//2
        data = []
        our_weather = self.my_weather
        
        data = [int(i) for i in self.semantic_data[k]]
        
        for weather in our_weather:
            data.append(create_bin(int(weather)))
            
        return data


    def Scene_Scorer(self,tick_data):
        """
        Scene_Scorer:
        Input: semantic labels, traffic density, weather
        Output: Driving Score of the two controllers
        """
        x = time.time()
        # current_scene = self.create_input_data(bin_weather=1)
        lbc_score, ap_score = predict(self.prior_scene_scores,self.current_scene,self.knn)
        print(time.time()-x)
        
        self.environment_classifier_queue.put([lbc_score,ap_score])

    def Segmentation_Outlier_Detector(self,result):
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
        image = torch.unsqueeze(image,0)

        self.outlier_score = self.seg_model.getEvaluation(image)

        self.segmentation_queue.put(1-np.mean(self.outlier_score))

    def predict_failure(self,result):
        #print(self.state_data)
        #Normalize data with values from training set
        state_avg = [2.975678 , 0.005583 , 0.098343]
        state_max = [7.285361 , 0.469301 , 1]
        data_normalized = []
        for i, row in enumerate(self.state_data):
            for s in range(3): #3 states
                for k in range(int(len(row)/3)):
                    row[3*k+s] = (float(row[3*k+s]) - state_avg[s])/state_max[s]
                    row[3*k+s] = 0.001*np.round(1000*row[3*k+s])
            data_normalized.append(row)

        softmax = nn.Softmax(dim=0)
        data = torch.from_numpy(np.array(data_normalized)).float().unsqueeze(0)

        with torch.no_grad():
            data = data.to(self.device)
            output_data = self.failure_model(data)
            probability = softmax(output_data[0]).cpu().numpy()

        self.failure_prediction_score = np.round(1000*probability[1])*0.001
        self.failure_prediction_queue.put(self.failure_prediction_score)


#------------------------------Controller Code---------------------------------------------------------------

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
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
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
            [np.sin(theta),  np.cos(theta)],
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
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)
        #self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord,wp_route,scene_gps, trajectory_wp):
        super().set_global_plan(global_plan_gps, global_plan_world_coord,wp_route,scene_gps, trajectory_wp)
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps


    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def write_data(self,steer,throttle,brake,pos,speed,theta):

        stats = []
        stats.append(self.step)
        stats.append(float(self.step/20))
        stats.append(speed)
        stats.append(steer)
        stats.append(throttle)
        stats.append(brake)
        #stats.append(self.angle)
        stats.append(pos[0])
        stats.append(pos[1])
        stats.append(theta)
        with open(self.data_folder, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(stats)


    def Learning_by_cheating(self,tick_data,pos,theta, speed):
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

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2DecisionManagerThread.start()

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        return steer,throttle,brake,desired_speed

    def autopilot(self,target, far_target, tick_data,theta, speed):
        #print("AP")
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

#--------------------------Simulation Tick-----------------------------------------------------

    def tick(self, input_data):
        if self.step == -1:
            self.my_weather = self.weather[0]
            self.current_scene = self.create_input_data(bin_weather=1)
            self.current_scene.append('0')
            self.current_scene.append('0')
            self.current_scene.append('0')
            self.current_scene.append('3.84')
            self.current_scene.append('0')
            self.current_scene.append('2.38')
            self.current_scene.append('0')
            self.current_scene.append(0)
            self.current_scene.append(0)
            print(self.current_scene)
            
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        topdown = input_data['map'][1][:, :, 2]
        #theta = input_data['compass']
        #speed = _data['speed']

        result = {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'topdown': topdown
                #'rgb_rear': rgb_rear,
				#'lidar': lidar,
                }

        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        EnvironmentClassifierThread = Thread(target=self.Scene_Scorer, args=(result,))
        OccusionDetectorThread = Thread(target=self.occlusion_detection, args=(result,))
        BlurDetectorThread = Thread(target=self.blur_detection, args=(result,))
        SegmentationModelThread = Thread(target=self.Segmentation_Outlier_Detector, args=(result,)) #image,model,calibration_set,pval_queue,sval_queue
        EnvironmentClassifierThread.daemon = True
        BlurDetectorThread.daemon = True
        OccusionDetectorThread.daemon = True
        SegmentationModelThread.daemon = True

        waypoints_left, current_waypoint_index, current_waypoint = self._command_planner.get_waypoints_remaining(gps)
        #print("step:%d waypoints:%d"%(self.step,(waypoints_left)))
        #print("waypoint index:%d"%current_waypoint_index)
        #print(current_waypoint)

        #waypoints_left, current_waypoint = self._command_planner.get_waypoints_remaining(gps)
        #k = (waypoints_left+1)//2 - 5#
        #k=0
        if self.step%400 == 0:
            self.k+=2
            self.val=1
            if int(self.my_weather[3]) >= 90:
                self.my_weather[3] = 90
            elif int(self.my_weather[3]) < 90:
                self.my_weather[3] = int(self.my_weather[3]) + 5
            
            for entry in range(len(self.my_weather)-1):
                if int(self.my_weather[entry]) < 100:
                    self.my_weather[entry] = int(self.my_weather[entry]) + 5
                else:
                    self.my_weather[entry] = 100

            #self.my_weather = [int(self.my_weather[0]),int(self.my_weather[1]),int(self.my_weather[2]),int(self.my_weather[3])]
            self.weather = [self.my_weather]
            print("-------------------------------")
            print("\n")
            print("Weather Change", self.my_weather)
            print("Location",self.semantic_data[self.k//2])
            print("\n")
            print("-------------------------------")
            carla_weather = carla.WeatherParameters(cloudiness=self.my_weather[0], precipitation=self.my_weather[1], precipitation_deposits=self.my_weather[2], sun_altitude_angle=self.my_weather[3])
            self._world.set_weather(carla_weather)

        if self.k%2 == 0 and self.val == 1:
            EnvironmentClassifierThread.start()

        if self.step > 60:
            FailurePredictionThread = Thread(target=self.predict_failure, args=(result,))
            FailurePredictionThread.daemon = True
            FailurePredictionThread.start()

        if self.step != 0 and self.step % 20 == 0:
            BlurDetectorThread.start()
            OccusionDetectorThread.start()
            SegmentationModelThread.start()

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        far_node, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target

        if self.step > 60:
            FailurePredictionThread.join()
            self.failure_prediction_score = self.failure_prediction_queue.get()
        
        self.collision_likelihood_window.append(self.failure_prediction_score)

        if len(self.collision_likelihood_window) >=self.sliding_window:
            self.collision_likelihood_window = self.collision_likelihood_window[-1*self.sliding_window:]
        
        self.collision_likelihood = round(sum(self.collision_likelihood_window)/self.sliding_window,2)
            

        if self.step != 0 and self.step % 20 == 0:

            BlurDetectorThread.join()
            OccusionDetectorThread.join()
            SegmentationModelThread.join()

            self.blur = self.blur_queue.get()
            self.occlusion = self.occlusion_queue.get()
            self.outlier_score = self.segmentation_queue.get()

            #self.combined_score = 0.1*self.blur[0] + 0.1*self.occlusion[0] + 0.1*self.outlier_score + 0.4*self.classifier_score + 0.3*self.failure_prediction_score

            #print("Combined Monitor Scores:%f"%self.combined_score)

            #print("Collision Likelihood:%f"%self.collision_likelihood)


        return result, far_node, theta, speed, EnvironmentClassifierThread


    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        
        #print(self.semantic_data)
        
        self.time+=1

        data = []

        x = time.time()

        tick_data, far_node, theta, speed, EnvironmentClassifierThread = self.tick(input_data)
        pos = self._get_position(tick_data)
        near_node, near_command = self._waypoint_planner.run_step(pos)

        if self.k%2 == 0 and self.val == 1:
            EnvironmentClassifierThread.join()
            self.prior_belief_score = self.environment_classifier_queue.get()
            self.lbc_prior_belief = self.prior_belief_score[0]
            self.ap_prior_belief = self.prior_belief_score[1]
            print("LBC prior belief:%f, AP prior belief:%f"%(self.lbc_prior_belief,self.ap_prior_belief))
            

        if (self.k%2 == 0 and self.val == 1) or float(self.collision_likelihood) > 0.5 or float(self.outlier_score) < 0.7 or sum(self.blur) >=1 or sum(self.occlusion)>=1:
            self.decision_query = 1
            self.selected_controller = Selector(self.failure_prediction_score,self.outlier_score,self.blur,self.occlusion,self.lbc_prior_belief,self.ap_prior_belief,speed,self.controller) #Decision Manager
            self.val = 0

        switch,action,self.controller = Switcher(speed,self.controller,self.selected_controller)

        # if switch == 0 and action == 0:
        #     self.controller = self.controller
        # elif switch == 1 and action == -1:
        #     print("slow speed to switch")
        #     self.controller = self.controller
        #     throttle = 0.4
        # elif switch == 1 and action == 1:
        #     self.controller = self.selected_controller

        if self.controller == "Autopilot" and self.time==10:
            alpha_args = dotdict({
                'tempThreshold': 15,  #
                'updateThreshold': 0.6,
                'numMCTSSims': 20,  # Number of games moves for MCTS to simulate.
                'cpuct': math.sqrt(2),
            })
            mcts = MCTS(alpha_args, self.semantic_data, 4)
            temp = 1
            pi = mcts.getActionProb(self.current_scene, self.semantic_data, self.action_set, temp=temp)
            print(pi)
            self.time = 0

        
        #print(self.controller)

        if self.controller == "LBC":
            steer,throttle,brake,desired_speed = self.Learning_by_cheating(tick_data,pos,theta, speed)
        elif self.controller == "Autopilot":
            steer,throttle,brake,desired_speed = self.autopilot(near_node, far_node, tick_data,theta, speed)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        print(speed)
        
        self.write_data(steer,throttle,brake,pos,speed,theta)

        if brake == 1:
            throttle = -1
        
        data.append(speed)
        data.append(steer)
        data.append(throttle)

        self.state_data.append(data)

        if len(self.state_data) >=self.sliding_window:
            self.state_data = self.state_data[-1*self.sliding_window:]


        return control

