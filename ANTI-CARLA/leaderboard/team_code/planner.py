import os
from collections import deque
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import xml.etree.ElementTree as ET
import leaderboard.utils.route_manipulation as rm
import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        #if not DEBUG:
        #    return

        import cv2

        #cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        #cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256, global_wp=[], scene_gps=[], taj_wp=[]):
        self.route = deque()
        self.scene_route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.trajectory_wp = taj_wp
        self.first = True
        self.global_wp = global_wp
        print(len(self.global_wp))
        print(self.global_wp)
        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10
        self.index = 1
        self.debug = Plotter(debug_size)
        self.route_length = 0
        self.scene_route_length = 0
        self.scene_gps=scene_gps
        self.set_route_scene(self.scene_gps,True)
    
    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))
            self.route_length = len(self.route)
    def set_route_scene(self, scene_gps, gps=False):
        self.scene_route.clear()

        for pos in scene_gps:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.scene_route.append(pos)
            self.scene_route_length = len(self.scene_route)
    def run_step(self, gps):
        self.debug.clear()
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        #print(len(self.route),self.route)
        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            self.debug.dot(gps, self.route[i][0], (r, g, b))

        to_pop_scene = 0
        farthest_in_range_scene = -np.inf
        cumulative_distance_scene = 0.0
        #print("length:",len(self.scene_route))
        for i in range(1, len(self.scene_route)):
            if cumulative_distance_scene > self.max_distance:
                break

            cumulative_distance_scene += np.linalg.norm(self.scene_route[i] - self.scene_route[i-1])
            distance = np.linalg.norm(self.scene_route[i] - gps)
            #print("distance:",i,":",distance)
            if distance <= self.min_distance and distance > farthest_in_range_scene:
                farthest_in_range_scene = distance
                to_pop_scene = i


        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        for _ in range(to_pop_scene):
            if len(self.scene_route) > 2:
                self.scene_route.popleft()

        self.index = self.scene_route_length - len(self.scene_route)
        self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.dot(gps, self.route[1][0], (0, 0, 255))
        self.debug.show()

        return self.route[1]

    def get_waypoints_remaining(self,gps):
        #print(self.route)
        if len(self.route) == 1:
            print(len(self.route), self.index, self.trajectory_wp.index(self.trajectory_wp[self.index]), len(self.trajectory_wp))
            return len(self.route), self.index, self.trajectory_wp.index(self.trajectory_wp[self.index]), len(self.trajectory_wp)
            #return len(self.route), self.route[1]
        else:
            #print(self.route)
            #print(self.trajectory_wp)
            return len(self.route), self.index, self.trajectory_wp.index(self.trajectory_wp[self.index]),len(self.trajectory_wp)
            #return len(self.route), self.route[1]
