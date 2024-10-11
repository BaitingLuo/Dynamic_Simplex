#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import os
import csv
import sys
import time
import datetime
import py_trees
import carla
import nvidia_smi
import psutil
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
import math
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manger
    """
    
    def __init__(self, timeout, debug_mode=False):
        
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None
        self.collison_count = 0
        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)


        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def function_handler(self,event):
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        #print(actor_we_collide_against)
        world = CarlaDataProvider.get_world()
        self.collison_count += 1
        print("#############################")
        print("collision happened")
        print(self.collison_count)
        print("#############################")
        #self._running = False



    def run_scenario(self,path1):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        world = CarlaDataProvider.get_world()
        blueprint_library = world.get_blueprint_library()

        collision_sensor = world.spawn_actor(blueprint_library.find('sensor.other.collision'),
                                             carla.Transform(), attach_to=self.ego_vehicles[0])
        collision_sensor.listen(lambda event: self.function_handler(event))
        x = 0
        while self._running:
            timestamp = None
            #world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
                    #print(timestamp, datetime.datetime.now())
            if timestamp:
                y = time.time()
                self._tick_scenario(timestamp)
                z = time.time() - y
            
            x+=1
            if(x%60==1):
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                #print(f'mem: {mem_res.used / (1024**2)} (GiB)') # usage in GiB
                print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%') # percentage usage
                cpu = psutil.cpu_percent()#cpu utilization stats
                mem = psutil.virtual_memory()#virtual memory stats
                print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

                fields = ['GPU Utilization',
                            'GPU Memory',
                            'CPU Utilization',
                            'CPU Memory',
                            'Time'
                        ]

                dict = [{'GPU Utilization':res.gpu, 'GPU Memory':100 * (mem_res.used / mem_res.total), 'CPU Utilization':cpu, 'CPU Memory':100 * (mem.used/mem.total),'Time':round(float(z),2)}]


                file_exists = os.path.isfile(path1)
                with open(path1, 'a') as csvfile:
                    # creating a csv dict writer object
                    writer = csv.DictWriter(csvfile, fieldnames = fields)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(dict)


    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            #to be called everytime a new tick is received
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)
            #print("num of ego:", len(self.ego_vehicles))
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()
            #print(timestamp)
            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                endtime = GameTime.get_time()
                with open('switch_record.txt', 'a') as f:
                    f.write(str(endtime))
                    f.write("\n")
                self._running = False

            #spectator = CarlaDataProvider.get_world().get_spectator()
            #ego_trans = self.ego_vehicles[0].get_transform()
            #spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
            #                                            carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'

        ResultOutputProvider(self, global_result)
