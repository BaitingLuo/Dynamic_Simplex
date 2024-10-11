#!/bin/python3
import numpy as np
import csv
import torch
import torch.nn as nn
from team_code.controller.Controller_Selector import Controller_Selector
from torch.utils.data import DataLoader

def Online_selector(lbc,ap,configuration):
    """
    Offline decision logic that uses prior belief
    """
    #if lbc == ap:
    #    controller = "Autopilot"
    #controller = "same"
    #    controller = "Autopilot"
    if configuration == "DS":
        controller = "same"
        #if lbc > ap:
        #    controller = "LBC"
        if lbc <= ap:
            controller = "Autopilot"
    elif configuration == "GS":
        if lbc > ap:
            controller = "LBC"
        if lbc <= ap:
            controller = "Autopilot"
    elif configuration == "SA":
        controller = "same"
        if lbc <= ap:
            controller = "Autopilot"
    else:
        controller = "same"
    #print(controller)
    
    return controller

def Offline_selector(lbc,ap):
    """
    Online selector that combines runtime monitor results and prior beliefs
    DNN classifier to predict the optimal controller
    """
    controller = lbc

    return controller

def Switching_Routine(speed,switching_speed):
    """
    Checks the speed before switching
    """
    if speed > switching_speed:
        action = -1
        switch = 0
    elif speed <= switching_speed:
        action = 1
        switch = 1

    return switch,action

def select_controller(action,switch,selected_controller,current_controller) :
    driving_controller = ""
    if switch == 0 and action == 0:
            driving_controller = current_controller
    elif switch == 1 and action == -1:
        print("slow speed to switch")
        driving_controller = current_controller
        throttle = 0.4
    elif switch == 1 and action == 1:
        driving_controller = selected_controller
    
    return driving_controller
    

# def Switcher(speed,current_controller,selected_controller):
#     """
#     Switcher observes the speed of the system and domain rules to decide when to switch
#     It activates a routine that starts to slow the car before switching
#     """
    
#     switching_speed = 2.5
#     #switch = 0
#     #action = 0
#     if current_controller != selected_controller:
#         if current_controller == "LBC" and selected_controller == "Autopilot":
#             switch,action = Switching_Routine(speed,switching_speed)
#         elif current_controller == "Autopilot" and selected_controller == "LBC":
#             planner_result = planner()
#             switch,action = Switching_Routine(speed,switching_speed)
#     else:
#         switch = 0
#         action = 0
    
#     driving_controller = select_controller(action,switch,selected_controller,current_controller) 
    
#     return switch,action,driving_controller

def Switcher(speed,current_controller,selected_controller):
    """
    Switcher observes the speed of the system and domain rules to decide when to switch
    It activates a routine that starts to slow the car before switching
    """
    
    switching_speed = 2.5
    #switch = 0
    #action = 0
    if current_controller != selected_controller:
        if current_controller == "LBC" and selected_controller == "Autopilot":
            switch = 3
        elif current_controller == "Autopilot" and selected_controller == "LBC":
            switch = 2
    else:
        switch = 0
    
    #driving_controller = select_controller(action,switch,selected_controller,current_controller) 
    
    return switch


def Selector(failure_prediction,ood_score,blur,occlusion,lbc_prior_belief,ap_prior_belief,speed,current_controller,configuration):
    """
    Decision Manager Code goes into this area
    """
    print("selector belief:", lbc_prior_belief, ap_prior_belief)
    selected_controller = Online_selector(lbc_prior_belief,ap_prior_belief,configuration)
    
    return selected_controller


