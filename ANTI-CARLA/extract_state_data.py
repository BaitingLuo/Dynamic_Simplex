#!/bin/python3
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import csv
import shutil


def read_folder_number(path,routes_folder):
    """
    Write the folder number in which the routes are stored
    """
    path = path + "/" + routes_folder + "/"
    file1 = open(path + "tmp.txt", "r")
    y = file1.read()
    file1.close() #to change file access modes
    os.remove(path + "tmp.txt")

    return y


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"
    print("running state data extraction")

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--project_path', type=str, help='Type the simulation folder to store the data')
    parser.add_argument('--data_folder', type=str, help='Location to store simulation statistics')
    parser.add_argument('--routes_folder', type=str, help='Location to store simulation statistics')
    parser.add_argument('--simulation_number', type=int, help='Type the simulation folder to store the data')
    parser.add_argument('--scene_number', type=int, default=1, help='Type the scene number to be executed')
    parser.add_argument('--store_folder', type=str, help='Location to store routes')

    args = parser.parse_args()

    data_path = args.project_path
    y = read_folder_number(data_path,args.routes_folder)
    #paths,run_root = create_root_folder(data_path,y,arguments)
    data_folder = '/home/baiting/Desktop/ICCPS/Dynamic_Controller_Testing_Final_nondocker/ANTI-CARLA' + args.data_folder + "/" + "run%s"%y + "/" + "simulation%d"%args.simulation_number + "/" + "scene%d"%args.scene_number + '/'
    file = data_folder + "state_data.csv"
    #os.makedirs(data_folder, exist_ok=True)
    store_folder = args.store_folder + "/" + "run%s"%y + "/" + "simulation%d"%args.simulation_number + "/"
    os.makedirs(store_folder, exist_ok=True)
    shutil.copy(file,store_folder)
    num = args.scene_number
    #num = args.simulation_number * 5 + args.scene_number
    os.rename(store_folder + "state_data.csv",store_folder + "state_data%d.csv"%num)


if __name__ == '__main__':
    main()
