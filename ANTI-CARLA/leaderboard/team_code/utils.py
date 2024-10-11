#!/bin/python3
import csv
from nltk import flatten
from tempfile import NamedTemporaryFile
import shutil
from decimal import Decimal

def extract_collision_data(path):
    """
    Extract collision times from collision.txt files
    """
    #collision_path = []
    #for collision_data in glob.glob(path):
    #    collision_path.append(collision_data)
    #collision_path.sort(reverse=False)
    #print(collision_path)
    final_collisions = []
    #for j in range(len(collision_path)):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    count = 0
    data = []
    collisions = []
    col = []
    for line in Lines:
            data.append(line.strip())
    for i in range(len(data)):
        number = []
        if(data[i]!= "" and data[i][0].isdigit()):
            for k in range(len(data[i])):
                if(data[i][k].isdigit()):
                    number.append(data[i][k])
                elif(data[i][k] == " "):
                    break
            collisions.append(number)
    for x in range(len(collisions)):
        col.append("".join(collisions[x]))
    final_collisions.append(col)

    return flatten(final_collisions)


def write_to_state_file(state_data_file,temp_file,collision_times):
    """
    Append the collisions to state data file
    """
    i,j=0,0
    collision = []
    final = []
    #print(collision_times)
    for entry in collision_times:
        collision.append(Decimal(entry))
    with open(temp_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            data = []
            data.append(row)
            if float(row[1]) in collision:
                print("match")
                row.append(1)
                i+=1
                #row = {'step':row[0],'time':row[1],'speed': row[2], 'steer': row[3], 'throttle': row[4], 'brake': row[5], 'x': row[6], 'y': row[7], 'theta': row[8],'collision': 1}
            else:
                row.append(0)
                #row = {'step':row[0],'time':row[1],'speed': row[2], 'steer': row[3], 'throttle': row[4], 'brake': row[5], 'x': row[6], 'y': row[7], 'theta': row[8],'collision': 0}
            final.append(row)
            #csv_writer.writerow(row)


    with open(state_data_file, 'w') as f:
        fields = ['step','time','speed','steer','throttle','brake','x','y','theta','collision']
        #for row in csv_reader:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(final)


# def write_to_state_file(state_data_file,collision_times):
#     """
#     Append the collisions to state data file
#     """row[9] = updated_collisions[j]
            #j+=1
#     i = 0
#     tempfile = NamedTemporaryFile(mode='w', delete=False)
#     fields = ['step','time','speed','steer','throttle','brake','x','y','theta','collision']
#
#     with open(state_data_file, 'r') as csvfile, tempfile:
#         reader = csv.DictReader(state_data_file, fieldnames=fields)
#         writer = csv.DictWriter(tempfile, fieldnames=fields)
#         for row in reader:
#             print(row)
#             if row['collision'] == int(collision_times[i]):
#                 print('updating row', row['collision'])
#                 row['collision'] = 1
#                 i+=1
#             else:
#                 row['collision'] = 0
#             row = {'step':row['step'],'time':row['time'],'speed': row['speed'], 'steer': row['steer'], 'throttle': row['throttle'], 'brake': row['brake'], 'x': row['x'], 'y': row['y'], 'theta': row['theta'],'collision': row['collision']}
#             #row = {'ID': row['ID'], 'Name': row['Name'], 'Course': row['Course'], 'Year': row['Year']}
#             writer.writerow(row)
#
#     shutil.move(tempfile.name, state_data_file)
