import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Introspective_LSTM import Introspective_LSTM

def predict_failure(model,device,data):

    #Normalize data with values from training set
    state_avg = [2.975678 , 0.005583 , 0.098343]
    state_max = [7.285361 , 0.469301 , 1]

    data_normalized = []
    for i, row in enumerate(data):
        for s in range(3): #3 states
            for k in range(int(len(row)/3)):
                row[3*k+s] = (float(row[3*k+s]) - state_avg[s])/state_max[s]
                row[3*k+s] = 0.001*np.round(1000*row[3*k+s])
        data_normalized.append(row)

    softmax = nn.Softmax(dim=0)
    data = torch.from_numpy(np.array(data_normalized)).float().unsqueeze(0)

    with torch.no_grad():
        data = data.to(device)
        output_data = model(data)
        probability = softmax(output_data[0]).cpu().numpy()

    print('Failure probability is', np.round(1000*probability[1])*0.001)
    return probability[1]

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device 

if __name__ == "__main__":
    path_root = './data_introspection/'
    model_size = 64
    n_classes = 2 #Failure and success classes
    sample_length = 60 #3 seconds at 20Hz
    ckpt = './checkpoints/state_lstm_best.pth'
    #Create model
    model = Introspective_LSTM(n_classes,model_size,sample_length)
    device = get_device()
    model.to(device)
    model.eval()
    #Load weights
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    cur_itrs = checkpoint["cur_itrs"]
    best_score = checkpoint['best_score']
    print("Training state restored from %s" % ckpt)
    print("Model restored from %s" % ckpt)
    del checkpoint  #free memory

    #Get test data
    data_path = path_root + 'data.csv'

    with open(data_path, newline='') as csvfile:
        input_list = list(csv.reader(csvfile, delimiter=',')) 
    
    n= 0
    test_data = []
    for i_data in range(n*60,(n+1)*60):
        d_i = []
        input_data = input_list[i_data]
        for s in input_data: d_i.append(float(s))
        test_data.append(d_i)

    #Get failure prediction
    prediction = predict_failure(model,device,test_data)