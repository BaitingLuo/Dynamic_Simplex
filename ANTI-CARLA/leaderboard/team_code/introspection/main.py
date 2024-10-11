import numpy as np
import random
import json
import argparse
import torch
import torchvision
from PIL import Image
from skimage.transform import resize
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils as utils
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from Introspective_LSTM import Introspective_LSTM
from Sample_Dataset import Sample_Dataset 

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_root", type=str, default='./data_introspection/',
                        help="path to root of dataset")    
    parser.add_argument("--eval_mode", type=str, default='train',
                        choices=['train', 'test'], help='Train model or test model')      
    parser.add_argument("--size", type=int, default='64',
                        choices=['8','16','32', '64', '128'], help='Specify size of each layer')  
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Saved model checkpoint file")   
    parser.add_argument("--weights", type=int, default=12,
                        help="Weight of failure class in loss function")        
    parser.add_argument("--sample_length", type=int, default='60',
                        help="Length of input sample in points (seconds times frequency)")                                       
    return parser

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device  

def train_lstm(opts, train_dataset, val_dataset):
    #Create model
    model = Introspective_LSTM(n_classes,opts.size,opts.sample_length)
    device = get_device()
    model.to(device)
    #Epochs
    n_epochs = 20
    epoch = 0
    #Learning rate and batch size
    learning_rate = 0.001
    batch_size = 64

    #Loss function
    class_weights = torch.FloatTensor([1,opts.weights]).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights) 
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None:
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  #free memory
    else:
        print("Training new model")
        model = nn.DataParallel(model)
        model.to(device)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    #Create data loaders for training process
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True
    )

    #Train model
    going_on = True
    pbar = tqdm(total=len(train_loader))
    pbar.set_description("Training introspective state-based LSTM...")

    val_loss_min = 1000

    while going_on:
        #Iterate through up to n_epochs epochs
        # monitor training and validation loss
        train_loss = 0.0
        val_loss = 0.0
        train_error = []
        train_fpr = []
        train_fnr = []

        #Training
        for input_data, target_data in train_loader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()
            output_data = model(input_data)
            loss = criterion(output_data, target_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*target_data.size(0)
            pbar.update()

            output = output_data.cpu().detach().numpy()
            target = target_data.cpu().detach().numpy()

            for i in range(len(output)):
                train_error.append(np.absolute(target[i]-np.argmax(output[i])))          
                if target[i] == 1:
                    train_fnr.append(np.absolute(target[i]-np.argmax(output[i])))   
                else:
                    train_fpr.append(np.absolute(target[i]-np.argmax(output[i])))   

        print('Train error: ', np.average(train_error))
        if len(train_fnr)>0: 
            print('Train error for collisions: ', np.average(train_fnr))
        else:
            print('No train samples for collisions available!')
        if len(train_fpr)>0: 
            print('Train error for successes: ', np.average(train_fpr))
        else:
            print('No train samples for successful driving available!')
            
        train_loss = train_loss/len(train_loader)

        #Get validation loss every epoch
        test_error = []
        fpr = []
        fnr = []

        for input_data, target_data in val_loader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            optimizer.zero_grad()
            output_data = model(input_data)
            loss = criterion(output_data, target_data)
            val_loss += loss.item()*target_data.size(0)

            output = output_data.cpu().detach().numpy()
            target = target_data.cpu().detach().numpy()

            for i in range(len(output)):
                test_error.append(np.absolute(target[i]-np.argmax(output[i])))          
                if target[i] == 1:
                    fnr.append(np.absolute(target[i]-np.argmax(output[i])))   
                else:
                    fpr.append(np.absolute(target[i]-np.argmax(output[i])))   
            pbar.update()

        print('Val error: ', np.average(test_error))
        if len(fnr)>0: 
            print('Val error for collisions: ', np.average(fnr))
        else:
            print('No validation samples for collisions available!')
        if len(fpr)>0: 
            print('Val error for successes: ', np.average(fpr))
        else:
            print('No validation samples for successful driving available!')
        
        val_loss = val_loss/len(val_loader)                  
        #print('Epoch:{} \t Training Loss: {:.6f} \t Validation Loss: {:.6f} \t'.format(epoch, train_loss, val_loss))
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            val_loss_rising = 0
            print('Best val model saved at epoch ', epoch)
            save_ckpt('checkpoints/state_lstm_best.pth')   

        #If n_epochs is reached or if validation loss increased for too many epochs, stop training
        if epoch == n_epochs:
            going_on = False
        else:
            epoch = epoch + 1
            print('Model saved at epoch ', epoch)
            save_ckpt('checkpoints/state_lstm_current.pth')   
     
        pbar.close()

def test_lstm(opts, test_dataset):   
    path_root = opts.path_root
    n_classes = 2 #Failure and success classes
    sample_length = opts.sample_length #3 seconds at 20Hz
    #Create model
    model = Introspective_LSTM(n_classes,opts.size,sample_length)
    device = get_device()
    model.to(device)    

    #Create data loaders for testing
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    if opts.ckpt is not None:
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            #optimizer.load_state_dict(checkpoint["optimizer_state"])
            #scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt)
            del checkpoint  #free memory
    else:
        print("Trained model required for testing!")
        return False

    pbar = tqdm(total=len(test_loader))
    pbar.set_description("Testing introspective LSTM...")

    test_error = []
    fpr = []
    fnr = []
    labels = []
    predictions = []

    with torch.no_grad():
        for input_data, target_data in test_loader:
            input_data = input_data.to(device)
            output_data = model(input_data)
            output = np.argmax(output_data.cpu()).numpy().squeeze(0)
            target = target_data.cpu().numpy().squeeze(0)
            test_error.append(np.absolute(target-output))     
            if target == 1:
                fnr.append(np.absolute(target-output))
            else:
                fpr.append(np.absolute(target-output))
            pbar.update()

            prediction = output_data.cpu().numpy()
            predictions.append(prediction[0])
            labels.append(target)

    pbar.close()

    np.savetxt('./predictions_val.csv', predictions,fmt="%f", delimiter=',')
    np.savetxt('./labels_val.csv', labels,fmt="%f", delimiter=',')

    print('Average test error: ', np.average(test_error))
    if len(fnr)>0: 
        print('Average error for collision samples: ', np.average(fnr))
    else:
        print('No test samples for collisions available!')
    if len(fpr)>0: 
        print('Average error for successful samples: ', np.average(fpr))
    else:
        print('No test samples for successful driving available!')

if __name__ == "__main__":

    n_classes = 2 #Failure and success classes
    seq_length = 200 #10 seconds at 20Hz
    opts = get_argparser().parse_args()

    full_dataset = Sample_Dataset(opts.path_root + 'data_normalized.csv', n_classes, opts.path_root + 'labels_downsampled.csv', opts.sample_length, augment_data = False)

    n_seqs = int(len(full_dataset)/seq_length)
    n_train = int(np.round(0.8*n_seqs))

    random.seed(0)
    index_rand = random.sample(range(0, n_seqs), n_seqs)

    index_train_seq = index_rand[:n_train]
    index_val_seq = index_rand[n_train:]

    index_train = []
    index_val = []

    for i,_ in enumerate(index_train_seq):
        r_i = range(seq_length*index_train_seq[i],seq_length*index_train_seq[i]+seq_length)
        for r in list(r_i): index_train.append(r)
    for i,_ in enumerate(index_val_seq):
        r_i = range(seq_length*index_val_seq[i],seq_length*index_val_seq[i]+seq_length)
        for r in list(r_i): index_val.append(r)

    train_dataset = torch.utils.data.Subset(full_dataset, index_train)
    val_dataset = torch.utils.data.Subset(full_dataset, index_val)

    if opts.eval_mode == 'train':
        train_lstm(opts, train_dataset, val_dataset)
    else:
        opts.ckpt = './checkpoints/state_lstm_best.pth'
        test_lstm(opts, val_dataset)
        test_lstm(opts, train_dataset)