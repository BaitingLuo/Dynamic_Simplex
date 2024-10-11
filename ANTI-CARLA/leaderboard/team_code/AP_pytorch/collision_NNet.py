import os
import sys
import time

import numpy as np
from tqdm import tqdm

from util import *


import torch
import torch.optim as optim

from .ANN import Network as nnet

args = dotdict({
    'lr': 0.00001,
    'dropout': 0.3,
    'epochs': 1000,
    'batch_size': 1,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper():
    def __init__(self, x_d, y_d):
        self.nnet = nnet(x_d, y_d)
        self.state_x, self.state_y = x_d, y_d

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        k_fold = 5
        total_size = len(examples)
        fraction = 1 / k_fold
        seg = int(total_size * fraction)
        print(total_size)
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            #self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            val_pi_losses = AverageMeter()
            val_v_losses = AverageMeter()
            k = epoch % k_fold
            trll = 0
            trlr = k * seg
            vall = trlr
            valr = k * seg + seg
            trrl = valr
            trrr = total_size
            train_left_indices = list(range(trll, trlr))
            train_right_indices = list(range(trrl, trrr))
            val_indices = list(range(vall, valr))
            if (len(train_left_indices) != 0 and len(train_right_indices) != 0):
                training_data = examples[train_left_indices[0]:(train_left_indices[-1] + 1)] + examples[train_right_indices[0]:(
                        train_right_indices[-1] + 1)]
            else:
                train_indices = train_left_indices + train_right_indices
                training_data = examples[train_indices[0]:(train_indices[-1] + 1)]
            validation_data = examples[val_indices[0]:(val_indices[-1] + 1)]
            training_batch_count = int(len(training_data) / args.batch_size)
            t = tqdm(range(training_batch_count), desc='Training Net')
            val_batch_count = int(len(validation_data) / args.batch_size)
            t_v = tqdm(range(val_batch_count), desc='Validation Net')
            for _ in t:
                sample_ids = np.random.randint(len(training_data), size=args.batch_size)
                states, speed, collision = list(zip(*[training_data[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float))

                target_speed = torch.FloatTensor(np.array(speed))
                target_collision = torch.FloatTensor(np.array(collision).astype(np.float))
                # predict
                if args.cuda:
                    states, target_speed, target_collision = states.contiguous().cuda(), target_speed.contiguous().cuda(), target_collision.contiguous().cuda()

                # compute output
                #print("################")
                #print(states)
                #out_pi, out_v = self.nnet(states)
                out_v = self.nnet(states)
                #print(out_pi, out_v)
                #out_v = (out_v>0.3).float()
                #print(out_v)
                #print(out_v, target_collision)
                #l_pi = self.loss_pi(target_speed, out_pi)
                l_v = self.loss_v(target_collision, out_v)
                #print("#############")
                #print("speed:",target_speed, out_pi)
                #print(l_pi)
                #print(target_collision, out_v)
                #print(l_v)
                #print(target_collision, out_v)
                total_loss = l_v
                #total_loss = l_pi + l_v
                out_v = (out_v > 0.3).float()
                #print(target_collision, out_v)
                threshold = self.loss_v(target_collision, out_v)
                # record loss
                #pi_losses.update(l_pi.item(), states.size(0))
                #v_losses.update(l_v.item(), states.size(0))
                #print(threshold)
                v_losses.update(threshold.item(), states.size(0))
                #t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                t.set_postfix(Loss_pi=v_losses)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            for _ in t_v:
                sample_ids = np.random.randint(len(validation_data), size=args.batch_size)

                states, pis, vs = list(zip(*[validation_data[i] for i in sample_ids]))

                states = torch.FloatTensor(np.array(states).astype(np.float))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float))
                # predict
                if args.cuda:
                    states, target_pis, target_vs = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_v = self.nnet(states)
                #val_pi = self.loss_pi(target_pis, out_pi)
                val_v = self.loss_v(target_vs, out_v)
                #total_loss = l_pi + l_v


                # record loss
                #val_pi_losses.update(val_pi.item(), states.size(0))
                val_v_losses.update(val_v.item(), states.size(0))
                #t_v.set_postfix(Loss_pi=val_pi_losses, Loss_v=val_v_losses)
                t_v.set_postfix(Loss_pi=val_v_losses)
                # compute gradient and do SGD step
            #print(pi_losses, v_losses)
            with open('training_losses.txt', 'a') as f:
                f.write("%f " % (epoch+1))
                f.write("pi: %f " % pi_losses.avg)
                f.write("v: %f" % v_losses.avg)
                f.write("\n")
        with open('training_losses.txt', 'a') as f:
            f.write("\n")
    def predict(self, state):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        

        state = torch.FloatTensor(np.array(state).astype(np.float))

        if args.cuda: state = state.contiguous().cuda()


        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(state)

        return pi[0], v[0]

    def loss_pi(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        #print(torch.sum(targets * outputs))
        #return -torch.sum(targets * outputs) / targets.size()[0]
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        #return -torch.sum(targets * outputs) / targets.size()[0]
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
