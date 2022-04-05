from http.client import ImproperConnectionState
from Config import Config
from Model import CNN_TS_Model
from TrainingDataset import TrainingDataset
from torch import nn
import os
import torch
from time import time
import shutil

class Trainer:

    def __init__(self):
        
        self.train_loss = []
        self.test_loss = []

        self.model = CNN_TS_Model()
        self.dataset = TrainingDataset()
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), Config.LR)

        self.experiment_path = 'Experiments/'+Config.EXPERIMENT_FOLDER_NAME
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        else:
            raise FileExistsError('This Folder is already created!')
        
        shutil.copyfile('Config.py', self.experiment_path+'/Config.py')

        data_info_string = str(self.dataset.count_balance()) + '/n/n' + self.dataset.print_info(if_print = False)
        file = open(self.experiment_path+'/dataset_info.txt', 'w')
        file.write(data_info_string)
        file.close()
        del data_info_string

        self.logs = open(self.experiment_path+'/logx.txt', 'w')

    def save_model(self):
        pass

    def print_log(self, string):
        self.logs.write(string + '\n')
        print(string)
    
    def train_one_epoch(self, epoch_number):
        self.print_log('Start training epoch number '+str(epoch_number)+'\n')
        self.model.train()
        total_loss = 0
        loss_to_print = 0
        start_time = time.time()
        for step_number in range(1, self.dataset.get_train_len()+1):
            self.optimizer.zero_grad()
            x, y = self.dataset.get_train_batch()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            loss_to_print += loss.item()
            loss.backward()
            self.optimizer.step()

            if step_number%Config.STEPS_TO_PRINT == 0:
                string = '\tStep number : '+str(step_number)+'/'+str(self.dataset.get_train_len())
                string += ' | Total loss : '+str(loss_to_print)+' | Mean loss : '+str(loss_to_print/Config.STEPS_TO_PRINT)
                string += ' | Time : '+str(time.time() - start_time)
                self.print_log(string)
                start_time = time.time()
                loss_to_print = 0

        self.train_loss.append(total_loss)

    def test_one_epoch(self, epoch_number):
        self.print_log('Start testing epoch number '+str(epoch_number)+'\n')
        self.model.eval()
        total_loss = 0
        loss_to_print = 0
        start_time = time.time()
        for step_number in range(1, self.dataset.get_test_len()+1):
            
            with torch.no_grad():
                x, y = self.dataset.get_test_batch()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                loss_to_print += loss.item()

            if step_number%Config.STEPS_TO_PRINT == 0:
                string = '\tStep number : '+str(step_number)+'/'+str(self.dataset.get_train_len())
                string += ' | Total loss : '+str(loss_to_print)+' | Mean loss : '+str(loss_to_print/Config.STEPS_TO_PRINT)
                string += ' | Time : '+str(time.time() - start_time)
                self.print_log(string)
                start_time = time.time()
                loss_to_print = 0

        self.test_loss.append(total_loss)
        self.save_model()

    def train(self):
        pass
