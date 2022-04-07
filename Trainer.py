from http.client import ImproperConnectionState
from Config import Config
from Model import CNN_TS_Model
from TrainingDataset import TrainingDataset
from torch import nn
import os
import torch
from time import time
import shutil
import pickle
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self):
        
        self.train_loss = []
        self.test_loss = []
        
        # TODO add accuracy history and plots for them

        self.model = CNN_TS_Model()
        self.dataset = TrainingDataset()
        
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), Config.LR)

        self.experiment_path = 'Experiments/'+Config.EXPERIMENT_FOLDER_NAME

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
        else:
            raise FileExistsError('This Folder is already created!')
        os.makedirs(self.experiment_path+'/Logs')
        os.makedirs(self.experiment_path+'/Binary_files_and_models')
        os.makedirs(self.experiment_path+'/Codes')
        os.makedirs(self.experiment_path+'/Plots')

        shutil.copyfile('Config.py', self.experiment_path+'Codes/Config.py')
        shutil.copyfile('Model.py', self.experiment_path+'Codes/Model.py')
        shutil.copyfile('TrainingDataset.py', self.experiment_path+'Codes/TrainingDataset.py')
        shutil.copyfile('TimeSeriesDataset.py', self.experiment_path+'Codes/TimeSeriesDataset.py')
        shutil.copyfile('Trainer.py', self.experiment_path+'Codes/Trainer.py')

        data_info_string = str(self.dataset.count_balance()) + '\n\n' + self.dataset.print_info(if_print = False)
        file = open(self.experiment_path+'/Logs/dataset_info.txt', 'w')
        file.write(data_info_string)
        file.close()
        del data_info_string

        self.logs = open(self.experiment_path+'/Logs/logx.txt', 'w')
        with open(self.experiment_path+'/Binary_files_and_models/dataset.pickle', 'wb') as handle:
            pickle.dump(self.dataset, handle)

    def save_model(self):
        torch.save(self.model.to('cpu'), self.experiment_path+'/Binary_files_and_models/last_checkpoint.trch')
        if len(self.test_loss) < 2:
            torch.save(self.model.to('cpu'), self.experiment_path+'/Binary_files_and_models/best_checkpoint.trch')
        else:
            if self.test_loss[-1] < self.test_loss[-2]:
                torch.save(self.model.to('cpu'), self.experiment_path+'/Binary_files_and_models/best_checkpoint.trch')
        self.model.to(Config.DEVICE)

    def print_log(self, string):
        self.logs.write(string + '\n')
        print(string)
    
    def train_one_epoch(self, epoch_number):
        self.print_log('\nStart training epoch number '+str(epoch_number)+'\n')
        self.model.train()
        total_loss = 0
        loss_to_print = 0
        start_time = time()
        for step_number in range(1, self.dataset.get_train_len()+1):
            self.optimizer.zero_grad()
            x, y = self.dataset.get_train_batch()
            x = x.nan_to_num(0)
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE)
            pred = self.model(x).to(Config.DEVICE)
            loss = self.criterion(pred.squeeze(), y)
            total_loss += loss.item()
            loss_to_print += loss.item()
            loss.backward()
            self.optimizer.step()

            if step_number%Config.STEPS_TO_PRINT == 0:
                string = '\tStep number : '+str(step_number)+'/'+str(self.dataset.get_train_len())
                string += ' | Total loss : '+str(loss_to_print)+' | Mean loss : '+str(loss_to_print/Config.STEPS_TO_PRINT)
                string += ' | Time : '+str(time() - start_time)
                self.print_log(string)
                start_time = time()
                loss_to_print = 0

        self.train_loss.append(total_loss)

    def test_one_epoch(self, epoch_number):
        self.print_log('\nStart testing epoch number '+str(epoch_number)+'\n')
        self.model.eval()
        total_loss = 0
        loss_to_print = 0
        start_time = time()
        for step_number in range(1, self.dataset.get_test_len()+1):
            
            with torch.no_grad():
                x, y = self.dataset.get_test_batch()
                x = x.nan_to_num(0)
                x = x.to(Config.DEVICE)
                y = y.to(Config.DEVICE)
                pred = self.model(x).to(Config.DEVICE)
                loss = self.criterion(pred.squeeze(), y)
                total_loss += loss.item()
                loss_to_print += loss.item()

            if step_number%Config.STEPS_TO_PRINT == 0:
                string = '\tStep number : '+str(step_number)+'/'+str(self.dataset.get_test_len())
                string += ' | Total loss : '+str(loss_to_print)+' | Mean loss : '+str(loss_to_print/Config.STEPS_TO_PRINT)
                string += ' | Time : '+str(time() - start_time)
                self.print_log(string)
                start_time = time()
                loss_to_print = 0

        self.test_loss.append(total_loss)
        self.save_model()

    def create_loss_plots(self):
        plt.figure(figsize=(19, 7))
        plt.title('Train and test loss history', font = 'Times New Roman')
        plt.plot(self.train_loss, label = 'Train loss')
        plt.plot(self.test_loss, label = 'Test loss')
        plt.legend()
        plt.savefig(self.experiment_path+'/Plots/Train&Test_loss_plot.pdf')


    def train(self):
        print('\nStart training...')
        try:
            self.model.to(Config.DEVICE)
            for epoch in range(1, Config.EPOCHS+1):
                self.train_one_epoch(epoch)
                self.test_one_epoch(epoch)
            with open(self.experiment_path+'/Binary_files_and_models/train_loss_history.pickle', 'wb') as handle:
                pickle.dump(self.train_loss, handle)
            with open(self.experiment_path+'/Binary_files_and_models/test_loss_history.pickle', 'wb') as handle:
                pickle.dump(self.test_loss, handle)
            self.create_loss_plots()
        except KeyboardInterrupt:
            self.print_log('Experiment stoped by user!')
            with open(self.experiment_path+'/Binary_files_and_models/train_loss_history.pickle', 'wb') as handle:
                pickle.dump(self.train_loss, handle)
            with open(self.experiment_path+'/Binary_files_and_models/test_loss_history.pickle', 'wb') as handle:
                pickle.dump(self.test_loss, handle)
            self.create_loss_plots()

