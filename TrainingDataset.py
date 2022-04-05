import torch
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Config import Config
import yfinance as yf
from TimeSeriesDataset import TimeSeriesDataset
from tqdm import tqdm

class TrainingDataset(Dataset):

    def __init__(self):
        self.dataframe = pd.read_csv(Config.HEAD_DATA_PATH)
        self.train_size = Config.TRAIN_SIZE
        self.test_size = Config.TEST_SIZE
        self.x_window_size = Config.INPUT_TIME_SERIES_SIZE
        self.y_window_size = Config.OUTPUT_TIME_SERIES_SIZE
        self.train_batch_size = Config.TRAIN_BATCH_SIZE
        self.test_batch_size = Config.TEST_BATCH_SIZE

        self.left_time_border = pd.to_datetime(Config.START_DATA.replace('-',''), format = '%Y%m%d')
        self.right_time_border = pd.to_datetime(Config.FINAL_DATA.replace('-',''), format = '%Y%m%d')

        self.sectors_to_datasets = dict()

        print('Forming datasets...')

        for ind in tqdm(range(self.dataframe.shape[0])):
            sector = self.dataframe.iloc[ind]['GICS Sector']
            ticker = self.dataframe.iloc[ind]['Symbol']

            if self.sectors_to_datasets.get(sector) is None:
                self.sectors_to_datasets[sector] = dict()
            
            data = np.array(yf.Ticker(ticker).history(period='max')[self.left_time_border:self.right_time_border]['Close'])
            data = np.diff(data) / data[1:]
            separate_ind = int(len(data)*self.train_size)
            data_train = data[:separate_ind]
            data_test = data[separate_ind:]
            self.sectors_to_datasets[sector][ticker] = dict()
            self.sectors_to_datasets[sector][ticker]['Train'] = TimeSeriesDataset(data_train, ticker, sector, self.x_window_size, self.y_window_size)
            self.sectors_to_datasets[sector][ticker]['Test'] = TimeSeriesDataset(data_test, ticker, sector, self.x_window_size, self.y_window_size)

            self.sectors = list(self.sectors_to_datasets.keys())
        
        print('Datasets created!')

    def get_train_len(self):
        return Config.TRAIN_STEPS

    def get_test_len(self):
        return Config.TEST_STEPS

    def get_random_element_from_array(self, array):
        proba = 1.0/len(array)
        return np.random.choice(array, p = [proba]*len(array))

    def get_train_batch(self):
        x = []
        y = []
        for _ in range(self.train_batch_size):
            sector_1 = self.get_random_element_from_array(self.sectors)
            sector_2 = self.get_random_element_from_array(self.sectors)
            ticker_1 = self.get_random_element_from_array(list(self.sectors_to_datasets[sector_1]))
            ticker_2 = self.get_random_element_from_array(list(self.sectors_to_datasets[sector_2]))
            x1, y1 = self.sectors_to_datasets[sector_1][ticker_1]['Train'].get_element()
            x2, y2 = self.sectors_to_datasets[sector_2][ticker_2]['Train'].get_element()
            x.append(self.prepare_object(x1, x2))
            y.append(self.prepare_label(y1, y2))

        return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))

    def get_test_batch(self):
        x = []
        y = []
        
        for _ in range(self.test_batch_size):
            sector_1 = self.get_random_element_from_array(self.sectors)
            sector_2 = self.get_random_element_from_array(self.sectors)
            ticker_1 = self.get_random_element_from_array(list(self.sectors_to_datasets[sector_1]))
            ticker_2 = self.get_random_element_from_array(list(self.sectors_to_datasets[sector_2]))
            x1, y1 = self.sectors_to_datasets[sector_1][ticker_1]['Test'].get_element()
            x2, y2 = self.sectors_to_datasets[sector_2][ticker_2]['Test'].get_element()
            x.append(self.prepare_object(x1, x2))
            y.append(self.prepare_label(y1, y2))

        return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))

    def prepare_label(self, label_1, label_2):
        label_1 = np.mean(label_1)
        label_2 = np.mean(label_2)
        result = None
        if label_2*label_1 <= 0:
            result = 0
        elif abs(label_1 - label_2) >= Config.THRESHOLD_DIFF_BETWEEN_PROFITS:
            result = 0
        else:
            result = 1
        return result

    def prepare_object(self, input_1, input_2):
        return np.array([[input_1], [input_2]])

    def print_info(self, if_print = True):
        final_string = 'Sectors : \n'
        for sector in self.sectors:
            final_string += '\t'+sector+' : \n'
            for key in self.sectors_to_datasets[sector].keys():
                final_string += '\t\t Ticker '+key+'\n\t\t\tTrain size : '+str(len(self.sectors_to_datasets[sector][key]['Train']))+'\n'
                final_string += '\t\t\tTest_size : '+str(len(self.sectors_to_datasets[sector][key]['Test']))+'\n'
            final_string += '\n'
        if if_print:
            print(final_string)
        return final_string
    
    def count_balance(self):
        Y = []
        for step in tqdm(range(self.get_train_len())):
            x, y = self.get_train_batch()
            for el in y:
                Y.append(int(el))
        return np.array(Y).mean()
    
if __name__ == '__main__':

    print('\nTest and info about Training dataaset.\n\n')

    file = open('TestLogs/TrainingDatasetTestLog.txt', 'w')
    obj = TrainingDataset()
    file.write(obj.print_info(if_print = False))
    balance = 'Mean result of labels : '+str(obj.count_balance())+'\n'
    print(balance)
    file.write('\n\n'+balance)
    file.close()

    print('Test confirmed!')



            

