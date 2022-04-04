from websockets import Data
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Config import Config
import yfinance as yf
from TimeSeriesDataset import TimeSeriesDataset

class TrainingDataset(Dataset):

    def __init__(self, dataframe , train_batch_size, test_batch_size):
        self.dataframe = dataframe
        self.train_size = Config.TRAIN_SIZE
        self.test_size = Config.TEST_SIZE
        self.x_window_size = Config.INPUT_TIME_SERIES_SIZE
        self.y_window_size = Config.OUTPUT_TIME_SERIES_SIZE
        self.train_batch_size = Config.TRAIN_BATCH_SIZE
        self.test_batch_size = Config.TEST_BATCH_SIZE

        self.left_time_border = pd.to_datetime(Config.START_DATA.replace('-',''), format = '%Y%m%d')
        self.right_time_border = pd.to_datetime(Config.FINAL_DATA.replace('-',''), format = '%Y%m%d')

        self.sectors_to_datasets = dict()
        for ind in range(dataframe.shape[0]):
            sector = dataframe.iloc[ind]['GICS Sector']
            ticker = dataframe.iloc[ind]['Symbol']

            if self.sectors_to_datasets.get(sector) is None:
                self.sectors_to_datasets[sector] = dict()
            
            data = np.array(yf.Ticker(ticker).history(period='max')[self.left_time_border:self.right_time_border]['Close'])
            data = np.diff(data) / data[:,1:]
            separate_ind = int(len(data)*self.train_size)
            data_train = data[:separate_ind]
            data_test = data[separate_ind:]
            self.sectors_to_datasets[sector][ticker]['Train'] = TimeSeriesDataset(data_train, ticker, sector, self.x_window_size, self.y_window_size)
            self.sectors_to_datasets[sector][ticker]['Test'] = TimeSeriesDataset(data_test, ticker, sector, self.x_window_size, self.y_window_size)

    def get_train_len(self):
        return self.train_size

    def get_test_len(self):
        return self.test_size

    def get_train_batch(self):
        pass

    def get_test_batch(self):
        pass

    def prepare_label(self, label_1, label_2):
        pass

    def prepare_object(self, input_1, input_2):
        pass

    def print_info(self):
        pass
