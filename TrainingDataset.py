from websockets import Data
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Config import Config
import yfinance as yf
from TimeSeriesDataset import TimeSeriesDataset

class TrainingDataset(Dataset):

    def __init__(self, dataframe, train_size, test_size , x_window_size, y_window_size, train_batch_size, test_batch_size):
        self.dataframe = dataframe
        self.train_size = train_size
        self.test_size = test_size
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.left_time_border = pd.to_datetime(Config.START_DATA.replace('-',''), format = '%Y%m%d')
        self.right_time_border = pd.to_datetime(Config.FINAL_DATA.replace('-',''), format = '%Y%m%d')

        self.sectors_to_datasets = dict()
        for ind in range(dataframe.shape[0]):
            sector = dataframe.iloc[ind]['GICS Sector']
            ticker = dataframe.iloc[ind]['Symbol']
            if self.sectors_to_datasets.get(sector) is None:
                self.sectors_to_datasets[sector] = dict()
            data = np.array(yf.Ticker(ticker).history(period='max')[self.left_time_border:self.right_time_border]['Close'])
            self.sectors_to_datasets[sector][ticker] = TimeSeriesDataset(data, ticker, sector, x_window_size, y_window_size)

    def get_train_len(self):
        pass

    def get_test_len(self):
        pass

    def get_train_batch(self):
        pass

    def get_test_batch(self):
        pass
