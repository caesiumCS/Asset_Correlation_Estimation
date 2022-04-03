from websockets import Data
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import yfinance as yf

class TrainingDataset(Dataset):

    def __init__(self, dataframe, train_size, test_size , x_window_size, y_window_size, train_batch_size, test_batch_size):
        self.dataframe = dataframe
        self.train_size = train_size
        self.test_size = test_size
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.Sectors_to_datasets = {}
        for ind in range(dataframe.shape[0]):
            sector = dataframe.iloc[ind]['GICS Sector']
            ticker = dataframe.iloc[ind]['Symbol']

    def get_train_len(self):
        pass

    def get_test_len(self):
        pass

    def get_train_batch(self):
        pass

    def get_test_batch(self):
        pass
