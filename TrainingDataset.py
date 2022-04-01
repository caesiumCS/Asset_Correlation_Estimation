from websockets import Data
from TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TrainingDataset(Dataset):

    def __init__(self, dataframe, train_size, test_size , x_window_size, y_window_size):
        self.dataframe = dataframe
        self.train_size = train_size
        self.test_size = test_size
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
