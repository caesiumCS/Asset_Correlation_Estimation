from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):

    def __init__(self, timeseries, ticker, sector, x_window_size, y_window_size):
        self.timeseries = timeseries
        self.ticker = ticker
        self.sector = sector
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.ind = 0

    def __getitem__(self, index):
        x = self.timeseries[index:index+self.x_window_size+1]
        x = np.diff(x) / x[1:]
        x[np.isnan(x)] = 0
        y = self.timeseries[index+self.x_window_size+1:index+self.x_window_size+self.y_window_size+1]
        return x, y

    def get_element(self):
        x, y = self.__getitem__(index=self.ind)
        self.ind += 1
        if self.ind == self.__len__():
            self.ind = 0
        return x, y

    def __len__(self):
        return len(self.timeseries) - (self.x_window_size + self.y_window_size) + 1