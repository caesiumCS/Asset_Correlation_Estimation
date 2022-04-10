from torch.utils.data import Dataset
import numpy as np
from Config import Config

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
        output = [1]
        for i in range(len(x)):
            output.append(output[-1]*(1+x[i]))
        output = output[:self.x_window_size]
        y = self.timeseries[index+self.x_window_size+1:index+self.x_window_size+1+self.y_window_size]
        y = np.diff(y) / y[1:]
        output_y = [1]
        for i in range(len(y)):
            output_y.append(output_y[-1]*(1+y[i]))
        return output, output_y

    def get_element(self):
        x, y = self.__getitem__(index=self.ind)
        self.ind += 1
        if self.ind == self.__len__():
            self.ind = 0
        return x, y

    def __len__(self):
        return len(self.timeseries) - (self.x_window_size + self.y_window_size) + 1