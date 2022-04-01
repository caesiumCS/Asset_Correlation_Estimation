from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):

    def __init__(self, timeseries, ticker, sector, x_window_size, y_window_size):
        self.timeseries = timeseries
        self.ticker = ticker
        self.sector = sector
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size

    def __getitem__(self, index):
        x = self.timeseries[index:index+self.x_window_size]
        y = self.timeseries[index+self.x_window_size:index+self.x_window_size+self.y_window_size]
        return x, y

    def __len__(self):
        return len(self.timeseries) - (self.x_window_size + self.y_window_size) + 1