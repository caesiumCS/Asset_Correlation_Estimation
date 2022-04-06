import pandas as pd
import numpy as np
from Config import Config
import yfinance as yf
from tqdm import tqdm



class HeadDataDownloader:

    def __init__(self):
        self.config = Config()

    def create_head_dataset(self):
        root_data = pd.read_csv(self.config.SandP500_PATH)
        left_time = pd.to_datetime(self.config.START_DATA.replace('-', ''), format='%Y%m%d')
        right_time = pd.to_datetime(self.config.FINAL_DATA.replace('-', ''), format='%Y%m%d')   
        tickers_to_save = []
        print('Creating head dataset file...')
        for ticker in tqdm(np.unique(root_data['Symbol'])):
            x = yf.Ticker(ticker)
            x = x.history(period='max')
            x = x.loc[left_time:right_time]
            total_days = int(str(right_time - left_time).split()[0])
            if x.shape[0] <= total_days and x.shape[0]>=self.config.MINIMUM_DAYS_IN_DATA:
                tickers_to_save.append(ticker)
            
        def f(el):
            return el in tickers_to_save
            
        new_data = root_data[ root_data['Symbol'].apply(f) ]
        new_data.to_csv(self.config.HEAD_DATA_PATH, index=False)
        print('Head dataset created!')

if __name__ == '__main__':
    print('Testing HeadDataDownloader class...')
    obj = HeadDataDownloader()
    obj.create_head_dataset()
    print('Test confirmed.')