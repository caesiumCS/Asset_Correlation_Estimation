class Config:

    '''
    Config for creating head data to create datasets for testing.
    '''
    START_DATA = '1970-01-01'
    FINAL_DATA = '2015-01-01'
    MINIMUM_DAYS_IN_DATA = 255 * 10 # 255 trading days in year
    HEAD_DATA_PATH = 'Data/head_data.csv'
    SandP500_PATH = 'Data/S&P500_companies.csv'

    '''
    Config for train and test dataset
    '''
    TRAIN_SIZE = 0.8
    TEST_SIZE = 1 - TRAIN_SIZE
    INPUT_TIME_SERIES_SIZE = 90 # Traiding days 
    OUTPUT_TIME_SERIES_SIZE = 30 # Traiding days
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    THRESHOLD_DIFF_BETWEEN_PROFITS = 0.2
    

    '''
    Model configs
    '''

    '''
    Training configs
    '''
    TRAIN_STEPS = 5000
    TEST_STEPS = 500