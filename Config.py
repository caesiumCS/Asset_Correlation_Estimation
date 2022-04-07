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
    INPUT_TIME_SERIES_SIZE = 255 # Traiding days 
    OUTPUT_TIME_SERIES_SIZE = 30 # Traiding days
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    #THRESHOLD_DIFF_BETWEEN_PROFITS = {0.1:1, 0.2:2, 0.3:3, 0.4:4, 0.5:5, 0.6:6, 0.7:7, 0.8:8, 0.9:9}
    THRESHOLD_DIFF_BETWEEN_PROFITS = 0.11

    '''
    Meta information about experiment
    '''
    EXPERIMENT_NUMBER = 1
    DATE = '07.04.2020'
    EXPERIMENT_FOLDER_NAME = DATE + '_'+str(EXPERIMENT_NUMBER)

    '''
    Model configs
    '''
    MODEL_NAME = 'CNN_Model_1'
    ADDITIONAL_INFO = 'CNN Model, with two inputs. First version. Predict sign.'
    LR = 0.001
    LOSS = 'BCELoss' # field for saving and reading experiment logs

    '''
    Training configs
    '''
    TRAIN_STEPS = 9000 # number of batches for one epoch
    TEST_STEPS = 9000
    EPOCHS = 100
    STEPS_TO_PRINT = 300 # number of steps to print info about learning
    DEVICE = 'cpu'
    