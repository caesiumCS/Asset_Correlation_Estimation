from DataDownloader import HeadDataDownloader
from Trainer import Trainer
import warnings


def start_training():
    warnings.filterwarnings('ignore')
    #HeadDataDownloader().create_head_dataset()
    Trainer().train()

if __name__ == '__main__':
    start_training()