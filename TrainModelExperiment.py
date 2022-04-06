from DataDownloader import HeadDataDownloader
from Trainer import Trainer


def start_training():
    #HeadDataDownloader().create_head_dataset()
    Trainer().train()

if __name__ == '__main__':
    start_training()