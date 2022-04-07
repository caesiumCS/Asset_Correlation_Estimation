from torch import nn, sigmoid
import torch

class CNN_TS_Model(nn.Module):

    def __init__(self):
        super(CNN_TS_Model, self).__init__()
        self.conv_1 = nn.Sequential( 
            nn.Conv1d(1, 5, 10),
            nn.ReLU(),
            nn.Conv1d(5, 10, 10), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(10, 15, 10),
            nn.ReLU(),
            nn.Conv1d(15, 20, 10), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(20, 30, 10),
            nn.ReLU(),
            nn.Conv1d(30, 40, 10),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(640*2, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 1),
            nn.Tanh()

        )

    def forward(self, x):
        x_1 = x[:,0,:]
        x_2 = x[:,1,:]
        x_1 = self.conv_1(x_1)
        x_2 = self.conv_1(x_2)
        x_res = torch.cat((x_1.view(x_1.size(0), -1),
                          x_2.view(x_2.size(0), -1)), dim=1)

        x_res = self.head(x_res)
        return x_res

if __name__ == '__main__':
    x = torch.ones([5, 2, 1, 255])
    model = CNN_TS_Model()
    print('Number of parameters in model : '+str(sum(p.numel() for p in model.parameters())))
    print(model(x).shape)
    del model