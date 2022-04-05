from torch import nn, sigmoid
import torch

class CNN_TS_Model(nn.Module):

    def __init__(self):
        super(CNN_TS_Model, self).__init__()
        self.conv = nn.Sequential( 
            nn.Conv1d(1, 5, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(5, 15, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(15, 30, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(240*2, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_1 = x[:,0,:]
        x_2 = x[:,1,:]
        x_1 = self.conv(x_1)
        x_2 = self.conv(x_2)
        x_res = torch.cat((x_1.view(x_1.size(0), -1),
                          x_2.view(x_2.size(0), -1)), dim=1)

        x_res = self.head(x_res)
        return x_res

if __name__ == '__main__':
    x = torch.ones([5, 2, 1, 90])
    model = CNN_TS_Model()
    print('Number of parameters in model : '+str(sum(p.numel() for p in model.parameters())))
    print(model(x).shape)
    del model