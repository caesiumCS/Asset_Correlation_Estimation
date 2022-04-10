from torch import nn, sigmoid
import torch

class CNN_TS_Model(nn.Module):

    def __init__(self):
        super(CNN_TS_Model, self).__init__()
        self.conv_1 = nn.Sequential( 
            nn.Conv1d(1, 5, 30),
            nn.ReLU(),
            nn.Conv1d(5, 10, 30), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(10, 15, 30),
            nn.ReLU(),
            nn.Conv1d(15, 20, 30), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(20, 30, 10),
            nn.ReLU(),
            nn.Conv1d(30, 40, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(160*2, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 1),
            #nn.Tanh()

        )

    def forward(self, x):
        x_1 = x[:,0,:]
        x_2 = x[:,1,:]
        x_1 = self.conv_1(x_1)
        x_2 = self.conv_1(x_2)
        x_res = torch.cat((x_1.view(x_1.size(0), -1),
                          x_2.view(x_2.size(0), -1)), dim=1)
        x_res = self.head(x_res)

        #x_1 = self.head(x_1)
        #x_2 = self.head(x_2)
        #a = torch.sum((x_1 - x_1.mean(1).unsqueeze(1))*(x_2 - x_2.mean(1).unsqueeze(1)), dim = -1)
        #b = torch.pow(torch.sum((x_1 - x_1.mean(1).unsqueeze(1)).pow(2), dim = -1)*torch.sum((x_2 - x_2.mean(1).unsqueeze(1)).pow(2), dim = -1), 0.5)
        #x_res = a/b
        return x_res

if __name__ == '__main__':
    x = torch.ones([5, 2, 1, 255])
    model = CNN_TS_Model()
    print('Number of parameters in model : '+str(sum(p.numel() for p in model.parameters())))
    print(model(x).shape)
    del model