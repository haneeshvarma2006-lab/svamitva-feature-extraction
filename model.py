import torch
import torch.nn as nn

class VillageModel(nn.Module):

    def __init__(self):
        super(VillageModel,self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32,16,2,stride=2),
            nn.ReLU(),

            nn.Conv2d(16,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x