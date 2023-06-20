import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.encoder1=nn.Sequential(
            nn.Conv3d(in_channels=2,out_channels=self.num_features,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.encoder2=nn.Sequential(
            nn.Conv3d(in_channels=self.num_features,out_channels=2*self.num_features,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm3d(2*self.num_features),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.encoder3=nn.Sequential(
            nn.Conv3d(in_channels=2*self.num_features,out_channels=4*self.num_features,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm3d(4*self.num_features),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.encoder4=nn.Sequential(
            nn.Conv3d(in_channels=4*self.num_features,out_channels=8*self.num_features,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm3d(8*self.num_features),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # TODO: 2 Bottleneck layers
        self.bottleneck=nn.Sequential(
            nn.Linear(in_features=self.num_features*8,out_features=self.num_features*8),
            nn.ReLU(),
            nn.Linear(in_features=self.num_features*8,out_features=self.num_features*8),
            nn.ReLU()
        )

        # TODO: 4 Decoder layers
        self.decoder1=nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features*8*2,out_channels=self.num_features*4,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm3d(self.num_features*4),
            nn.ReLU()
        )
        self.decoder2=nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features*4*2,out_channels=self.num_features*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm3d(self.num_features*2),
            nn.ReLU()
        )
        self.decoder3=nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features*2*2,out_channels=self.num_features*1,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm3d(self.num_features*1),
            nn.ReLU()
        )
        self.decoder4=nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features*1*2,out_channels=1,kernel_size=4,stride=2,padding=1),
        )
        

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1=self.encoder1(x)
        x_e2=self.encoder2(x_e1)
        x_e3=self.encoder3(x_e2)
        x_e4=self.encoder4(x_e3)
        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x_d1=self.decoder1(torch.cat((x,x_e4),dim=1))
        x_d2=self.decoder2(torch.cat((x_d1,x_e3),dim=1))
        x_d3=self.decoder3(torch.cat((x_d2,x_e2),dim=1))
        x=self.decoder4(torch.cat((x_d3,x_e1),dim=1))

        x = torch.squeeze(x, dim=1)

        # TODO: Log scaling
        x=torch.log(torch.abs(x)+1)

        return x
