import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.first_4=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features=latent_size+3,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=253)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.last_4=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.utils.weight_norm(nn.Linear(in_features=512,out_features=512)),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=512,out_features=1)
        )

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x=self.first_4(x_in)
        x=torch.cat((x,x_in),dim=1)
        x=self.last_4(x)
        return x
