import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

features = 16  # Latent resolution


class LinearVAE(nn.Module):

    def __init__(self):
        super(LinearVAE, self).__init__()

        # Encoder

        # MNIST images are 28*28*1 (grayscale).
        self.encoder1 = nn.Linear(in_features=28*28, out_features=512)
        self.encoder2 = nn.Linear(in_features=512, out_features=features*2)

        # Decoder
        self.decoder1 = nn.Linear(in_features=features, out_features=512)
        self.decoder2 = nn.Linear(in_features=512, out_features=28*28)

    def reparameterise(self, mu, log_var):
        '''
        :param mu: mean of the distribution from the encoder's latent space.
        :param log_var: log of the variance from the encoder's latent space.
        '''

        # Note: exp(log(var)) = var, and exp(0.5*log_var) = sqrt(var) = std
        std = torch.exp(0.5*log_var)  # Standard deviation
        epsilon = torch.randn_like(std)  # Random vector, same size as std
        sample = mu + (epsilon * std)
        return sample

    def forward(self, x):

            # Encoding
            x = self.encoder1(x)
            x = F.relu(x)
            x = self.encoder2(x)
            print(x.shape)
            x = x.view(-1, 2, features)  # -1 means "Figure it out".

            # Derivation of the code below is not entirely clear yet.
            mu = x[:, 0, :]  # The first feature values are the mean.
            log_var = x[:, 1, :]  # The second feature values are the variance.

            z = self.reparameterise(mu, log_var)

            # Decoding
            x = self.decoder1(z)
            x = F.relu(x)
            x = self.decoder2(x)

            reconstruction = torch.sigmoid(x)
            return reconstruction, mu, log_var


