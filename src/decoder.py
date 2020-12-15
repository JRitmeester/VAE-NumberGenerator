import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
features = 16


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder1 = nn.Linear(in_features=features, out_features=512)
        self.decoder2 = nn.Linear(in_features=512, out_features=28 * 28)
        self.load_state_dict(
            torch.load(os.getcwd() + '\\outputs\\decoder20.pth'))  # Load the decoder weights learned from model.py

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

    def generate(self, amount):
        '''
        So far I have run this from Python console. Run:
        from src.decoder import Decoder
        d = Decoder()
        d.generate(amount)
        '''

        for i in range(amount):
            random = torch.randn((16, 2))
            outcome = self.forward(random)
            # for result in outcome:
            plt.figure()
            plt.set_cmap('Greys')
            img = outcome[0].view(28, 28)
            plt.imshow(img.detach().numpy())
