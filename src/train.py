import torch
import os
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model  # Our model

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import gc

# matplotlib.style.use('ggplot')

# Construct argument praser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, help='Amount of epochs for training of the VAE')
args = vars(parser.parse_args())

# Hyperparameters
epochs = args['epochs']
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

# Train and validation data
train_dataset = datasets.MNIST(
    root=os.getcwd()+'\\input\\data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=os.getcwd()+'\\input\\data',
    train=False,
    download=True,
    transform=transform
)

# Training and validation data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Initialisation
model = model.LinearVAE().to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')


def final_loss(bce_loss, mu, logvar):
    '''
    Get the complete loss (BDC + KLD)

    :param bce_loss: Loss value of the reconstruction.
    :param mu: Mean from the latent vector.
    :param logvar: Log variance from the latent vector.
    :return: bce_loss + KL-Divergence
    '''

    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0

    # tqdm show a progress bar for for-loops.
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_dataset)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)

        optimiser.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()  # Convert from tensor to Python number.

        loss.backward()  # Backpropagate.
        optimiser.step()

    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def test(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_dataset)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)

            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()  # Convert from tensor to Python number.

            # Save the last batch input and output of every epoch.
            if i == int(len(test_dataset)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), f"{os.getcwd()}\\outputs\\output{epoch}.png", nrow=num_rows)
        test_loss = running_loss/len(dataloader.dataset)
        return test_loss


# This is where the magic happens.
train_loss = []
test_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, train_loader)
    test_epoch_loss = test(model, train_loader)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    print(f"Train loss: {train_epoch_loss:4f}")
    print(f"Test loss: {test_epoch_loss:4f}")
    gc.collect()

# Save the model
torch.save(model.state_dict(), f"{os.getcwd()}\\outputs\\sparse_ae{epochs}.pth")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='blue', label='train loss')
plt.plot(test_loss, color='orange', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{os.getcwd()}\\outputs\\loss.png')
plt.show()
