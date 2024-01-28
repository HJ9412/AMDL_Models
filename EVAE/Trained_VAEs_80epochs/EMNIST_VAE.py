import torch
import numpy as np
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, EMNIST, FashionMNIST, EMNIST
from torchvision import transforms


# Define CNN encoder
class Encoder(nn.Module):
    def __init__(self, data_channels, channels_A, channels_B, channels_C, channels_D, latent_dim):
        super(Encoder, self).__init__()
        # Layers
        self.conv_dataA = nn.Conv2d(data_channels, channels_A, kernel_size=5, stride=2)
        self.conv_AB    = nn.Conv2d(channels_A, channels_B, kernel_size=3, stride=1)
        self.conv_BC    = nn.Conv2d(channels_B, channels_C, kernel_size=5, stride = 1)
        self.conv_CD    = nn.Conv2d(channels_C, channels_D, kernel_size=3, stride = 1)
        self.fc_Dmean   = nn.Linear(channels_D * 16, latent_dim)
        self.fc_Dlog_var= nn.Linear(channels_D * 16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv_dataA(x))
        x = F.relu(self.conv_AB(x))
        x = F.relu(self.conv_BC(x))
        x = F.relu(self.conv_CD(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_Dmean(x)
        log_var = self.fc_Dlog_var(x)
        return mu, log_var


# Define CNN decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, channels_D, channels_C, channels_B, channels_A, data_channels):
        super(Decoder, self).__init__()
        self.channels_D = channels_D
        # Layers
        self.fc_LatentD = nn.Linear(latent_dim, channels_D * 16)
        self.conv_tDC   = nn.ConvTranspose2d(channels_D, channels_C, kernel_size=3, stride=1)
        self.conv_tCB   = nn.ConvTranspose2d(channels_C, channels_B, kernel_size=5, stride=1)
        self.conv_tBA   = nn.ConvTranspose2d(channels_B, channels_A, kernel_size=3, stride=1)
        self.conv_tAdata= nn.ConvTranspose2d(channels_A, data_channels, kernel_size=5, stride=2, output_padding=1)

    def forward(self, z):
        z = F.relu(self.fc_LatentD(z))
        z = z.view(-1, self.channels_D, 4, 4)
        z = F.relu(self.conv_tDC(z))
        z = F.relu(self.conv_tCB(z))
        z = F.relu(self.conv_tBA(z))
        recon = torch.sigmoid(self.conv_tAdata(z))
        return recon


# Define VAE
class VAE(nn.Module):
    def __init__(self, latent_dim, channels_A, channels_B, channels_C, channels_D, data_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(data_channels, channels_A, channels_B, channels_C, channels_D, latent_dim)
        self.decoder = Decoder(latent_dim, channels_D, channels_C, channels_B, channels_A, data_channels)

    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def forward(self, x, do_reparam = True):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var) if do_reparam else mu
        return self.decoder(z), mu, log_var


if __name__ == '__main__':
    # Metadata
    dataset_folder = '../../Datasets'
    data_is_rgb = False
    data_channels = 3 if data_is_rgb else 1
    data_x = 28
    data_y = 28
    latent_dim = 40
    channels_D = 40*data_channels
    channels_C = 27*data_channels
    channels_B = 18*data_channels
    channels_A = 12*data_channels
    data_dim = data_x*data_y*data_channels
    hidden_dim1 = 400
    hidden_dim2 = 200
    value_dim = 100

    # Training parameters
    batch_size = 25
    VAE_num_epochs = 80
    log_interval = 30000

    # Sampling train data
    sample_train_data = False
    sampling_epoch = 50
    train_sampling = 30
    test_sampling = 5

    # Set computing device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    
        # Datasets
    data_train = EMNIST(dataset_folder, train=True, split='byclass', transform=transform)

    
    # Neural Networks
    model = VAE(latent_dim, channels_A, channels_B, channels_C, channels_D, data_channels).to(device)



    # Training VAE
    optimizer_VAE = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(VAE_num_epochs):
        # Shuffle data
        data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
        print(f"Train Epoch: {epoch+1}")
        for batch_idx, (data, data_label) in enumerate(data_loader):  # Assuming x is a batch from your data
            x = data.to(device)
            recon_x, mean, log_var = model(x)

            # Calculate loss
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            VAE_loss = recon_loss + kl_loss

            # Backpropagation
            optimizer_VAE.zero_grad()
            VAE_loss.backward()
            optimizer_VAE.step()

            if (batch_idx+1) * batch_size % log_interval == 0:
                print(f"[{(batch_idx+1) * batch_size}/{len(data_loader.dataset)}] recon_loss: {recon_loss.item() / len(data):.1f}  kl_loss: {kl_loss.item() / len(data):.2f}  VAE Loss: {VAE_loss.item() / len(data):.1f}")

        if sample_train_data and epoch % sampling_epoch == sampling_epoch - 1:
            displayer(train_sampling, x.shape[0], data_x, data_y, data_is_rgb)
    optimizer_VAE.zero_grad()

    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_VAE_state_dict':optimizer_VAE.state_dict()
                }, './EMNIST_VAE_80epochs_trainable')