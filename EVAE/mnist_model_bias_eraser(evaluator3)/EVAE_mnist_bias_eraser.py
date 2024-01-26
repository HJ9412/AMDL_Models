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


# Define Evaluator
class Evaluator(nn.Module):
    def __init__(self, data_dim, hidden_dim1, hidden_dim2, value_dim):
        super(Evaluator, self).__init__()
        # Layers
        self.fc_x1h1 = nn.Linear(data_dim, hidden_dim1)
        self.fc_x2h2 = nn.Linear(data_dim, hidden_dim1)
        self.fc_hA = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_hB = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_eval = nn.Linear(hidden_dim2, value_dim)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)
        h1 = F.relu(self.fc_x1h1(x1))               # Layers x1h1, x2h2 makes the evaluation a noncommutative operation
        h2 = F.relu(self.fc_x2h2(x2))
        sA = F.relu(self.fc_hA(h1)+self.fc_hA(h2))  # Applying common layers to h1 and h2 produces intermingled states sA and sB
        sB = F.relu(self.fc_hB(h1)+self.fc_hB(h2))
        values = F.relu(self.fc_eval(sA+sB))        # The last layer even intermingles the two states sA and sB to make the model completely forget the pathway of inputs
        return values


# Define sample displayer
def displayer(sampling, x_len, data_x, data_y, rgb = False):
    fig_x = min(sampling, x_len)
    fig, axes = plt.subplots(2, fig_x, figsize=(fig_x, 2))

    if rgb:
        for i in range(0, fig_x):
            pos = i % fig_x
            image = x[i].permute(1, 2, 0)
            axes[int(i/fig_x)*2][pos].imshow(image.cpu())
            axes[int(i/fig_x)*2][pos].axis('off')

            image = recon_x[i].permute(1, 2, 0)
            axes[int(i/fig_x)*2+1][pos].imshow(image.detach().cpu())
            axes[int(i/fig_x)*2+1][pos].axis('off')
        plt.show()
        plt.close()
    else:
        for i in range(0, fig_x):
            pos = i % fig_x
            image = x[i].view(data_x, data_y)
            axes[int(i/fig_x)*2][pos].imshow(image.cpu(), cmap='gray')
            axes[int(i/fig_x)*2][pos].axis('off')

            image = recon_x[i].view(data_x, data_y)
            axes[int(i/fig_x)*2+1][pos].imshow(image.detach().cpu(), cmap='gray')
            axes[int(i/fig_x)*2+1][pos].axis('off')
        plt.show()
        plt.close()


if __name__ == '__main__':
    # Randomness control
    torch.manual_seed(2022313045)
    np.random.seed(2022313045)

    # Metadata
    dataset_folder = '../Datasets'
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
    VAE_num_epochs = 50
    Eval_num_epochs = 30
    EVAE_num_epochs = 50
    log_interval = 10000

    # Sampling train data
    sample_train_data = False
    sampling_epoch = 10
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
    data_train = MNIST(dataset_folder, train=True, transform=transform)
    data_test = MNIST(dataset_folder, train=False, transform=transform)

    
    # Neural Networks
    model = VAE(latent_dim, channels_A, channels_B, channels_C, channels_D, data_channels).to(device)
    evaluator = Evaluator(data_dim, hidden_dim1, hidden_dim2, value_dim).to(device)

    




    # Training VAE
    optimizer_VAE = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(VAE_num_epochs):
        # Shuffle data
        data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
        print(f"Train Epoch: {epoch+1}")
        for batch_idx, (data, data_label) in enumerate(data_loader):
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
                print(f"[{(batch_idx+1) * batch_size}/{len(data_loader.dataset)}] recon_loss: {recon_loss.item() / len(data):.2f}  kl_loss: {kl_loss.item() / len(data):.2f}  VAE Loss: {VAE_loss.item() / len(data):.2f}")

        if sample_train_data and epoch % sampling_epoch == sampling_epoch - 1:
            displayer(train_sampling, x.shape[0], data_x, data_y, data_is_rgb)
    optimizer_VAE.zero_grad()






    # Training Evaluator
    optimizer_Eval = optim.Adam(evaluator.parameters(), lr=2e-7)

    for epoch in range(Eval_num_epochs):
        # Shuffle data
        data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
        print(f"Train Epoch: {epoch+1}")
        for batch_idx, (data, data_label) in enumerate(data_loader):
            x = data.to(device)
            recon_x, mean, log_var = model(x)
            values = evaluator(x, recon_x)
                
            # Calculate loss
            eval_loss = torch.sum(torch.exp(-values))

            # Backpropagation
            optimizer_Eval.zero_grad()
            eval_loss.backward()
            optimizer_Eval.step()
            
            if (batch_idx+1) * batch_size % log_interval == 0:
                print(f"[{(batch_idx+1) * batch_size}/{len(data_loader.dataset)}] eval_loss: {eval_loss.item() / len(data):.2f}")

        if sample_train_data and epoch % sampling_epoch == sampling_epoch - 1:
            displayer(train_sampling, x.shape[0], data_x, data_y, data_is_rgb)
    optimizer_Eval.zero_grad()




    

    # Training EVAE
    optimizer_EVAE = optim.Adam(list(model.parameters()) + list(evaluator.parameters()), lr=2e-6)

    for epoch in range(EVAE_num_epochs):
        # Shuffle data
        data_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
        print(f"Train Epoch: {epoch+1}")
        for batch_idx, (data, data_label) in enumerate(data_loader):
            x = data.to(device)
            recon_x, mean, log_var = model(x)
            values = evaluator(x, recon_x)

            # Calculate loss
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            eval_loss = torch.sum(torch.exp(-values))
            VAE_loss = recon_loss + kl_loss
            total_loss = VAE_loss + eval_loss

            # Backpropagation
            optimizer_EVAE.zero_grad()
            total_loss.backward()
            optimizer_EVAE.step()

            if (batch_idx+1) * batch_size % log_interval == 0:
                print(f"[{(batch_idx+1) * batch_size}/{len(data_loader.dataset)}] recon_loss: {recon_loss.item() / len(data):.2f}  kl_loss: {kl_loss.item() / len(data):.2f}  VAE_loss: {VAE_loss.item()/len(data):.2f}  eval_loss: {eval_loss.item() / len(data):.2f}")

        if sample_train_data and epoch % sampling_epoch == sampling_epoch - 1:
            displayer(train_sampling, x.shape[0], data_x, data_y, data_is_rgb)
    optimizer_EVAE.zero_grad()






    # Testing
    data_loader = DataLoader(data_test, batch_size = 13)

    if data_is_rgb:
        for data, data_label in data_loader:
            fig, axes = plt.subplots(1, 3, figsize=(6*0.7, 2))
            image_origin = data[0].view(1, 3, data_x, data_y).to(device)
            values = evaluator(image_origin.reshape(1,-1),image_origin.reshape(1,-1))
            value_origin = values.sum().item()
            axes[0].imshow(image_origin[0].permute(1, 2, 0).cpu())
            axes[0].set_title(f'0th: {value_origin:.1f}')
            axes[0].axis('off')

            recon_x = image_origin
            count = 0
            for i in range(1, test_sampling):
                recon_x, mean, log_var = model(recon_x, do_reparam = False)
                image = recon_x[0].detach()
                values = evaluator(recon_x, recon_x)
                value_image = values[0].sum().item()

                if i == 1:
                    value = value_image
                    axes[1].imshow(image.permute(1, 2, 0).cpu())
                    axes[1].set_title(f'1st: {value:.1f}')
                    axes[1].axis('off')
                if value_image >= value and value_image > value_origin:
                    count += 1
                    print(f"{i}, {value_image}, {value}")
                    value = value_image
                    axes[2].imshow(image.permute(1, 2, 0).cpu())
                    axes[2].set_title(f'{i}th: {value:.1f}')
                    axes[2].axis('off')
                    if count == 1:
                        break
            if count == 0:
                axes[2].imshow(image_origin[0].permute(1, 2, 0).cpu())
                axes[2].set_title(f'{0}th: {value_origin:.1f}')
                axes[2].axis('off')
            plt.show()
            plt.close()
    else:
        for data, data_label in data_loader:
            fig, axes = plt.subplots(1, 3, figsize=(6*0.7, 2))
            image = data[0].view(1, 1, data_x, data_y).to(device)
            values = evaluator(image.reshape(1,-1),image.reshape(1,-1))
            image_origin = image[0][0].cpu()
            value_origin = torch.log(values+1).sum().item()
            axes[0].imshow(image_origin, cmap='gray')
            axes[0].set_title(f'0th: {value_origin:.1f}')
            axes[0].axis('off')

            count = 0
            for i in range(1, test_sampling):
                recon_x, mean, log_var = model(image, do_reparam = False)
                values = evaluator(recon_x.reshape(1,-1), recon_x.reshape(1,-1))
                value_image = torch.log(values+1).sum().item()
                image = recon_x.detach()

                if i == 1:
                    value = value_image
                    axes[1].imshow(image[0][0].cpu(), cmap='gray')
                    axes[1].set_title(f'1st: {value_image:.1f}')
                    axes[1].axis('off')

                if value_image >= value and value_image > value_origin:
                    count += 1
                    value = value_image
                    axes[2].imshow(image[0][0].cpu(), cmap='gray')
                    axes[2].set_title(f'{i}th: {value_image:.1f}')
                    axes[2].axis('off')
                    if count == 1:
                        break
            if count == 0:
                axes[2].imshow(image_origin, cmap='gray')
                axes[2].set_title(f'Original')
                axes[2].axis('off')
            plt.show()
            plt.close()
