from __future__ import print_function, division
import torch
#import torch.nn.sigmoid as sigmoid
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.pre_latent = 200
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(num_channels, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, self.pre_latent)
        self.fc3 = nn.Linear(n_categories, 1000)
        self.mu = nn.Linear(self.pre_latent, self.latent_dim)
        self.var = nn.Linear(self.pre_latent, self.latent_dim)

    def forward(self, x, labels):
        batch_size = x.size(0)
        #print(x.size())
        x = x.view(batch_size, 1, self.input_dim, self.input_dim)
        #print(x.size())
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * self.input_dim * self.input_dim)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # latent parameters
        mean = self.mu(x)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(x)
        # log_var is of shape [batch_size, latent_dim]
        return mean, log_var

    def update_y_layer(self, new_layer):
        self.fc3 = new_layer
        
        



class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        self.z_dim = latent_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.fc2 = nn.Linear(n_categories, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * self.input_dim * self.input_dim)
        # self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, num_channels, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, self.input_dim, self.input_dim)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x

    def update_y_layer(self, new_layer):
        self.fc2 = new_layer


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''

    def __init__(self, input_dim, hidden_dim, latent_dim, n_categories,num_channels, device):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_categories: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, n_categories,num_channels, device).to(device)

    def forward(self, x, y):
        # x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x, y)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(x_sample, y)
        # generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

    def update_y_layers(self, encoder, decoder):
        self.encoder.update_y_layer(encoder)
        self.decoder.update_y_layer(decoder)

    def encode_then_decode_without_randomness(self, x, y):

        # encode
        z_mu, z_var = self.encoder(x,y)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)*0
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(x_sample, y)
        # generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

    def loss(self,x, reconstructed_x, mean, log_var):
        # reconstruction loss
        RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
        # kl divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        #print(RCL)
        #print(KLD,"\n")
        return RCL + KLD

    def generate random_sample(self,x, reconstructed_x, mean, log_var):

    z = torch.randn(1, LATENT_DIM).to(device)
    y_1d = cat_tensor_list[y_int]
    # because wierd labels for EMNIST

    y = idx2onehot(y_1d, n=N_CLASSES_OLD).to(device, dtype=z.dtype)
    # z = torch.cat((z, y), dim=1)

    reconstructed_img = old_MINIST_model.decoder(z, y)

    y = idx2onehot(y_1d, n=N_CLASSES).to(device, dtype=z.dtype)
    # print(y)
    img = reconstructed_img.view(28, 28).data