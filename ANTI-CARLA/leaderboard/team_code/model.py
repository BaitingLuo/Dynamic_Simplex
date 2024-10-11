import torch as torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import resnet152

latent_dim = 32
variational_beta = 0.7

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        # Use resnet152 as the core network. Then remove last fc layer and our own fc layer.
        resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(in_features=resnet.fc.in_features, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=resnet.fc.in_features, out_features=self.latent_dim)

    def forward(self, x):
        x = self.resnet(x)
        
        x = x.view(x.size()[0], -1)
        x_mu = self.fc_mu(x)
        x_log_var = self.fc_logvar(x)
        return x_mu, x_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=128*4*4)
        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size()[0], 128, 4, 4)
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = F.leaky_relu(self.batchnorm4(self.conv4(x)))
        
        x = torch.sigmoid(self.conv5(x))
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_reconstruct = self.decoder(latent)

        return x_reconstruct, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
            #return mu
def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), reduction='sum')
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #print("recon_loss, KL loss:",kldivergence)
    return recon_loss + variational_beta * kldivergence
