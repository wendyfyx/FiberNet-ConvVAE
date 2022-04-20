import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_util import *

def init_model(model_type='3Ls', Z=2, SEED=2021):
    '''Initialize model with string name'''
    set_seed(SEED)
    encClass = getattr(importlib.import_module("model.model"), 
                      f"Encoder{model_type}")
    decClass = getattr(importlib.import_module("model.model"), 
                      f"Decoder{model_type}")
    return convVAE(3, Z, encClass, decClass)


def parse_model_setting(model_name):
    '''Parse model name to get model hyperparam info'''
    d_info = {}
    ls = model_name.split("_")
    d_info['model_type'] = ls[0].split('convVAE')[1] # encoder/decoder type
    d_info['conv_init'] = ls[1][:2] # initialization for conv layer weights
    d_info['linear_init'] = ls[1][2:] # initialization for linear layer weights
    d_info[ls[2][:1]] = int(ls[2][1:]) # Z, size of embedding
    d_info[ls[3][:1]] = int(ls[3][1:]) # B, batch size
    d_info[ls[4][:2]] = float(ls[4][2:]) # LR, learning rate
    d_info[ls[5][:2]] = float(ls[5][2:]) # WD, weight decay
    d_info[ls[6][:3]] = float(ls[6][3:]) # GC norm or value
    d_info['subj_train'] = ('_'.join(ls[7:])) # subject model was trained on
    return d_info

def init_weights(m):
    '''Initialize model weight with Xavier uniform'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)

# Block of conv + non-linear + batchnorm + upsampling/pool
class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding, 
                 deconvolution=False):
        super(ConvBlock, self).__init__()
        
        self.deconvolution = deconvolution
        self.conv = nn.Conv1d(channels_in, channels_out, 
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=padding)
        self.actv = nn.ReLU()
        self.bn = nn.BatchNorm1d(channels_out)
        if self.deconvolution:
            # use upsample
            self.sample = nn.Upsample(scale_factor=2, mode='linear', 
                                      align_corners=False)
        else:
            # use avg pool
            self.sample = nn.AvgPool1d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.actv(x)
        x = self.bn(x)
        x = self.sample(x)
        return x


#  Encoder with 3 conv layers (64, 32, 16)
class Encoder3Lxs(nn.Module):
    def __init__(self, channels_in):
        super(Encoder3Lxs, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 16, 64, 1, 32),
            ConvBlock(16, 32, 32, 1, 16),
            ConvBlock(32, 64, 16, 1, 8),
            nn.Flatten(),
        )
        self.linear_dim = 64*32
        
    def forward(self, x):
        return self.encoder(x.permute(0,2,1))
    

#  Decoder with 3 conv layers (15, 31, 63)
class Decoder3Lxs(nn.Module):
    def __init__(self, dims_in, channels_out):
        super(Decoder3Lxs, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(dims_in, 64*32),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            ConvBlock(64, 32, 15, 1, 7, deconvolution=True),
            ConvBlock(32, 16, 31, 1, 15, deconvolution=True),
            ConvBlock(16, channels_out, 63, 1, 31, deconvolution=True)
        )
    
    def forward(self, x):
        out = self.decoder_linear(x).reshape((-1, 64, 32))
        out = self.decoder_conv(out).permute(0,2,1)
        return out


# Encoder with 3 conv layers (128, 64, 32)
class Encoder3Ls(nn.Module):
    def __init__(self, channels_in):
        super(Encoder3Ls, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 32, 128, 1, 64),
            ConvBlock(32, 64, 64, 1, 32),
            ConvBlock(64, 128, 32, 1, 16),
            nn.Flatten(),
        )
        self.linear_dim = 128*32
        
    def forward(self, x):
        return self.encoder(x.permute(0, 2, 1))
    

# Decoder with 3 conv layers (31, 63, 127)
class Decoder3Ls(nn.Module):
    def __init__(self, dims_in, channels_out):
        super(Decoder3Ls, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(dims_in, 128*32),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            ConvBlock(128, 64, 31, 1, 15, deconvolution=True),
            ConvBlock(64, 32, 63, 1, 31, deconvolution=True),
            ConvBlock(32, channels_out, 127, 1, 63, deconvolution=True)
        )
    
    def forward(self, x):
        out = self.decoder_linear(x).reshape((-1, 128, 32))
        out = self.decoder_conv(out).permute(0, 2, 1)
        return out


# Encoder with 3 conv layers (256, 128, 64)
class Encoder3L(nn.Module):
    def __init__(self, channels_in):
        super(Encoder3L, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 32, 256, 1, 128),
            ConvBlock(32, 64, 128, 1, 64),
            ConvBlock(64, 128, 64, 1, 32),
            nn.Flatten()
        )
        self.linear_dim = 128*32

    def forward(self, x):
        return self.encoder(x.permute(0,2,1))


# Decoder with 3 conv layers (63, 127, 255) 
class Decoder3L(nn.Module):
    '''Decoder with 3 layers'''
    def __init__(self, dims_in, channels_out):
        super(Decoder3L, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(dims_in, 128*32),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            ConvBlock(128, 64, 63, 1, 31, deconvolution=True),
            ConvBlock(64, 32, 127, 1, 63, deconvolution=True),
            ConvBlock(32, channels_out, 255, 1, 127, deconvolution=True)
        )
    
    def forward(self, x):
        out = self.decoder_linear(x).reshape((-1, 128, 32))
        out = self.decoder_conv(out).permute(0,2,1)
        return out


# Encoder with 4 conv layers (256, 128, 64, 32)
class Encoder4L(nn.Module):
    def __init__(self, channels_in):
        super(Encoder4L, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(channels_in, 32, 256, 1, 128),
            ConvBlock(32, 64, 128, 1, 64),
            ConvBlock(64, 128, 64, 1, 32),
            ConvBlock(128, 256, 32, 1, 16),
            nn.Flatten()
        )
        self.linear_dim = 128*32
        
    def forward(self, x):
        return self.encoder(x.permute(0, 2, 1))
    

# Decoder with 4 conv layers (31, 63, 127, 255)
class Decoder4L(nn.Module):
    def __init__(self, dims_in, channels_out):
        super(Decoder4L, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(dims_in, 64*64),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            ConvBlock(256, 128, 31, 1, 15, deconvolution=True),
            ConvBlock(128, 64, 63, 1, 31, deconvolution=True),
            ConvBlock(64, 32, 127, 1, 63, deconvolution=True),
            ConvBlock(32, channels_out, 255, 1, 127, deconvolution=True)
        )
    
    def forward(self, x):
        out = self.decoder_linear(x).reshape((-1, 256, 16))
        out = self.decoder_conv(out).permute(0, 2, 1)
        return out


# Encoder with 4 conv layers (5, 256, 128, 64)
class Encoder4Ls(nn.Module):
    def __init__(self, channels_in):
        super(Encoder4Ls, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 5, 1, 2),
            ConvBlock(16, 32, 256, 1, 128),
            ConvBlock(32, 64, 128, 1, 64),
            ConvBlock(64, 128, 64, 1, 32),
            nn.Flatten()
        )
        self.linear_dim = 128*16
        
    def forward(self, x):
        return self.encoder(x.permute(0, 2, 1))
    

# Decoder with 4 conv layers (63, 127, 255, 5)
class Decoder4Ls(nn.Module):
    def __init__(self, dims_in, channels_out):
        super(Decoder4Ls, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(dims_in, 128*16),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            ConvBlock(128, 64, 63, 1, 31, deconvolution=True),
            ConvBlock(64, 32, 127, 1, 63, deconvolution=True),
            ConvBlock(32, 16, 255, 1, 127, deconvolution=True),
            ConvBlock(16, 3, 5, 1, 2, deconvolution=True)
        )
    
    def forward(self, x):
        out = self.decoder_linear(x).reshape((-1, 128, 16))
        out = self.decoder_conv(out).permute(0, 2, 1)
        return out


# 1D Convolutional VAE class with custom encoder & decoder
class convVAE(nn.Module):
    def __init__(self, channels_in, hidden_dim, encoder_cl, decoder_cl):
        super(convVAE, self).__init__()
        
        self.encoder = encoder_cl(channels_in)
        self.mu = nn.Linear(self.encoder.linear_dim, hidden_dim)
        self.logvar = nn.Linear(self.encoder.linear_dim, hidden_dim)
        
        self.decoder = decoder_cl(hidden_dim, channels_in)
        
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.result_dict = {}

    def encode(self, x):
        '''Encoder q(z|x) given input x, generate parameters for q(z|x)'''
        out = self.encoder(x)
        return self.mu(out), self.logvar(out)

    def reparameterize(self, mu, logvar):
        '''
            Reparameterize trick to allow backpropagation through mu, std.
            We can't backpropagate that if we sample mu and std directly.
            z is sampled from q
        '''
        std = torch.exp(logvar/2)
        # q = torch.distributions.Normal(mu, std)
        # z = q.rsample()
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        z = mu + (eps * std) # sampling as if coming from the input space
        return z, std

    def decode(self, z):
        '''Decode given z, get p(x|z) parameters'''
        return self.decoder(z)
    
    def kl_qp_mc(self, z, mu, std):
        '''Compute KL divergence loss (regularization)'''
        
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), 
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)  # we want q to be close to p
        return kl.sum(-1)
    
    def reconstruction(self, x, x_hat):
        '''Compute reconstruction loss, log p(x|z), log Gaussian likelihood'''

        scale = torch.exp(self.log_scale)
        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))
    
    def forward(self, x):
        '''To get output and z, use this for evaluation'''
        
        # encode x, get q(z|x) parameters
        mu, logvar = self.encode(x)
        
        # reparametrize, sample z from q(z|x)
        z, std = self.reparameterize(mu, logvar)

        # decode, get p(x|z) parameters
        return self.decode(z).view(x.size()), z
        
    def loss(self, x, computeMSE=False):
        '''Use this for training'''
        
        # encode x, get q(z|x) parameters
        mu, logvar = self.encode(x)

        # reparametrize, sample z from q(z|x)
        z, std = self.reparameterize(mu, logvar)

        # decode, get p(x|z) parameters
        x_hat = self.decode(z).view(x.size()) 
        
        # MSE Loss
        if computeMSE:
            loss = F.mse_loss(x, x_hat)
            self.result_dict["loss_mse"] = loss.item()
         
        # ELBO Loss
        else: 
            loss_kl = self.kl_qp_mc(z, mu, std)
            loss_recon = self.reconstruction(x, x_hat)
            loss = (loss_kl - loss_recon).mean()

            self.result_dict["loss_kl"] = loss_kl.mean().item()
            self.result_dict["loss_recon"] = -loss_recon.mean().item()
            self.result_dict["loss_elbo"] = loss.item()

        return x_hat, z, loss