import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math


class ChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction):
        super(ChannelAttention, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._max_pool = nn.AdaptiveMaxPool2d(1)
        self._fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avgOut = self._fc(self._avg_pool(x).view(b, c))  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        maxOut = self._fc(self._max_pool(x).view(b, c))  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self._sigmoid(avgOut + maxOut).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        out = x * y.expand_as(x)  # [B, C, H, W]
        return out


class SpatialAttention(nn.Module):
    def __init__(self,
                 kernel_size):
        super(SpatialAttention, self).__init__()
        self._conv = nn.Conv2d(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=False)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, h, w = x.size()
        avgOut = torch.mean(x, dim=1, keepdim=True)  # [B, C, H, W] -> [B, 1, H, W]
        maxOut, _ = torch.max(x, dim=1, keepdim=True)  # [B, C, H, W] -> [B, 1, H, W]
        y = torch.cat([avgOut, maxOut], dim=1)  # [B, 2, H, W]
        y = self._sigmoid(self._conv(y))  # [B, 2, H, W] -> [B, 1, H, W]
        out = x * y.expand_as(x)  # [B, C, H, W]
        return out


class CBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction,
                 kernel_size):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(in_channels, reduction)
        self.SpatialAtt = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ChannelAtt(x)  # [B, C, H, W]
        x = self.SpatialAtt(x)  # [B, C, H, W]
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels):
        super(ResidualBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hid_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_channels,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class DownSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self._block(x)


class UpSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self._block(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class NonLocalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels):
        super(NonLocalBlock, self).__init__()

        self.hid_channels = hid_channels
        self._conv_theta = nn.Conv2d(in_channels=in_channels,
                                     out_channels=hid_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_phi = nn.Conv2d(in_channels=in_channels,
                                   out_channels=hid_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_g = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self._soft_max = nn.Softmax(dim=1)
        self._conv_mask = nn.Conv2d(in_channels=hid_channels,
                                    out_channels=in_channels,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # [B, C, H, W] -> [B, C', HW] -> [B, HW, C']
        theta = self._conv_theta(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, C, H, W] -> [B, C', HW]
        phi = self._conv_phi(x).view(b, self.hid_channels, -1)
        # [B, C, H, W] -> [B, C', HW] -> [B, HW, C']
        g = self._conv_g(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, HW, C'] * [B, C', HW] = [B, HW, HW]
        mul_theta_phi = self._soft_max(torch.matmul(theta, phi))
        # [B, HW, HW] * [B, HW, C'] = [B, HW, C']
        mul_theta_phi_g = torch.matmul(mul_theta_phi, g)
        # [B, HW, C'] -> [B, C', HW] -> [B, C', H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.hid_channels, h, w)
        # [B, C', H, W] -> [B, C, H, W]
        mask = self._conv_mask(mul_theta_phi_g)

        return x + mask


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups):
        super(Encoder, self).__init__()

        # [B, C, H, W] -> [B, C', H, W]
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels_1,
                                 kernel_size=3, stride=1, padding=1)

        # [B, C', H, W] -> [B, C'', h, w]
        self._down_samples = nn.ModuleList()
        for i in range(down_samples):
            cur_in_channels = hid_channels_1 if i == 0 else hid_channels_2
            self._down_samples.append(
                ResidualBlock(in_channels=cur_in_channels,
                              hid_channels=cur_in_channels // 2)
            )
            self._down_samples.append(
                DownSampleBlock(in_channels=cur_in_channels,
                                out_channels=hid_channels_2)
            )

        # [B, C'', h, w] -> [B, C'', h, w]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)

        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_2)
        self._swish = Swish()

        # [B, C'', h, w] -> [B, n, h, w]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_2,
                                 out_channels=out_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self._conv_1(x)

        for layer in self._down_samples:
            x = layer(x)

        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 up_samples,
                 num_groups):
        super(Decoder, self).__init__()

        # [B, n, h, w] -> [B, C'', h, w]
        self._conv_1 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=hid_channels_2,
                                 kernel_size=3, stride=1, padding=1)

        # [B, C'', h, w] -> [B, C'', h, w]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)

        # [B, C'', h, w] -> [B, C', H, W]
        self._up_samples = nn.ModuleList()
        for i in range(up_samples):
            cur_in_channels = hid_channels_2 if i == 0 else hid_channels_1
            self._up_samples.append(
                ResidualBlock(in_channels=cur_in_channels,
                              hid_channels=cur_in_channels // 2)
            )
            self._up_samples.append(
                UpSampleBlock(in_channels=cur_in_channels,
                              out_channels=hid_channels_1)
            )

        # [B, C', H, W] -> [B, C', H, W]
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_1)
        self._swish = Swish()

        # [B, C', H, W] -> [B, C, H, W]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_1,
                                 out_channels=in_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self._conv_1(x)

        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        for layer in self._up_samples:
            x = layer(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        return x


class LISTA(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 h,
                 w,
                 device):
        super(LISTA, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        one = torch.ones(h, w)
        one = torch.unsqueeze(one, 0)
        one = torch.unsqueeze(one, -1)  # [1, h, w, 1]
        self._alpha = nn.Parameter(one)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        self._num_iters = num_iters

    def initialize_dct_weights(self):
        weights = torch.zeros(self._num_atoms, self._num_dims).to(self._device)  # [K, D]
        for i in range(self._num_atoms):
            atom = torch.cos((2 * torch.arange(self._num_dims) + 1) * i * (
                        3.141592653589793 / (2 * self._num_dims)))  # * math.sqrt(2 / self._num_dims)
            weights[i, :] = atom / torch.norm(atom, p=2)
        return weights.t()  # [D, K]

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):
        input_z = input_z.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        return x_recon

    def forward(self, x):
        l = self._alpha / self._L  # scalar

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]

        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B, H, W, D] * [D, K] -> [B, H, W, K]

        z = self.soft_thresh(y, l)  # [B, H, W, K]
        for t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]

        z = z.permute(0, 3, 1, 2).contiguous()  # [B, H, W, K] -> [B, K, H, W]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]

        return z, x_recon, self._Dict


class AttentiveLISTA(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(AttentiveLISTA, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        self._conv = nn.Conv2d(in_channels=num_dims,
                               out_channels=num_atoms,
                               kernel_size=3, stride=1, padding=1)
        self._res1 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms // 2)
        self._res2 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms // 2)
        self._cbam = CBAM(in_channels=num_atoms,
                          reduction=16,
                          kernel_size=3)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        self._num_iters = num_iters

    def initialize_dct_weights(self):
        weights = torch.zeros(self._num_atoms, self._num_dims).to(self._device)  # [K, D]
        for i in range(self._num_atoms):
            atom = torch.cos((2 * torch.arange(self._num_dims) + 1) * i * (
                        3.141592653589793 / (2 * self._num_dims)))  # * math.sqrt(2 / self._num_dims)
            weights[i, :] = atom / torch.norm(atom, p=2)
        return weights.t()  # [D, K]

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):
        input_z = input_z.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        return x_recon

    def forward(self, x):
        l = self._conv(x)  # [B, D, H, W] -> [B, K, H, W]
        l = self._res1(l)
        l = self._res2(l)
        lam_before = l / self._L
        l = self._cbam(l) / self._L
        lam_after = l
        l = l.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]

        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B, H, W, D] * [D, K] -> [B, H, W, K]

        z = self.soft_thresh(y, l)  # [B, H, W, K]
        for t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]

        z = z.permute(0, 3, 1, 2).contiguous()  # [B, H, W, K] -> [B, K, H, W]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]

        return z, x_recon, self._Dict


class SCVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(SCVAE, self).__init__()

        self._encoder = Encoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder = Decoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)

        self._LISTA = LISTA(num_atoms=num_atoms,
                            num_dims=num_dims,
                            num_iters=num_iters,
                            h=256 // (2 ** down_samples),
                            w=256 // (2 ** down_samples),
                            device=device)

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)
        x_generation = self._decoder(ex)
        x_generation = torch.sigmoid(x_generation)
        return x_generation

    def forward(self, x):
        ex = self._encoder(x)  # [B, C, H, W] -> [B, D, h, w]
        z, ex_recon, dictionary = self._LISTA(ex)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
        x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
        x_recon = torch.sigmoid(x_recon)
        latent_loss = torch.sum((ex_recon - ex).pow(2), dim=1).mean()

        return x_recon, z, latent_loss, dictionary


class SSCVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(SSCVAE, self).__init__()

        self._encoder = Encoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder = Decoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)

        self._LISTA = AttentiveLISTA(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device)

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)  # [B, K, h, w] -> [B, D, h, w]
        x_generation = self._decoder(ex)  # [B, D, h, w] -> [B, C, H, W]
        x_generation = torch.sigmoid(x_generation)
        return x_generation

    def forward(self, x):
        ex = self._encoder(x)  # [B, C, H, W] -> [B, D, h, w]
        z, ex_recon, dictionary = self._LISTA(ex)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
        x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
        x_recon = torch.sigmoid(x_recon)
        latent_loss = torch.sum((ex_recon - ex).pow(2), dim=1).mean()

        return x_recon, z, latent_loss, dictionary


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    """

    def __init__(self,
                 num_atoms,
                 num_dims,
                 beta):
        super(VectorQuantizer, self).__init__()

        self.K = num_atoms
        self.D = num_dims
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW, D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW, K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW, K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B, H, W, D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, self.embedding.weight.permute(1,
                                                                                                          0).contiguous()  # [B, D, H, W]


class VQVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 beta):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder = Decoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)

        self._vq_layer = VectorQuantizer(num_atoms=num_atoms,
                                         num_dims=num_dims,
                                         beta=beta)

    def forward(self, x):
        ex = self._encoder(x)  # [B, C, H, W] -> [B, D, h, w]
        ex_recon, vq_loss, dictionary = self._vq_layer(ex)  # [B, D, h, w] -> [B, D, h, w]
        x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
        x_recon = torch.sigmoid(x_recon)

        return x_recon, vq_loss, dictionary


class SpecificEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 down_samples):
        super(SpecificEncoder, self).__init__()

        # [B, C, H, W] -> [B, C', H, W]
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels_1,
                                 kernel_size=3, stride=1, padding=1)

        # [B, C', H, W] -> [B, C'', h, w]
        # 只进行简单的下采样，不包含复杂的处理
        self._down_samples = nn.ModuleList()
        for i in range(down_samples):
            cur_in_channels = hid_channels_1 if i == 0 else hid_channels_2
            self._down_samples.append(
                DownSampleBlock(in_channels=cur_in_channels,
                                out_channels=hid_channels_2)
            )

    def forward(self, x):
        x = self._conv_1(x)

        for layer in self._down_samples:
            x = layer(x)

        return x


class SharedEncoder(nn.Module):
    def __init__(self,
                 hid_channels_2,
                 num_dims,
                 down_samples,
                 num_groups):
        super(SharedEncoder, self).__init__()
        # [B, C', H, W] -> [B, C'', h, w]
        self._down_samples = nn.ModuleList()
        for i in range(down_samples):
            cur_in_channels = hid_channels_2
            self._down_samples.append(
                ResidualBlock(in_channels=cur_in_channels,
                              hid_channels=cur_in_channels // 2)
            )
            self._down_samples.append(
                DownSampleBlock(in_channels=cur_in_channels,
                                out_channels=hid_channels_2)
            )

        # [B, C'', h, w] -> [B, C'', h, w]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)

        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_2)
        self._swish = Swish()

        # [B, C'', h, w] -> [B, num_dims, h, w]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_2,
                                 out_channels=num_dims,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        return x


class MultiSSCVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 num_iters,
                 cond,
                 device):
        super(MultiSSCVAE, self).__init__()

        # 保留原始编码器以保持兼容性
        # self._encoder = Encoder(in_channels=in_channels,
        #                         hid_channels_1=hid_channels_1,
        #                         hid_channels_2=hid_channels_2,
        #                         out_channels=num_dims,  # 编码器输出应该是num_dims
        #                         down_samples=down_samples,
        #                         num_groups=num_groups)
        # 为每个条件创建专属编码器（如果不存在）
        # 专属编码器字典 - 每个条件一个
        self._spec_encoders = nn.ModuleDict()
        for cond_name in cond:
            if cond_name not in self._spec_encoders:
                self._spec_encoders[cond_name] = SpecificEncoder(
                    in_channels=1,  # 获取输入通道数
                    hid_channels_1=hid_channels_1,  # 使用保存的参数
                    hid_channels_2=hid_channels_2,  # 使用保存的参数
                    down_samples=down_samples // 2  # 下采样层数减半
                )

        # 共享编码器 - 所有条件共用
        self._share_encoder = SharedEncoder(hid_channels_2=hid_channels_2,
                                            num_dims=num_dims,
                                            down_samples=down_samples // 2,
                                            num_groups=num_groups)

        # Shared decoder for target condition reconstruction
        self._decoder = Decoder(in_channels=in_channels,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=num_dims,  # 解码器输入应该是num_dims
                                up_samples=down_samples,
                                num_groups=num_groups)

        # Shared LISTA for sparse coding
        self._LISTA = AttentiveLISTA(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device)

        # 保存参数以便后续创建专属编码器
        # self.in_channels = in_channels
        # self.hid_channels_1 = hid_channels_1
        # self.hid_channels_2 = hid_channels_2
        # self.out_channels = out_channels
        # self.down_samples = down_samples
        # self.num_groups = num_groups

    def forward(self, x_dict):
        """
        Forward pass for multi-condition alignment.
        
        Args:
            x_dict: Dictionary with condition names as keys and tensors as values
                   Must contain 'target' key for target condition
        
        Returns:
            recon_dict: Dictionary with reconstructed images for each condition
            z_dict: Dictionary with sparse codes for each condition  
            latent_loss: Latent loss
            alignment_loss: Loss for aligning non-target conditions to target
            sparsity_loss: Sparsity loss for sparse codes
        """
        if 'target' not in x_dict:
            raise ValueError("Input dictionary must contain 'target' key.")

        recon_dict = {}
        z_dict = {}
        ex_dict = {}
        ex_recon_dict = {}
        dictionary = None

        # 处理非目标条件
        for cond_name, x in x_dict.items():
            if cond_name != 'target':
                # 使用专属编码器和共享编码器
                ex_spec = self._spec_encoders[cond_name](x)  # 专属编码器处理
                ex = self._share_encoder(ex_spec)  # 共享编码器处理
                ex_dict[cond_name] = ex

                # 稀疏编码
                z, ex_recon, _ = self._LISTA(ex)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
                ex_recon_dict[cond_name] = ex_recon
                z_dict[cond_name] = z

                # 解码到目标条件空间
                x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
                x_recon = torch.sigmoid(x_recon)
                recon_dict[cond_name] = x_recon

        # 计算潜在损失（重构损失）
        latent_loss = 0.0
        for cond_name in ex_dict.keys():
            latent_loss += torch.sum((ex_recon_dict[cond_name] - ex_dict[cond_name]).pow(2), dim=1).mean()
        latent_loss /= len(ex_dict)

        # 计算稀疏性损失
        sparsity_loss = 0.0
        for cond_name, z in z_dict.items():
            sparsity_loss += torch.mean(torch.abs(z))
        sparsity_loss /= len(z_dict)

        return recon_dict, z_dict, latent_loss, sparsity_loss

    def align_to_target(self, x_dict):
        """
        Align all conditions to target condition.
        
        Args:
            x_dict: Dictionary with condition names as keys and tensors as values
                   Must contain 'target' key for target condition
        
        Returns:
            aligned_dict: Dictionary with aligned images for each condition
        """
        if 'target' not in x_dict:
            raise ValueError("Input dictionary must contain 'target' key.")

        # 为每个条件创建专属编码器（如果不存在）
        for cond_name in x_dict.keys():
            if cond_name not in self._spec_encoders:
                self._spec_encoders[cond_name] = SpecificEncoder(
                    in_channels=x_dict[cond_name].size(1),  # 获取输入通道数
                    hid_channels_1=self.hid_channels_1,  # 使用保存的参数
                    hid_channels_2=self.hid_channels_2,  # 使用保存的参数
                    down_samples=self.down_samples // 2  # 下采样层数减半
                )

        aligned_dict = {}

        # 处理目标条件
        target_x = x_dict['target']
        target_ex_spec = self._spec_encoders['target'](target_x)  # 专属编码器处理
        target_ex = self._share_encoder(target_ex_spec)  # 共享编码器处理
        target_z, target_ex_recon, _ = self._LISTA(target_ex)  # 稀疏编码

        # 解码目标条件
        target_recon = self._decoder(target_ex_recon)
        target_recon = torch.sigmoid(target_recon)
        aligned_dict['target'] = target_recon

        # 处理非目标条件
        for cond_name, x in x_dict.items():
            if cond_name != 'target':
                # 使用专属编码器和共享编码器
                ex_spec = self._spec_encoders[cond_name](x)  # 专属编码器处理
                ex = self._share_encoder(ex_spec)  # 共享编码器处理

                # 稀疏编码
                z, ex_recon, _ = self._LISTA(ex)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]

                # 解码到目标条件空间
                x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
                x_recon = torch.sigmoid(x_recon)
                aligned_dict[cond_name] = x_recon

        return aligned_dict

    def generation(self, input_z_dict):
        """
        Generate images from sparse codes.
        
        Args:
            input_z_dict: Dictionary with condition names as keys and sparse codes as values
        
        Returns:
            generation_dict: Dictionary with generated images for each condition
        """
        generation_dict = {}

        for cond_name, z in input_z_dict.items():
            ex = self._LISTA.generation(z)  # [B, K, h, w] -> [B, D, h, w]
            x_generation = self._decoder(ex)  # [B, D, h, w] -> [B, C, H, W]
            x_generation = torch.sigmoid(x_generation)
            generation_dict[cond_name] = x_generation

        return generation_dict
