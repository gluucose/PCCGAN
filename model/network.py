# -*- coding = utf-8 -*-

# @time:2023/5/8 21:47

# Author:Cui

import random

import torch
import torch.nn as nn
from timm.models.layers.helpers import to_3tuple
from torch.autograd import Variable

from context_cluster3D import ContextCluster, basic_blocks


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=5, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Generator(nn.Module):
    def __init__(self,
                 layers=[1, 1, 1, 1],
                 norm_layer=GroupNorm,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 downsamples=[True, True, True, True],
                 proposal_w=[2, 2, 2, 2],
                 proposal_h=[2, 2, 2, 2],
                 proposal_d=[2, 2, 2, 2],
                 fold_w=[1, 1, 1, 1],
                 fold_h=[1, 1, 1, 1],
                 fold_d=[1, 1, 1, 1],
                 heads=[4, 4, 8, 8],
                 head_dim=[24, 24, 24, 24],
                 down_patch_size=3,
                 down_pad=1
                 ):
        super(Generator, self).__init__()

        self.CoCs = ContextCluster(
            layers=layers, embed_dims=embed_dims, norm_layer=norm_layer,
            mlp_ratios=mlp_ratios, downsamples=downsamples,
            down_patch_size=down_patch_size, down_pad=down_pad,
            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim,
        )

    def forward(self, LPET):
        EPET = self.CoCs(LPET)

        return EPET


class Discriminator(nn.Module):
    def __init__(self,
                 layers=[1, 1, 1, 1, 1],
                 norm_layer=GroupNorm,
                 embed_dims=[8, 16, 32, 64, 128],
                 mlp_ratios=[8, 8, 4, 4, 4],
                 proposal_w=[2, 2, 2, 2, 2], proposal_h=[2, 2, 2, 2, 2], proposal_d=[2, 2, 2, 2, 2],
                 fold_w=[1, 1, 1, 1, 1], fold_h=[1, 1, 1, 1, 1], fold_d=[1, 1, 1, 1, 1],
                 heads=[4, 4, 8, 8, 8], head_dim=[24, 24, 24, 24, 24],
                 # fixed settings
                 down_patch_size=3, down_stride=2, down_pad=1, in_patch_size=3, in_stride=2, in_pad=1, drop_rate=0.,
                 act_layer=nn.GELU, use_layer_scale=True, layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        """ Encoder """
        self.patch_embed = PointRecuder(patch_size=in_patch_size, stride=in_stride, padding=in_pad,
                                        in_chans=5, embed_dim=embed_dims[0])
        # en0
        self.en0 = basic_blocks(embed_dims[0], 0, layers, mlp_ratio=mlp_ratios[0], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[0], proposal_h=proposal_h[0], proposal_d=proposal_d[0],
                                fold_w=fold_w[0], fold_h=fold_h[0], fold_d=fold_d[0],
                                heads=heads[0], head_dim=head_dim[0], return_center=False)
        # en1
        self.down1 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.en1 = basic_blocks(embed_dims[1], 1, layers, mlp_ratio=mlp_ratios[1], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[1], proposal_h=proposal_h[1], proposal_d=proposal_d[1],
                                fold_w=fold_w[1], fold_h=fold_h[1], fold_d=fold_d[1],
                                heads=heads[1], head_dim=head_dim[1], return_center=False)
        # en2
        self.down2 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.en2 = basic_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[2], proposal_h=proposal_h[2], proposal_d=proposal_d[2],
                                fold_w=fold_w[2], fold_h=fold_h[2], fold_d=fold_d[2],
                                heads=heads[2], head_dim=head_dim[2], return_center=False)
        # en3
        self.down3 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.en3 = basic_blocks(embed_dims[3], 3, layers, mlp_ratio=mlp_ratios[3], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[3], proposal_h=proposal_h[3], proposal_d=proposal_d[3],
                                fold_w=fold_w[3], fold_h=fold_h[3], fold_d=fold_d[3],
                                heads=heads[3], head_dim=head_dim[3], return_center=False)
        # en3
        self.down4 = PointRecuder(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                  in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.en4 = basic_blocks(embed_dims[4], 4, layers, mlp_ratio=mlp_ratios[4], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[4], proposal_h=proposal_h[4], proposal_d=proposal_d[4],
                                fold_w=fold_w[4], fold_h=fold_h[4], fold_d=fold_d[4],
                                heads=heads[4], head_dim=head_dim[4], return_center=False)

        self.sigmoid = nn.Sigmoid()

    def forward_embeddings(self, x):
        _, c, img_w, img_h, img_d = x.shape
        # print(f"img size is {c} * {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        range_d = torch.arange(0, img_d, step=1) / (img_d - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, range_d), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(3, 0, 1, 2).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward(self, x):
        en0 = self.forward_embeddings(x)
        en0 = self.en0(en0)

        en1 = self.down1(en0)
        en1 = self.en1(en1)

        en2 = self.down2(en1)
        en2 = self.en2(en2)

        en3 = self.down3(en2)
        en3 = self.en3(en3)

        en4 = self.down4(en3)
        en4 = self.en4(en4)

        output = self.sigmoid(en4)

        return output


class GANLoss_smooth(nn.Module):
    def __init__(self, device, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss_smooth, self).__init__()
        self.device = device
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real, smooth):
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label + smooth * 0.5 - 0.3)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label + smooth * 0.3)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        a = random.uniform(0, 1)
        target_tensor = self.get_target_tensor(input, target_is_real, a)
        return self.loss(input, target_tensor.to(self.device))


def D_train(D: Discriminator, G: Generator, LPET, SPET, optimizer_D, device):
    LPET = LPET.to(device)
    SPET = SPET.to(device)

    PET = torch.cat([LPET, SPET], dim=1)

    D.zero_grad()

    # real data
    D_output_r = D(PET).squeeze()
    criGAN = GANLoss_smooth(device=device)
    D_real_loss = criGAN(D_output_r, True)

    # fake data
    G_output = G(LPET)
    X_fake = torch.cat([LPET, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    D_fake_loss = criGAN(D_output_f, False)

    # back prop
    D_loss = (D_real_loss + D_fake_loss) * 0.5

    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


def G_train(D: Discriminator, G: Generator, LPET, SPET, L1, optimizer_G, device, lamb=100):
    LPET = LPET.to(device)
    SPET = SPET.to(device)

    G.zero_grad()

    # fake data
    G_output = G(LPET)
    X_fake = torch.cat([LPET, G_output], dim=1)
    D_output_f = D(X_fake).squeeze()
    criGAN = GANLoss_smooth(device=device)
    G_BCE_loss = criGAN(D_output_f, True)

    G_L1_Loss = L1(G_output, SPET)

    G_loss = G_BCE_loss + lamb * G_L1_Loss

    G_loss.backward()
    optimizer_G.step()

    return G_loss.data.item()


def GD_Train(D: Discriminator, G: Generator, LPET, SPET,
             optimizer_G, optimizer_D, L1, device, imgpool, lamb=100):
    x = LPET.to(device)
    y = SPET.to(device)

    xy = torch.cat([x, y], dim=1)

    criGAN = GANLoss_smooth(device=device)
    G_output = G(x)

    X_fake = imgpool.query(torch.cat([x, G_output], dim=1))

    # train_D
    optimizer_D.zero_grad()
    D_output_r = D(xy.detach()).squeeze()
    D_output_f = D(X_fake.detach()).squeeze()
    D_real_loss = criGAN(D_output_r, True)  # real loss
    D_fake_loss = criGAN(D_output_f, False)  # fake loss
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    # train_G
    optimizer_G.zero_grad()
    D_output = D(X_fake).squeeze()
    G_BCE_loss = criGAN(D_output, True)
    G_L1_Loss = L1(G_output, y)

    G_loss = G_BCE_loss + lamb * G_L1_Loss
    G_loss.backward()
    optimizer_G.step()

    return D_loss.data.item(), G_loss.data.item()


if __name__ == '__main__':
    pass
