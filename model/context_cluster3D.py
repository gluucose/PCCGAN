# -*- coding = utf-8 -*-

# @time:2023/5/8 12:39

# Author:Cui

"""
3D ContextCluster implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers.helpers import to_3tuple
from timm.models.registry import register_model


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride, D/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
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


class PointExpander(nn.Module):
    """
    Point Expander is implemented by a layer of decov since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W, D]
    Output: tensor in shape [B, embed_dim, H*stride, W*stride, D*stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        # print(in_chans, embed_dim)
        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size,
                                       stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim,
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param proposal_d: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param fold_d: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv3d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((proposal_w, proposal_h, proposal_d))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.fold_d = fold_d
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h, d]
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h d -> (b e) c w h d", e=self.heads)
        value = rearrange(value, "b (e c) w h d -> (b e) c w h d", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0, d0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0 and d0 % self.fold_d == 0, \
                f"Ensure the feature map size ({w0}*{h0}*{w0}) can be divided by fold " \
                f"{self.fold_w}*{self.fold_h}*{self.fold_d}"
            x = rearrange(x, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2 f3) c w h d", f1=self.fold_w,
                          f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1],ks[2]]
            value = rearrange(value, "b c (f1 w) (f2 h) (f3 d) -> (b f1 f2) c w h d", f1=self.fold_w,
                              f2=self.fold_h, f3=self.fold_d)
        b, c, w, h, d = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H,C_D], we set M = C_W*C_H and N = w*h*d
        value_centers = rearrange(self.centers_proposal(value), 'b c w h d -> b (w h d) c')  # [b,C_W,C_H,c]
        b, c, ww, hh, dd = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h d -> b (w h d) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        # a small bug: mask.sum should be sim.sum according to Eq. (1),
        # mask can be considered as a hard version of sim in our implementation.
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h d) c -> b c w h d", w=ww, h=hh)  # center shape
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h d) c -> b c w h d", w=w, h=h)  # cluster shape

        if self.fold_w > 1 and self.fold_h > 1 and self.fold_d > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2 f3) c w h d -> b c (f1 w) (f2 h) (f3 d)", f1=self.fold_w,
                            f2=self.fold_h, f3=self.fold_d)
        out = rearrange(out, "(b e) c w h d -> b (e c) w h d", e=self.heads)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 4, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 4, 1, 2, 3)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm, drop=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim,
                                   proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
                                   fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
                                   heads=heads, head_dim=head_dim, return_center=return_center)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # The following technique is useful to train deep ContextClusters.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))
            x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        else:
            x = x + self.token_mixer(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, proposal_d=2,
                 fold_w=2, fold_h=2, fold_d=2,
                 heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
            fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim, return_center=return_center
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ContextCluster(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --norm_layer, --act_layer: define the types of normalization and activation
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None,
                 norm_layer=nn.BatchNorm3d, act_layer=nn.GELU,
                 in_patch_size=3, in_stride=2, in_pad=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 up_patch_size=2, up_stride=2, up_pad=0,
                 drop_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], proposal_d=[2, 2, 2, 2],
                 fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1], fold_d=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):
        super().__init__()

        """ Encoder """
        self.patch_embed = PointRecuder(patch_size=in_patch_size, stride=in_stride, padding=in_pad,
                                        in_chans=4, embed_dim=embed_dims[0])
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

        """Decoder"""
        # de0
        self.de0 = basic_blocks(embed_dims[3], 3, layers, mlp_ratio=mlp_ratios[3], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[3], proposal_h=proposal_h[3], proposal_d=proposal_h[3],
                                fold_w=fold_w[3], fold_h=fold_h[3], fold_d=fold_d[3],
                                heads=heads[3], head_dim=head_dim[3], return_center=False)
        self.up0 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                                 in_chans=embed_dims[3], embed_dim=embed_dims[2])
        # de1
        self.de1 = basic_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[2], proposal_h=proposal_h[2], proposal_d=proposal_d[2],
                                fold_w=fold_w[2], fold_h=fold_h[2], fold_d=fold_d[2],
                                heads=heads[2], head_dim=head_dim[2], return_center=False)
        self.up1 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                                 in_chans=embed_dims[2], embed_dim=embed_dims[1])
        # de2
        self.de2 = basic_blocks(embed_dims[1], 1, layers, mlp_ratio=mlp_ratios[1], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[1], proposal_h=proposal_h[1], proposal_d=proposal_d[1],
                                fold_w=fold_w[1], fold_h=fold_h[1], fold_d=fold_d[1],
                                heads=heads[1], head_dim=head_dim[1], return_center=False)
        self.up2 = PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                                 in_chans=embed_dims[1], embed_dim=embed_dims[0])
        # de3
        self.de3 = basic_blocks(embed_dims[0], 0, layers, mlp_ratio=mlp_ratios[0], act_layer=act_layer,
                                norm_layer=norm_layer, drop_rate=drop_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                proposal_w=proposal_w[0], proposal_h=proposal_h[0], proposal_d=proposal_d[0],
                                fold_w=fold_w[0], fold_h=fold_h[0], fold_d=fold_d[0],
                                heads=heads[0], head_dim=head_dim[0], return_center=False)
        self.patch_expand = nn.Sequential(
            PointExpander(patch_size=up_patch_size, stride=up_stride, padding=up_pad,
                          in_chans=embed_dims[0], embed_dim=3),
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1),
        )

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

    def restore_embeddings(self, x):
        x = self.patch_expand(x)

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

        de0 = self.de0(en3)
        de0 = self.up0(de0) + en2

        de1 = self.de1(de0)
        de1 = self.up1(de1) + en1

        de2 = self.de2(de1)
        de2 = self.up2(de2) + en0

        de3 = self.de3(de2)

        de3 = self.patch_expand(de3)

        output = de3 + x

        return output


@register_model
def coc_tiny_plain(**kwargs):
    layers = [1, 1, 1, 1]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    proposal_d = [2, 2, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    fold_d = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, proposal_d=proposal_d,
        fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
        heads=heads, head_dim=head_dim,
        **kwargs)

    return model


if __name__ == '__main__':
    pass
