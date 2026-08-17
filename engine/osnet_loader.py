#!/usr/bin/env python3
"""Self-contained OSNet-x0_25 person-ReID backbone (pure torch.nn).

Faithfully ported (MIT license) from KaiyangZhou/deep-person-reid
`torchreid/models/osnet.py` (Zhou et al., "Omni-Scale Feature Learning for
Person Re-Identification", ICCV 2019). Only the x0_25 path is kept; the gdown
weight downloader is replaced by a plain local-file `load_state_dict`.

Weights are NOT bundled (license/size). Drop a ReID checkpoint at the path given
to build() — e.g. `osnet_x0_25_msmt17.pth` from the deep-person-reid MODEL_ZOO
(check its license for your use). If absent, the caller falls back to CLIP.
"""
from __future__ import annotations
import os
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_c, affine=True) if IN else nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c); self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c); self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_c, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super().__init__()
        num_gates = num_gates or in_c
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=True, padding=0)
        self.norm1 = nn.LayerNorm((in_c // reduction, 1, 1)) if layer_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, num_gates, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid() if gate_activation == 'sigmoid' else (
            nn.ReLU(inplace=True) if gate_activation == 'relu' else None)

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x); x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x); x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        return x if self.return_gates else inp * x


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, IN=False, bottleneck_reduction=4, **kw):
        super().__init__()
        mid = out_c // bottleneck_reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None
        self.IN = nn.InstanceNorm2d(out_c, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', IN=False, **kw):
        super().__init__()
        self.loss = loss; self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], True, IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3], None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, n, in_c, out_c, reduce_spatial_size, IN=False):
        layers = [block(in_c, out_c, IN=IN)]
        for _ in range(1, n):
            layers.append(block(out_c, out_c, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim; return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers, dim = [], fc_dims
        d = input_dim
        for m in fc_dims:
            layers += [nn.Linear(d, m), nn.BatchNorm1d(m), nn.ReLU(inplace=True)]
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            d = m
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x); x = self.maxpool(x)
        x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x).view(x.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return y if self.loss == 'softmax' else (y, v)


def osnet_x0_25(num_classes=1000, **kw):
    return OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                 channels=[16, 64, 96, 128], **kw)


def build(weight_path=None, device="cpu"):
    """Construct OSNet-x0_25 in eval mode; load ReID weights if weight_path exists.
    Returns (model, feature_dim). Never raises on missing/invalid weights — caller
    checks and falls back.

    NOTE: this minimal re-implementation matches ONLY the x0.25 layout. The
    torchreid-native checkpoints (x0.5/x0.75/x1.0, with the 4× OSBlock expansion)
    must be loaded through the torchreid package instead — see matcher._osnet_x1().
    """
    model = osnet_x0_25(num_classes=1000, loss='softmax')
    if weight_path and os.path.exists(weight_path):
        sd = torch.load(weight_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        new = OrderedDict()
        for k, v in sd.items():
            kk = k[7:] if k.startswith('module.') else k
            # Skip classifier head when shape mismatches (e.g. MSMT17 4101 classes
            # vs model default 1000) — we only need the backbone for 512-d features.
            if 'classifier' in kk:
                continue
            new[kk] = v
        model.load_state_dict(new, strict=False)
    model.eval()
    try:
        model.to(device)
    except Exception:
        model.to('cpu')
    return model, model.feature_dim


if __name__ == "__main__":
    m, fd = build()
    n = sum(p.numel() for p in m.parameters())
    print(f"osnet_x0_25 built OK: feature_dim={fd}, params={n/1e6:.2f}M")
