import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Nrsfm_Net(nn.Module):
    def __init__(self, cfg):
        super(Nrsfm_Net, self).__init__()
        self.cfg = cfg

        if cfg.exp_atom_decrease:
            self.channels = [cfg.num_points] + [int(cfg.num_atoms / (2 ** f)) for f in range(cfg.num_dictionaries)]

            assert self.channels[-1] == cfg.num_atoms_bottleneck

        else:
            self.channels = [cfg.num_points] + np.linspace(
                cfg.num_atoms,
                cfg.num_atoms_bottleneck,
                cfg.num_dictionaries - 1).astype(np.int64).tolist()

        # TODO: bias?!
        # self.projection_layer = nn.ConvTranspose2d(in_channels=cfg.num_points, out_channels=cfg.num_atoms, kernel_size=(cfg.BLOCK_HEIGHT, 1), stride=1, padding=0)

        self.weights = []
        for l, (inp_ch, out_ch) in enumerate(zip(self.channels[:-1], self.channels[1:])):
            if l == 0:
                weight = nn.Parameter(torch.randn(inp_ch, out_ch, cfg.BLOCK_HEIGHT, 1))
            else:
                weight = nn.Parameter(torch.randn(inp_ch, out_ch, 1, 1))

            init.kaiming_uniform_(weight.data)
            self.weights.append(weight)

        self.decoder_bias = nn.Parameter(torch.randn(cfg.BLOCK_HEIGHT * self.channels[0]))

        self.encoder_layers = []
        self.decoder_layers = []
        for w in self.weights:
            enc = nn.ConvTranspose2d(in_channels=w.shape[0], out_channels=w.shape[1],
                                     kernel_size=(w.shape[2], w.shape[3]), stride=1, padding=0)
            enc.weight = w
            self.encoder_layers.append(enc)
            dec = nn.Conv2d(in_channels=w.shape[1], out_channels=w.shape[0], kernel_size=1, stride=1, padding=0)
            dec.weight = w
            self.decoder_layers.append(dec)
        self.decoder_layers.reverse()

        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

        self.coef_layer = nn.Conv2d(in_channels=cfg.num_atoms_bottleneck, out_channels=cfg.num_atoms_bottleneck,
                                    kernel_size=(cfg.BLOCK_HEIGHT, cfg.BLOCK_WIDTH), stride=1, padding='valid',
                                    bias=True)
        self.camera_layer = nn.Conv2d(in_channels=cfg.num_atoms_bottleneck, out_channels=1, kernel_size=1, stride=1,
                                      padding='valid', bias=True)

    def forward_encoder(self, x):
        x = torch.unsqueeze(x, 2)
        for encoder_layer in self.encoder_layers:
            x = torch.relu(encoder_layer(x))

        return x

    def forward_decoder(self, d):
        for decoder_layer in self.decoder_layers[:-1]:
            d = torch.relu(decoder_layer(d))

        wht = self.decoder_layers[-1].weight
        out_ch, in_ch, h, w = wht.shape
        wht = wht.permute(2, 3, 1, 0)  # w to tf
        wht = wht.permute(1, 2, 3, 0)
        wht = wht.contiguous().view(w, 1, in_ch * out_ch, h)
        wht = wht.contiguous().view(w, 1, in_ch, h * out_ch)
        wht = wht.permute(3, 2, 0, 1)  # w to torch

        d = F.conv2d(d, wht, bias=self.decoder_bias, stride=1, padding=0)
        d = d.view(-1, self.cfg.num_points, 3)

        return d

    def forward_camera(self, x):
        cameras = self.camera_layer(x)
        cameras = cameras.view(-1, self.cfg.BLOCK_HEIGHT, self.cfg.BLOCK_WIDTH)

        U, _, Vh = torch.linalg.svd(cameras, full_matrices=False)

        cameras = torch.matmul(U, Vh.transpose(-2, -1))
        return cameras

    def forward(self, x_list):

        x_list = [self.forward_encoder(x) for x in x_list]
        cameras = [self.forward_camera(x) for x in x_list]

        d = torch.stack(x_list, dim=0).sum(dim=0)
        d = self.coef_layer(d)
        d = self.forward_decoder(d)

        return d, cameras


class Multi_Nrsfm_Net(Nrsfm_Net):
    def __init__(self, cfg):
        super(Multi_Nrsfm_Net, self).__init__(cfg)

    def forward(self, x_list):
        x_list = [self.forward_encoder(x) for x in x_list]

        cameras = [self.forward_camera(x) for x in x_list]

        d = self.coef_layer(torch.add(x_list))
        d = self.forward_decoder(d)

        return d, cameras


class Multi_RF_Nrsfm_Net(Nrsfm_Net):
    def __init__(self, cfg):
        super(Multi_RF_Nrsfm_Net, self).__init__(cfg)

        in_features = cfg.num_atoms_bottleneck * cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH
        out_features = cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH + cfg.num_atoms_bottleneck

        self.rf_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward_camera(self, cameras, return_UV=False):
        xx = cameras.contiguous().view(-1, self.cfg.BLOCK_HEIGHT, self.cfg.BLOCK_WIDTH)

        U, _, Vh = torch.linalg.svd(xx, full_matrices=False)
        cameras = torch.matmul(U, Vh.transpose(-2, -1))

        if return_UV:
            return cameras, xx, Vh.transpose(-2, -1)
        else:
            return cameras

    def forward_rf(self, x):
        b = x.shape[0]
        x = self.rf_layer(x.view(b, -1))

        camera_atoms = self.cfg.BLOCK_HEIGHT * self.cfg.BLOCK_WIDTH

        return x[:, :camera_atoms].view(b, self.cfg.BLOCK_HEIGHT, self.cfg.BLOCK_WIDTH), x[:, camera_atoms:].view(b,
                                                                                                                  self.cfg.num_atoms_bottleneck,
                                                                                                                  1, 1)

    def forward(self, x_list, return_repr=False):
        x_list = [self.forward_encoder(x) for x in x_list]

        cd = [self.forward_rf(x) for x in x_list]

        cameras = [self.forward_camera(c, return_UV=False) for c, _ in cd]

        d = torch.stack([d for _, d in cd], dim=0).sum(dim=0)

        d = self.forward_decoder(d)

        if return_repr:
            return d, cameras, x_list
        else:
            return d, cameras


class Siamese_RF_Nrsfm_Net(Multi_RF_Nrsfm_Net):
    def __init__(self, cfg):
        super(Siamese_RF_Nrsfm_Net, self).__init__(cfg)

        in_features = 2 * cfg.num_atoms_bottleneck * cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH
        out_features = cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH + 2 * cfg.num_atoms_bottleneck

        self.rf_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward_rf(self, x_a, x_b):
        bs = x_a.shape[0]
        x = self.rf_layer(torch.cat([x_a.view(bs, -1), x_b.view(bs, -1)], 1))

        camera_atoms = self.cfg.BLOCK_HEIGHT * self.cfg.BLOCK_WIDTH

        d_a = x[:, :self.cfg.num_atoms_bottleneck].view(bs, self.cfg.num_atoms_bottleneck, 1, 1)
        cam = x[:, self.cfg.num_atoms_bottleneck:self.cfg.num_atoms_bottleneck + camera_atoms].view(bs,
                                                                                                    self.cfg.BLOCK_HEIGHT,
                                                                                                    self.cfg.BLOCK_WIDTH)
        d_b = x[:, self.cfg.num_atoms_bottleneck + camera_atoms:].view(bs, self.cfg.num_atoms_bottleneck, 1, 1)

        return cam, d_a, d_b

    def forward(self, x_a_list, x_b_list):
        x_a_list = [self.forward_encoder(x) for x in x_a_list]
        x_b_list = [self.forward_encoder(x) for x in x_b_list]

        cd = [self.forward_rf(x_a, x_b) for x_a, x_b in zip(x_a_list, x_b_list)]

        cameras = [self.forward_camera(c) for c, _, _ in cd]

        d_a = torch.stack([d for _, d, _ in cd], dim=0).sum(dim=0)
        d_b = torch.stack([d for _, _, d in cd], dim=0).sum(dim=0)

        d_a = self.forward_decoder(d_a)
        d_b = self.forward_decoder(d_b)

        return d_a, d_b, cameras


class Rel_Multi_RF_Nrsfm_Net(Multi_RF_Nrsfm_Net):
    def __init__(self, cfg):
        super(Rel_Multi_RF_Nrsfm_Net, self).__init__(cfg)

        in_features = 2 * cfg.num_atoms_bottleneck * cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH
        out_features = 2 * cfg.BLOCK_HEIGHT * cfg.BLOCK_WIDTH

        self.relative_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward_rel(self, x_a, x_b):
        bs = x_a.shape[0]
        feat = torch.cat([x_a, x_b], 1).view(bs, -1)
        cam_feats = self.relative_layer(feat)
        camera_atoms = self.cfg.BLOCK_HEIGHT * self.cfg.BLOCK_WIDTH
        rel_cam_a = self.forward_camera(cam_feats[:, :camera_atoms])
        rel_cam_b = self.forward_camera(cam_feats[:, camera_atoms:])

        return rel_cam_a, rel_cam_b

    def forward(self, x_a_list, x_b_list, x_r_list):
        d_a, cam_a, rep_a = super(Rel_Multi_RF_Nrsfm_Net, self).forward(x_a_list, return_repr=True)
        d_b, cam_b, rep_b = super(Rel_Multi_RF_Nrsfm_Net, self).forward(x_b_list, return_repr=True)

        rel_cams = [self.forward_rel(ra, rb) for ra, rb in zip(rep_a, rep_b)]

        rel_cam_a = [a for a, _ in rel_cams]
        rel_cam_b = [b for _, b in rel_cams]

        return d_a, cam_a, d_b, cam_b, rel_cam_a, rel_cam_b
