from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_fusion import FusionForwardOutput, FusionModelBase


def _fuse_info_diag(xhat: torch.Tensor, pdiag: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    y = 1.0 / pdiag.clamp_min(eps)
    if xhat.dim() == 2:
        wy = w[:, None] * y
        return (wy * xhat).sum(0) / wy.sum(0).clamp_min(eps)
    wy = w.unsqueeze(-1) * y
    return (wy * xhat).sum(1) / wy.sum(1).clamp_min(eps)


class _GraphFusionCore(FusionModelBase):
    def __init__(self, hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.valid_idx = valid_idx
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.node_logit = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _graph(self, h0: torch.Tensor):
        if h0.dim() == 2:
            n = h0.size(0)
            hi = h0.unsqueeze(1).expand(n, n, -1)
            hj = h0.unsqueeze(0).expand(n, n, -1)
            e = self.attn(torch.cat([hi, hj], -1)).squeeze(-1)
            e = e + torch.eye(n, device=h0.device, dtype=e.dtype) * (-1e9)
            a = torch.softmax(e, dim=1)
            h1 = self.upd(torch.cat([h0, a @ h0], -1))
            return h1, a
        b, n, _ = h0.shape
        hi = h0.unsqueeze(2).expand(b, n, n, -1)
        hj = h0.unsqueeze(1).expand(b, n, n, -1)
        e = self.attn(torch.cat([hi, hj], -1)).squeeze(-1)
        e = e + torch.eye(n, device=h0.device, dtype=e.dtype).unsqueeze(0) * (-1e9)
        a = torch.softmax(e, dim=2)
        h1 = self.upd(torch.cat([h0, torch.matmul(a, h0)], -1))
        return h1, a

    def _decode(self, post_feat, h1, mask, raw_logits, return_weights, aux):
        if mask is None:
            mask = torch.ones_like(raw_logits)
        valid = post_feat[..., self.valid_idx]
        logits = raw_logits + (valid - 1.0) * 2.0 + (mask - 1.0) * 1e9
        w = torch.softmax(logits, dim=0 if post_feat.dim() == 2 else 1)
        xhat = post_feat[..., 0:4].clone()
        xhat[..., 0] *= self.pos_scale
        xhat[..., 1] *= self.pos_scale
        xhat[..., 2] *= self.vel_scale
        xhat[..., 3] *= self.vel_scale
        pdiag = torch.expm1(post_feat[..., 4:8]).clamp_min(1e-6)
        pred = _fuse_info_diag(xhat, pdiag, w)
        if return_weights:
            aux = {"node_emb": h1, **aux}
            return FusionForwardOutput(pred=pred, weights=w, aux=aux)
        return FusionForwardOutput(pred=pred)


class OriginalGNNFusion(_GraphFusionCore):
    def __init__(self, in_dim=9, hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale)
        self.node_enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None):
        h1, a = self._graph(self.node_enc(post_feat))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})


class PostMeasDirectFusion(_GraphFusionCore):
    def __init__(self, post_in_dim=9, meas_in_dim=14, hidden_dim=64, meas_hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale)
        self.post_enc = nn.Sequential(nn.Linear(post_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.meas_enc = nn.Sequential(nn.Linear(meas_in_dim, meas_hidden_dim), nn.ReLU(), nn.Linear(meas_hidden_dim, meas_hidden_dim), nn.ReLU())
        self.fuse_proj = nn.Sequential(nn.Linear(hidden_dim + meas_hidden_dim, hidden_dim), nn.ReLU())

    def _build_h0(self, post_feat, meas_feat):
        if meas_feat is None:
            raise ValueError(f"{type(self).__name__} requires meas_feat.")
        return self.fuse_proj(torch.cat([self.post_enc(post_feat), self.meas_enc(meas_feat)], -1))

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None):
        h1, a = self._graph(self._build_h0(post_feat, meas_feat))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})


class PostMeasSoftGateFusion(PostMeasDirectFusion):
    def __init__(self, post_in_dim=9, meas_in_dim=14, hidden_dim=64, meas_hidden_dim=64, gate_hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0, gate_init_bias=0.0, gate_weight_alpha=1.0, gate_eps=1e-4, use_cov_calibration=False, cov_calib_min_scale=1.0, cov_calib_max_scale=20.0):
        super().__init__(post_in_dim, meas_in_dim, hidden_dim, meas_hidden_dim, valid_idx, pos_scale, vel_scale)
        self.gate_weight_alpha = gate_weight_alpha
        self.gate_eps = gate_eps
        self.use_cov_calibration = use_cov_calibration
        self.cov_calib_min_scale = cov_calib_min_scale
        self.cov_calib_max_scale = cov_calib_max_scale
        self.gate_net = nn.Sequential(nn.Linear(hidden_dim + meas_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init_bias))
        self.cov_calib = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if use_cov_calibration else None

    def _build_h0_gate(self, post_feat, meas_feat):
        h_post = self.post_enc(post_feat)
        h_meas = self.meas_enc(meas_feat)
        gate_soft = torch.sigmoid(self.gate_net(torch.cat([h_post, h_meas], -1)).squeeze(-1))
        gate = gate_soft * post_feat[..., self.valid_idx]
        h0 = self.fuse_proj(torch.cat([h_post, h_meas * gate.unsqueeze(-1)], -1))
        return h0, {"gate": gate, "gate_soft": gate_soft, "h_post": h_post, "h_meas": h_meas}

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None):
        if meas_feat is None:
            raise ValueError("PostMeasSoftGateFusion requires meas_feat.")
        h0, aux = self._build_h0_gate(post_feat, meas_feat)
        h1, a = self._graph(h0)
        raw = self.node_logit(h1).squeeze(-1)
        gate_bias = self.gate_weight_alpha * torch.log(aux["gate"].clamp_min(self.gate_eps))
        raw = raw + gate_bias
        cov_scale = torch.ones_like(raw)
        if self.cov_calib is not None:
            cov_scale = (self.cov_calib_min_scale + F.softplus(self.cov_calib(h1).squeeze(-1))).clamp(max=self.cov_calib_max_scale)
        aux.update({"attn_matrix": a, "raw_weight_logits": raw, "gate_weight_bias": gate_bias, "reliability_logits": raw, "cov_scale": cov_scale})
        return self._decode(post_feat, h1, mask, raw, return_weights, aux)


class PostMeasWindowDirectFusion(PostMeasDirectFusion):
    def __init__(self, post_in_dim=9, meas_in_dim=14, hidden_dim=64, meas_hidden_dim=64, window_size=6, valid_idx=8, pos_scale=1000.0, vel_scale=30.0):
        super().__init__(post_in_dim, meas_in_dim, hidden_dim, meas_hidden_dim, valid_idx, pos_scale, vel_scale)
        self.window_size = window_size
        self.post_gru = nn.GRU(post_in_dim, hidden_dim, batch_first=True)
        self.meas_gru = nn.GRU(meas_in_dim, meas_hidden_dim, batch_first=True)

    def _encode_window(self, post_win, meas_win):
        if post_win.dim() == 3:
            _, hp = self.post_gru(post_win)
            _, hm = self.meas_gru(meas_win)
            return self.fuse_proj(torch.cat([hp.squeeze(0), hm.squeeze(0)], -1))
        b, n, l, dp = post_win.shape
        dm = meas_win.shape[-1]
        _, hp = self.post_gru(post_win.reshape(b * n, l, dp))
        _, hm = self.meas_gru(meas_win.reshape(b * n, l, dm))
        return self.fuse_proj(torch.cat([hp.squeeze(0), hm.squeeze(0)], -1)).reshape(b, n, self.hidden_dim)

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None):
        if post_win is None or meas_win is None:
            raise ValueError("PostMeasWindowDirectFusion requires post_win and meas_win.")
        h1, a = self._graph(self._encode_window(post_win, meas_win))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})
