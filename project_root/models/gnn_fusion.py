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


def _fuse_aa(xhat: torch.Tensor, w: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    ww = w * active
    if xhat.dim() == 2:
        denom = ww.sum(0).clamp_min(eps)
        return (ww[:, None] * xhat).sum(0) / denom
    denom = ww.sum(1, keepdim=True).clamp_min(eps)
    return (ww.unsqueeze(-1) * xhat).sum(1) / denom


def _fuse_aa_mm_diag(xhat: torch.Tensor, pdiag: torch.Tensor, w: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-6
    ww = w * active
    if xhat.dim() == 2:
        ww = ww / ww.sum(0).clamp_min(eps)
        pred = (ww[:, None] * xhat).sum(0)
        delta = xhat - pred[None, :]
        p_fused = (ww[:, None] * (pdiag + delta * delta)).sum(0)
        return pred, p_fused
    ww = ww / ww.sum(1, keepdim=True).clamp_min(eps)
    pred = (ww.unsqueeze(-1) * xhat).sum(1)
    delta = xhat - pred.unsqueeze(1)
    p_fused = (ww.unsqueeze(-1) * (pdiag + delta * delta)).sum(1)
    return pred, p_fused


def _apply_weight_cap(w: torch.Tensor, cap: torch.Tensor, active: torch.Tensor, max_iter: int = 8) -> torch.Tensor:
    eps = 1e-6
    if w.dim() == 1:
        w2 = w.unsqueeze(0)
        cap2 = cap.unsqueeze(0)
        active2 = active.unsqueeze(0)
        return _apply_weight_cap(w2, cap2, active2, max_iter=max_iter).squeeze(0)

    active = (active > 0.0).to(w.dtype)
    cap = cap.clamp_min(eps).clamp(max=1.0) * active
    cap_sum = cap.sum(dim=1, keepdim=True).clamp_min(eps)
    cap = torch.where(cap_sum < 1.0, cap / cap_sum, cap)

    base = (w * active).clamp_min(0.0)
    base = base / base.sum(dim=1, keepdim=True).clamp_min(eps)
    fixed = torch.zeros_like(base)
    free = active.bool()
    remaining_mass = torch.ones((w.shape[0], 1), device=w.device, dtype=w.dtype)

    for _ in range(max_iter):
        free_f = free.to(w.dtype)
        denom = (base * free_f).sum(dim=1, keepdim=True).clamp_min(eps)
        proposal = base * free_f / denom * remaining_mass
        over = (proposal > cap) & free
        if not bool(over.any()):
            fixed = fixed + proposal
            break
        add = torch.where(over, cap, torch.zeros_like(cap))
        fixed = fixed + add
        remaining_mass = (1.0 - fixed.sum(dim=1, keepdim=True)).clamp_min(0.0)
        free = free & (~over)
        if not bool(free.any()):
            break

    capped = fixed * active
    return capped / capped.sum(dim=1, keepdim=True).clamp_min(eps)


class _GraphFusionCore(FusionModelBase):
    def __init__(self, hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0, output_fusion_mode="info_diag"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.valid_idx = valid_idx
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.output_fusion_mode = str(output_fusion_mode)
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

    def _decode(self, post_feat, h1, mask, raw_logits, return_weights, aux, cov_scale=None, weight_uniform_mix=0.0, weight_cap=None):
        if mask is None:
            mask = torch.ones_like(raw_logits)
        valid = post_feat[..., self.valid_idx]
        active = (valid * mask).clamp_min(0.0)
        logits = raw_logits + (valid - 1.0) * 2.0 + (mask - 1.0) * 1e9
        w = torch.softmax(logits, dim=0 if post_feat.dim() == 2 else 1)
        mix = float(weight_uniform_mix)
        if mix > 0.0:
            mix = min(mix, 1.0)
            denom = active.sum(dim=0 if post_feat.dim() == 2 else 1, keepdim=True).clamp_min(1e-6)
            uniform_w = active / denom
            w = (1.0 - mix) * w + mix * uniform_w
        if weight_cap is not None:
            w = _apply_weight_cap(w, weight_cap, active)
        xhat = post_feat[..., 0:4].clone()
        xhat[..., 0] *= self.pos_scale
        xhat[..., 1] *= self.pos_scale
        xhat[..., 2] *= self.vel_scale
        xhat[..., 3] *= self.vel_scale
        pdiag = torch.expm1(post_feat[..., 4:8]).clamp_min(1e-6)
        if cov_scale is not None:
            pdiag = pdiag * cov_scale.clamp_min(1e-6).unsqueeze(-1)
        if self.output_fusion_mode == "info_diag":
            pred = _fuse_info_diag(xhat, pdiag, w)
        elif self.output_fusion_mode == "aa":
            pred = _fuse_aa(xhat, w, active)
        elif self.output_fusion_mode == "aa_mm":
            pred, fused_cov_diag = _fuse_aa_mm_diag(xhat, pdiag, w, active)
            aux = {"fused_cov_diag": fused_cov_diag, **aux}
        else:
            raise ValueError(f"Unknown output_fusion_mode: {self.output_fusion_mode}")
        if return_weights:
            aux = {"node_emb": h1, **aux}
            return FusionForwardOutput(pred=pred, weights=w, aux=aux)
        return FusionForwardOutput(pred=pred)


class OriginalGNNFusion(_GraphFusionCore):
    def __init__(self, in_dim=9, hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0, output_fusion_mode="info_diag"):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.node_enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        h1, a = self._graph(self.node_enc(post_feat))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})


class PostMeasDirectFusion(_GraphFusionCore):
    def __init__(self, post_in_dim=9, meas_in_dim=18, hidden_dim=64, meas_hidden_dim=64, valid_idx=8, pos_scale=1000.0, vel_scale=30.0, output_fusion_mode="info_diag"):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.post_enc = nn.Sequential(nn.Linear(post_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.meas_enc = nn.Sequential(nn.Linear(meas_in_dim, meas_hidden_dim), nn.ReLU(), nn.Linear(meas_hidden_dim, meas_hidden_dim), nn.ReLU())
        self.fuse_proj = nn.Sequential(nn.Linear(hidden_dim + meas_hidden_dim, hidden_dim), nn.ReLU())

    def _build_h0(self, post_feat, meas_feat):
        if meas_feat is None:
            raise ValueError(f"{type(self).__name__} requires meas_feat.")
        return self.fuse_proj(torch.cat([self.post_enc(post_feat), self.meas_enc(meas_feat)], -1))

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        h1, a = self._graph(self._build_h0(post_feat, meas_feat))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})


class PostMeasSoftGateFusion(PostMeasDirectFusion):
    def __init__(
        self,
        post_in_dim=9,
        meas_in_dim=18,
        hidden_dim=64,
        meas_hidden_dim=64,
        gate_hidden_dim=64,
        valid_idx=8,
        pos_scale=1000.0,
        vel_scale=30.0,
        gate_init_bias=0.0,
        gate_weight_alpha=1.0,
        gate_eps=1e-4,
        base_logit_temperature=1.0,
        weight_uniform_mix=0.0,
        use_meas_in_representation=True,
        use_gate_on_meas_feature=True,
        use_cov_calibration=False,
        cov_calib_min_scale=1.0,
        cov_calib_max_scale=20.0,
        use_cov_in_fusion=False,
        cov_weight_beta=0.0,
        output_fusion_mode="info_diag",
    ):
        super().__init__(post_in_dim, meas_in_dim, hidden_dim, meas_hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.gate_weight_alpha = gate_weight_alpha
        self.gate_eps = gate_eps
        self.base_logit_temperature = max(float(base_logit_temperature), 1e-6)
        self.weight_uniform_mix = max(float(weight_uniform_mix), 0.0)
        self.use_meas_in_representation = use_meas_in_representation
        self.use_gate_on_meas_feature = use_gate_on_meas_feature
        self.use_cov_calibration = use_cov_calibration
        self.cov_calib_min_scale = cov_calib_min_scale
        self.cov_calib_max_scale = cov_calib_max_scale
        self.use_cov_in_fusion = use_cov_in_fusion
        self.cov_weight_beta = cov_weight_beta
        self.gate_net = nn.Sequential(nn.Linear(hidden_dim + meas_hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init_bias))
        self.cov_calib = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if use_cov_calibration else None

    def _build_h0_gate(self, post_feat, meas_feat):
        h_post = self.post_enc(post_feat)
        h_meas = self.meas_enc(meas_feat)
        gate_soft = torch.sigmoid(self.gate_net(torch.cat([h_post, h_meas], -1)).squeeze(-1))
        gate = gate_soft * post_feat[..., self.valid_idx]
        if self.use_meas_in_representation:
            meas_gate = gate if self.use_gate_on_meas_feature else post_feat[..., self.valid_idx]
            h_meas_repr = h_meas * meas_gate.unsqueeze(-1)
        else:
            h_meas_repr = torch.zeros_like(h_meas)
        h0 = self.fuse_proj(torch.cat([h_post, h_meas_repr], -1))
        return h0, {"gate": gate, "gate_soft": gate_soft, "h_post": h_post, "h_meas": h_meas}

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        if meas_feat is None:
            raise ValueError("PostMeasSoftGateFusion requires meas_feat.")
        h0, aux = self._build_h0_gate(post_feat, meas_feat)
        h1, a = self._graph(h0)
        base_logits = self.node_logit(h1).squeeze(-1) / self.base_logit_temperature
        gate_bias = self.gate_weight_alpha * torch.log(aux["gate"].clamp_min(self.gate_eps))
        cov_scale = torch.ones_like(base_logits)
        if self.cov_calib is not None:
            cov_scale = (self.cov_calib_min_scale + F.softplus(self.cov_calib(h1).squeeze(-1))).clamp(max=self.cov_calib_max_scale)
        cov_bias = -float(self.cov_weight_beta) * torch.log(cov_scale.clamp_min(self.gate_eps))
        reliability_logits = base_logits + gate_bias + cov_bias
        aux.update({
            "attn_matrix": a,
            "base_weight_logits": base_logits,
            "raw_weight_logits": reliability_logits,
            "gate_weight_bias": gate_bias,
            "cov_weight_bias": cov_bias,
            "reliability_logits": reliability_logits,
            "cov_scale": cov_scale,
        })
        cov_scale_for_fusion = cov_scale if self.use_cov_in_fusion else None
        return self._decode(
            post_feat,
            h1,
            mask,
            reliability_logits,
            return_weights,
            aux,
            cov_scale=cov_scale_for_fusion,
            weight_uniform_mix=self.weight_uniform_mix,
        )


class SkepticalNeuralFusionA(_GraphFusionCore):
    def __init__(
        self,
        post_in_dim=9,
        meas_in_dim=18,
        hidden_dim=64,
        meas_hidden_dim=64,
        gate_hidden_dim=64,
        valid_idx=8,
        pos_scale=1000.0,
        vel_scale=30.0,
        gate_init_bias=0.0,
        gate_weight_alpha=1.2,
        gate_eps=1e-4,
        base_logit_temperature=2.0,
        weight_uniform_mix=0.02,
        cov_calib_min_scale=1.0,
        cov_calib_max_scale=30.0,
        cov_weight_beta=0.5,
        cap_min=0.05,
        cap_max=0.85,
        quarantine_penalty=2.0,
        output_fusion_mode="aa_mm",
    ):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.gate_weight_alpha = float(gate_weight_alpha)
        self.gate_eps = float(gate_eps)
        self.base_logit_temperature = max(float(base_logit_temperature), 1e-6)
        self.weight_uniform_mix = max(float(weight_uniform_mix), 0.0)
        self.cov_calib_min_scale = float(cov_calib_min_scale)
        self.cov_calib_max_scale = float(cov_calib_max_scale)
        self.cov_weight_beta = float(cov_weight_beta)
        self.cap_min = float(cap_min)
        self.cap_max = float(cap_max)
        self.quarantine_penalty = float(quarantine_penalty)

        self.post_enc = nn.Sequential(nn.Linear(post_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.meas_enc = nn.Sequential(nn.Linear(meas_in_dim, meas_hidden_dim), nn.ReLU(), nn.Linear(meas_hidden_dim, meas_hidden_dim), nn.ReLU())
        cross_dim = hidden_dim + meas_hidden_dim + 2 * min(hidden_dim, meas_hidden_dim)
        self.cross_exam = nn.Sequential(
            nn.Linear(cross_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reliability_head = nn.Sequential(nn.Linear(hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        self.quarantine_head = nn.Sequential(nn.Linear(hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        self.cap_head = nn.Sequential(nn.Linear(hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        self.cov_calib = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.risk_head = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.reliability_head[-1].bias, float(gate_init_bias))

    def _cross_features(self, h_post: torch.Tensor, h_meas: torch.Tensor) -> torch.Tensor:
        d = min(h_post.shape[-1], h_meas.shape[-1])
        hp = h_post[..., :d]
        hm = h_meas[..., :d]
        return torch.cat([h_post, h_meas, torch.abs(hp - hm), hp * hm], dim=-1)

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        if meas_feat is None:
            raise ValueError("SkepticalNeuralFusionA requires meas_feat.")
        h_post = self.post_enc(post_feat)
        h_meas = self.meas_enc(meas_feat)
        h0 = self.cross_exam(self._cross_features(h_post, h_meas))
        h1, a = self._graph(h0)

        valid = post_feat[..., self.valid_idx]
        reliability_soft = torch.sigmoid(self.reliability_head(h1).squeeze(-1))
        reliability = reliability_soft * valid
        quarantine = torch.sigmoid(self.quarantine_head(h1).squeeze(-1)) * valid
        cov_scale = (self.cov_calib_min_scale + F.softplus(self.cov_calib(h1).squeeze(-1))).clamp(max=self.cov_calib_max_scale)

        cap_raw = torch.sigmoid(self.cap_head(h1).squeeze(-1))
        cap = self.cap_min + (self.cap_max - self.cap_min) * cap_raw
        cap = cap * (1.0 - 0.65 * quarantine).clamp_min(0.25)

        base_logits = self.node_logit(h1).squeeze(-1) / self.base_logit_temperature
        reliability_bias = self.gate_weight_alpha * torch.log(reliability.clamp_min(self.gate_eps))
        cov_bias = -self.cov_weight_beta * torch.log(cov_scale.clamp_min(self.gate_eps))
        quarantine_bias = -self.quarantine_penalty * quarantine
        reliability_logits = base_logits + reliability_bias + cov_bias + quarantine_bias

        risk_context = torch.stack([
            1.0 - reliability_soft,
            quarantine,
            torch.log(cov_scale.clamp_min(self.gate_eps)),
        ], dim=-1)
        risk_node = torch.sigmoid(self.risk_head(torch.cat([h1, risk_context], dim=-1)).squeeze(-1))

        aux = {
            "attn_matrix": a,
            "h_post": h_post,
            "h_meas": h_meas,
            "gate": reliability,
            "gate_soft": reliability_soft,
            "reliability": reliability,
            "quarantine": quarantine,
            "weight_cap": cap,
            "risk_node": risk_node,
            "base_weight_logits": base_logits,
            "raw_weight_logits": reliability_logits,
            "gate_weight_bias": reliability_bias,
            "cov_weight_bias": cov_bias,
            "quarantine_weight_bias": quarantine_bias,
            "reliability_logits": reliability_logits,
            "cov_scale": cov_scale,
        }
        out = self._decode(
            post_feat,
            h1,
            mask,
            reliability_logits,
            return_weights,
            aux,
            cov_scale=cov_scale,
            weight_uniform_mix=self.weight_uniform_mix,
            weight_cap=cap,
        )
        if out.weights is not None:
            out.aux["risk"] = (out.weights * risk_node).sum(dim=0 if post_feat.dim() == 2 else 1)
        return out


class Phase1RRGCF(_GraphFusionCore):
    """Corrected RGCF for Phase1R: evidence informs reliability, but only track nodes fuse."""

    def __init__(
        self,
        post_in_dim=9,
        meas_in_dim=18,
        evidence_in_dim=16,
        hidden_dim=64,
        meas_hidden_dim=64,
        gate_hidden_dim=64,
        valid_idx=8,
        pos_scale=1000.0,
        vel_scale=30.0,
        gate_init_bias=0.5,
        gate_weight_alpha=1.0,
        gate_eps=1e-4,
        base_logit_temperature=1.5,
        weight_uniform_mix=0.02,
        cov_calib_min_scale=1.0,
        cov_calib_max_scale=25.0,
        cov_weight_beta=0.35,
        output_fusion_mode="info_diag",
    ):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.gate_weight_alpha = float(gate_weight_alpha)
        self.gate_eps = float(gate_eps)
        self.base_logit_temperature = max(float(base_logit_temperature), 1e-6)
        self.weight_uniform_mix = max(float(weight_uniform_mix), 0.0)
        self.cov_calib_min_scale = float(cov_calib_min_scale)
        self.cov_calib_max_scale = float(cov_calib_max_scale)
        self.cov_weight_beta = float(cov_weight_beta)

        self.post_enc = nn.Sequential(nn.Linear(post_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.meas_enc = nn.Sequential(nn.Linear(meas_in_dim, meas_hidden_dim), nn.ReLU(), nn.Linear(meas_hidden_dim, meas_hidden_dim), nn.ReLU())
        self.evidence_enc = nn.Sequential(nn.Linear(evidence_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.evidence_score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        cross_dim = hidden_dim + meas_hidden_dim + hidden_dim + 2 * min(hidden_dim, meas_hidden_dim)
        self.cross = nn.Sequential(nn.Linear(cross_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.reliability_head = nn.Sequential(nn.Linear(hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        self.cov_calib = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.reliability_head[-1].bias, float(gate_init_bias))

    def _pool_evidence(self, evidence_feat, evidence_mask, like):
        if evidence_feat is None:
            shape = (*like.shape[:-1], self.hidden_dim)
            if like.dim() == 2:
                shape = (self.hidden_dim,)
            return like.new_zeros(shape)
        h_ev = self.evidence_enc(evidence_feat)
        if evidence_mask is None:
            evidence_mask = torch.ones(h_ev.shape[:-1], device=h_ev.device, dtype=h_ev.dtype)
        score = self.evidence_score(h_ev).squeeze(-1) + (evidence_mask - 1.0) * 1e9
        alpha = torch.softmax(score, dim=-1)
        pooled = (alpha.unsqueeze(-1) * h_ev).sum(dim=-2)
        return pooled

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        if meas_feat is None:
            raise ValueError("Phase1RRGCF requires track meas_feat.")
        h_post = self.post_enc(post_feat)
        h_meas = self.meas_enc(meas_feat)
        h_ev = self._pool_evidence(evidence_feat, evidence_mask, h_post)
        if h_post.dim() == 3:
            h_ev_node = h_ev.unsqueeze(1).expand(-1, h_post.shape[1], -1)
        else:
            h_ev_node = h_ev.unsqueeze(0).expand(h_post.shape[0], -1)
        d = min(h_post.shape[-1], h_meas.shape[-1])
        cross_feat = torch.cat([
            h_post,
            h_meas,
            h_ev_node,
            torch.abs(h_post[..., :d] - h_meas[..., :d]),
            h_post[..., :d] * h_meas[..., :d],
        ], dim=-1)
        h0 = self.cross(cross_feat)
        h1, a = self._graph(h0)

        valid = post_feat[..., self.valid_idx]
        reliability_soft = torch.sigmoid(self.reliability_head(h1).squeeze(-1))
        reliability = reliability_soft * valid
        cov_scale = (self.cov_calib_min_scale + F.softplus(self.cov_calib(h1).squeeze(-1))).clamp(max=self.cov_calib_max_scale)
        base_logits = self.node_logit(h1).squeeze(-1) / self.base_logit_temperature
        reliability_bias = self.gate_weight_alpha * torch.log(reliability.clamp_min(self.gate_eps))
        cov_bias = -self.cov_weight_beta * torch.log(cov_scale.clamp_min(self.gate_eps))
        reliability_logits = base_logits + reliability_bias + cov_bias

        aux = {
            "attn_matrix": a,
            "h_post": h_post,
            "h_meas": h_meas,
            "h_evidence": h_ev,
            "gate": reliability,
            "gate_soft": reliability_soft,
            "reliability": reliability,
            "base_weight_logits": base_logits,
            "raw_weight_logits": reliability_logits,
            "gate_weight_bias": reliability_bias,
            "cov_weight_bias": cov_bias,
            "reliability_logits": reliability_logits,
            "cov_scale": cov_scale,
        }
        return self._decode(
            post_feat,
            h1,
            mask,
            reliability_logits,
            return_weights,
            aux,
            cov_scale=cov_scale,
            weight_uniform_mix=self.weight_uniform_mix,
        )


class MeasurementEvaluatedRGCFA0(_GraphFusionCore):
    """ME-A0 heterogeneous graph: P nodes fuse, M nodes evaluate evidence."""

    def __init__(
        self,
        post_in_dim=9,
        meas_in_dim=18,
        evidence_in_dim=16,
        hidden_dim=64,
        meas_hidden_dim=64,
        gate_hidden_dim=64,
        valid_idx=8,
        pos_scale=1000.0,
        vel_scale=30.0,
        gate_init_bias=0.5,
        gate_weight_alpha=1.0,
        gate_eps=1e-4,
        base_logit_temperature=1.5,
        weight_uniform_mix=0.02,
        cov_calib_min_scale=1.0,
        cov_calib_max_scale=25.0,
        cov_weight_beta=0.35,
        use_mm_attention=True,
        use_mp_attention=True,
        output_fusion_mode="info_diag",
    ):
        super().__init__(hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
        self.meas_in_dim = int(meas_in_dim)
        self.evidence_in_dim = int(evidence_in_dim)
        self.gate_weight_alpha = float(gate_weight_alpha)
        self.gate_eps = float(gate_eps)
        self.base_logit_temperature = max(float(base_logit_temperature), 1e-6)
        self.weight_uniform_mix = max(float(weight_uniform_mix), 0.0)
        self.cov_calib_min_scale = float(cov_calib_min_scale)
        self.cov_calib_max_scale = float(cov_calib_max_scale)
        self.cov_weight_beta = float(cov_weight_beta)
        self.use_mm_attention = bool(use_mm_attention)
        self.use_mp_attention = bool(use_mp_attention)

        self.post_enc = nn.Sequential(nn.Linear(post_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.meas_enc = nn.Sequential(nn.Linear(meas_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.measurement_role_embedding = nn.Embedding(2, hidden_dim)
        self.p_type_embedding = nn.Parameter(torch.zeros(hidden_dim))

        self.mm_attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.mm_upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mp_attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.mp_upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.pp_attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.pp_upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.reliability_head = nn.Sequential(nn.Linear(hidden_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1))
        self.cov_calib = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.constant_(self.reliability_head[-1].bias, float(gate_init_bias))

    def _pad_evidence(self, evidence_feat: torch.Tensor | None, post_feat: torch.Tensor) -> torch.Tensor:
        if evidence_feat is None:
            if post_feat.dim() == 3:
                return post_feat.new_zeros((post_feat.shape[0], 2, self.meas_in_dim))
            return post_feat.new_zeros((2, self.meas_in_dim))
        if evidence_feat.shape[-1] == self.meas_in_dim:
            return evidence_feat
        if evidence_feat.shape[-1] > self.meas_in_dim:
            return evidence_feat[..., : self.meas_in_dim]
        pad = self.meas_in_dim - evidence_feat.shape[-1]
        return F.pad(evidence_feat, (0, pad))

    def _measurement_nodes(self, meas_feat, evidence_feat, evidence_mask, post_feat):
        if meas_feat is None:
            raise ValueError("MeasurementEvaluatedRGCFA0 requires track meas_feat.")
        ev = self._pad_evidence(evidence_feat, post_feat)
        m_feat = torch.cat([meas_feat, ev], dim=-2)
        if meas_feat.dim() == 3:
            track_mask = post_feat[..., self.valid_idx].clamp_min(0.0)
            if evidence_mask is None:
                evidence_mask = torch.ones(ev.shape[:-1], device=ev.device, dtype=ev.dtype)
            m_mask = torch.cat([track_mask, evidence_mask.to(track_mask.dtype)], dim=1)
            role = torch.cat([
                torch.zeros(meas_feat.shape[1], device=meas_feat.device, dtype=torch.long),
                torch.ones(ev.shape[1], device=meas_feat.device, dtype=torch.long),
            ], dim=0)
            role_emb = self.measurement_role_embedding(role).unsqueeze(0)
        else:
            track_mask = post_feat[..., self.valid_idx].clamp_min(0.0)
            if evidence_mask is None:
                evidence_mask = torch.ones(ev.shape[:-1], device=ev.device, dtype=ev.dtype)
            m_mask = torch.cat([track_mask, evidence_mask.to(track_mask.dtype)], dim=0)
            role = torch.cat([
                torch.zeros(meas_feat.shape[0], device=meas_feat.device, dtype=torch.long),
                torch.ones(ev.shape[0], device=meas_feat.device, dtype=torch.long),
            ], dim=0)
            role_emb = self.measurement_role_embedding(role)
        h_m = self.meas_enc(m_feat) + role_emb
        return h_m, m_mask

    def _masked_self_graph(self, h0: torch.Tensor, mask: torch.Tensor | None, attn, upd, remove_self=True):
        if mask is None:
            mask = torch.ones(h0.shape[:-1], device=h0.device, dtype=h0.dtype)
        if h0.dim() == 2:
            n = h0.size(0)
            hi = h0.unsqueeze(1).expand(n, n, -1)
            hj = h0.unsqueeze(0).expand(n, n, -1)
            e = attn(torch.cat([hi, hj], -1)).squeeze(-1)
            e = e + (mask.unsqueeze(0) - 1.0) * 1e9
            if remove_self and n > 1:
                e = e + torch.eye(n, device=h0.device, dtype=e.dtype) * (-1e9)
            a = torch.softmax(e, dim=1)
            h1 = upd(torch.cat([h0, a @ h0], -1))
            return h1, a
        b, n, _ = h0.shape
        hi = h0.unsqueeze(2).expand(b, n, n, -1)
        hj = h0.unsqueeze(1).expand(b, n, n, -1)
        e = attn(torch.cat([hi, hj], -1)).squeeze(-1)
        e = e + (mask.unsqueeze(1) - 1.0) * 1e9
        if remove_self and n > 1:
            e = e + torch.eye(n, device=h0.device, dtype=e.dtype).unsqueeze(0) * (-1e9)
        a = torch.softmax(e, dim=2)
        h1 = upd(torch.cat([h0, torch.matmul(a, h0)], -1))
        return h1, a

    def _cross_measurement_to_posterior(self, h_p: torch.Tensor, h_m: torch.Tensor, m_mask: torch.Tensor | None, mp_pair_feat: torch.Tensor | None = None):
        if m_mask is None:
            m_mask = torch.ones(h_m.shape[:-1], device=h_m.device, dtype=h_m.dtype)
        if h_p.dim() == 2:
            p, m = h_p.shape[0], h_m.shape[0]
            hp = h_p.unsqueeze(1).expand(p, m, -1)
            hm = h_m.unsqueeze(0).expand(p, m, -1)
            e = self.mp_attn(torch.cat([hp, hm], -1)).squeeze(-1)
            e = e + (m_mask.unsqueeze(0) - 1.0) * 1e9
            a = torch.softmax(e, dim=1)
            ctx = a @ h_m
            return self.mp_upd(torch.cat([h_p, ctx], -1)), a
        b, p, _ = h_p.shape
        m = h_m.shape[1]
        hp = h_p.unsqueeze(2).expand(b, p, m, -1)
        hm = h_m.unsqueeze(1).expand(b, p, m, -1)
        e = self.mp_attn(torch.cat([hp, hm], -1)).squeeze(-1)
        e = e + (m_mask.unsqueeze(1) - 1.0) * 1e9
        a = torch.softmax(e, dim=2)
        ctx = torch.matmul(a, h_m)
        return self.mp_upd(torch.cat([h_p, ctx], -1)), a

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None, mp_pair_feat=None):
        h_p = self.post_enc(post_feat) + self.p_type_embedding
        h_m, m_mask = self._measurement_nodes(meas_feat, evidence_feat, evidence_mask, post_feat)

        if self.use_mm_attention:
            h_m, mm_a = self._masked_self_graph(h_m, m_mask, self.mm_attn, self.mm_upd)
        else:
            mm_a = h_m.new_zeros((*h_m.shape[:-1], h_m.shape[-2]))

        if self.use_mp_attention:
            h_p0, mp_a = self._cross_measurement_to_posterior(h_p, h_m, m_mask, mp_pair_feat=mp_pair_feat)
        else:
            h_p0 = h_p
            if h_p.dim() == 3:
                mp_a = h_p.new_zeros((h_p.shape[0], h_p.shape[1], h_m.shape[1]))
            else:
                mp_a = h_p.new_zeros((h_p.shape[0], h_m.shape[0]))

        p_mask = mask if mask is not None else post_feat[..., self.valid_idx].clamp_min(0.0)
        h1, pp_a = self._masked_self_graph(h_p0, p_mask, self.pp_attn, self.pp_upd)

        valid = post_feat[..., self.valid_idx]
        reliability_soft = torch.sigmoid(self.reliability_head(h1).squeeze(-1))
        reliability = reliability_soft * valid
        cov_scale = (self.cov_calib_min_scale + F.softplus(self.cov_calib(h1).squeeze(-1))).clamp(max=self.cov_calib_max_scale)
        base_logits = self.node_logit(h1).squeeze(-1) / self.base_logit_temperature
        reliability_bias = self.gate_weight_alpha * torch.log(reliability.clamp_min(self.gate_eps))
        cov_bias = -self.cov_weight_beta * torch.log(cov_scale.clamp_min(self.gate_eps))
        reliability_logits = base_logits + reliability_bias + cov_bias

        aux = {
            "attn_matrix": pp_a,
            "mm_attn": mm_a,
            "mp_attn": mp_a,
            "h_post": h_p,
            "h_meas": h_m,
            "measurement_mask": m_mask,
            "gate": reliability,
            "gate_soft": reliability_soft,
            "reliability": reliability,
            "base_weight_logits": base_logits,
            "raw_weight_logits": reliability_logits,
            "gate_weight_bias": reliability_bias,
            "cov_weight_bias": cov_bias,
            "reliability_logits": reliability_logits,
            "cov_scale": cov_scale,
        }
        return self._decode(
            post_feat,
            h1,
            mask,
            reliability_logits,
            return_weights,
            aux,
            cov_scale=cov_scale,
            weight_uniform_mix=self.weight_uniform_mix,
        )


class MeasurementEvaluatedRGCFA0Directional(MeasurementEvaluatedRGCFA0):
    """ME-A0D: pair-aware M->P attention using per-track evidence residuals."""

    def __init__(
        self,
        post_in_dim=9,
        meas_in_dim=18,
        evidence_in_dim=16,
        pair_dim=8,
        hidden_dim=64,
        meas_hidden_dim=64,
        gate_hidden_dim=64,
        valid_idx=8,
        pos_scale=1000.0,
        vel_scale=30.0,
        gate_init_bias=0.5,
        gate_weight_alpha=1.0,
        gate_eps=1e-4,
        base_logit_temperature=1.5,
        weight_uniform_mix=0.02,
        cov_calib_min_scale=1.0,
        cov_calib_max_scale=25.0,
        cov_weight_beta=0.35,
        use_mm_attention=True,
        use_mp_attention=True,
        identity_bias_init=0.5,
        evidence_residual_bias_init=0.25,
        output_fusion_mode="info_diag",
    ):
        super().__init__(
            post_in_dim=post_in_dim,
            meas_in_dim=meas_in_dim,
            evidence_in_dim=evidence_in_dim,
            hidden_dim=hidden_dim,
            meas_hidden_dim=meas_hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            valid_idx=valid_idx,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            gate_init_bias=gate_init_bias,
            gate_weight_alpha=gate_weight_alpha,
            gate_eps=gate_eps,
            base_logit_temperature=base_logit_temperature,
            weight_uniform_mix=weight_uniform_mix,
            cov_calib_min_scale=cov_calib_min_scale,
            cov_calib_max_scale=cov_calib_max_scale,
            cov_weight_beta=cov_weight_beta,
            use_mm_attention=use_mm_attention,
            use_mp_attention=use_mp_attention,
            output_fusion_mode=output_fusion_mode,
        )
        self.pair_dim = int(pair_dim)
        self.pair_enc = nn.Sequential(nn.Linear(self.pair_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mp_attn = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.mp_msg = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mp_upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.identity_bias = nn.Parameter(torch.tensor(float(identity_bias_init)))
        self.evidence_residual_bias = nn.Parameter(torch.tensor(float(evidence_residual_bias_init)))

    def _default_pair_feat(self, h_p: torch.Tensor, h_m: torch.Tensor) -> torch.Tensor:
        if h_p.dim() == 2:
            p = h_p.shape[0]
            m = h_m.shape[0]
            out = h_p.new_zeros((p, m, self.pair_dim))
            for pi in range(min(p, m, 3)):
                out[pi, pi, 0] = 1.0
            if m > 3:
                out[:, 3:, 2] = 1.0
            out[..., 6:8] = 1.0
            return out
        b, p, _ = h_p.shape
        m = h_m.shape[1]
        out = h_p.new_zeros((b, p, m, self.pair_dim))
        for pi in range(min(p, m, 3)):
            out[:, pi, pi, 0] = 1.0
        if m > 3:
            out[:, :, 3:, 2] = 1.0
        out[..., 6:8] = 1.0
        return out

    def _cross_measurement_to_posterior(self, h_p: torch.Tensor, h_m: torch.Tensor, m_mask: torch.Tensor | None, mp_pair_feat: torch.Tensor | None = None):
        if m_mask is None:
            m_mask = torch.ones(h_m.shape[:-1], device=h_m.device, dtype=h_m.dtype)
        if mp_pair_feat is None:
            mp_pair_feat = self._default_pair_feat(h_p, h_m)
        mp_pair_feat = mp_pair_feat.to(device=h_p.device, dtype=h_p.dtype)

        if h_p.dim() == 2:
            p, m = h_p.shape[0], h_m.shape[0]
            hp = h_p.unsqueeze(1).expand(p, m, -1)
            hm = h_m.unsqueeze(0).expand(p, m, -1)
            pair_h = self.pair_enc(mp_pair_feat)
            e = self.mp_attn(torch.cat([hp, hm, pair_h], -1)).squeeze(-1)
            e = e + self.identity_bias * mp_pair_feat[..., 0]
            e = e + self.evidence_residual_bias * mp_pair_feat[..., 2] * mp_pair_feat[..., 5]
            e = e + (m_mask.unsqueeze(0) - 1.0) * 1e9
            a = torch.softmax(e, dim=1)
            msg = self.mp_msg(torch.cat([hm, pair_h], -1))
            ctx = (a.unsqueeze(-1) * msg).sum(dim=1)
            return self.mp_upd(torch.cat([h_p, ctx], -1)), a

        b, p, _ = h_p.shape
        m = h_m.shape[1]
        hp = h_p.unsqueeze(2).expand(b, p, m, -1)
        hm = h_m.unsqueeze(1).expand(b, p, m, -1)
        pair_h = self.pair_enc(mp_pair_feat)
        e = self.mp_attn(torch.cat([hp, hm, pair_h], -1)).squeeze(-1)
        e = e + self.identity_bias * mp_pair_feat[..., 0]
        e = e + self.evidence_residual_bias * mp_pair_feat[..., 2] * mp_pair_feat[..., 5]
        e = e + (m_mask.unsqueeze(1) - 1.0) * 1e9
        a = torch.softmax(e, dim=2)
        msg = self.mp_msg(torch.cat([hm, pair_h], -1))
        ctx = (a.unsqueeze(-1) * msg).sum(dim=2)
        return self.mp_upd(torch.cat([h_p, ctx], -1)), a

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None, mp_pair_feat=None):
        out = super().forward(
            post_feat=post_feat,
            mask=mask,
            meas_feat=meas_feat,
            return_weights=return_weights,
            post_win=post_win,
            meas_win=meas_win,
            evidence_feat=evidence_feat,
            evidence_mask=evidence_mask,
            mp_pair_feat=mp_pair_feat,
        )
        if return_weights and mp_pair_feat is not None:
            out.aux["mp_pair_feat"] = mp_pair_feat
        return out


class PostMeasWindowDirectFusion(PostMeasDirectFusion):
    def __init__(self, post_in_dim=9, meas_in_dim=18, hidden_dim=64, meas_hidden_dim=64, window_size=6, valid_idx=8, pos_scale=1000.0, vel_scale=30.0, output_fusion_mode="info_diag"):
        super().__init__(post_in_dim, meas_in_dim, hidden_dim, meas_hidden_dim, valid_idx, pos_scale, vel_scale, output_fusion_mode)
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

    def forward(self, post_feat, mask=None, meas_feat=None, return_weights=False, post_win=None, meas_win=None, evidence_feat=None, evidence_mask=None):
        if post_win is None or meas_win is None:
            raise ValueError("PostMeasWindowDirectFusion requires post_win and meas_win.")
        h1, a = self._graph(self._encode_window(post_win, meas_win))
        raw = self.node_logit(h1).squeeze(-1)
        return self._decode(post_feat, h1, mask, raw, return_weights, {"attn_matrix": a})
