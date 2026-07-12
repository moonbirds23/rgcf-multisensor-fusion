import unittest

import torch

from models.gnn_fusion import OriginalGNNFusion, _fuse_info_diag, _fuse_info_diag_with_cov


class InfoDiagCovarianceTests(unittest.TestCase):
    def test_matches_manual_information_sum(self):
        xhat = torch.tensor([[0.0, 10.0], [4.0, 2.0]])
        pdiag = torch.tensor([[1.0, 4.0], [4.0, 1.0]])
        weights = torch.tensor([0.25, 0.75])

        pred, covariance = _fuse_info_diag_with_cov(xhat, pdiag, weights)

        expected_information = weights[:, None] / pdiag
        self.assertTrue(torch.allclose(pred, (expected_information * xhat).sum(0) / expected_information.sum(0)))
        self.assertTrue(torch.allclose(covariance, 1.0 / expected_information.sum(0)))
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertTrue((covariance > 0.0).all())

    def test_zero_mask_weight_keeps_prediction(self):
        xhat = torch.tensor([[[1.0, 5.0], [9.0, 9.0]]])
        pdiag = torch.tensor([[[1.0, 4.0], [0.01, 0.01]]])
        masked_weights = torch.tensor([[1.0, 0.0]])

        legacy_pred = _fuse_info_diag(xhat, pdiag, masked_weights)
        pred, covariance = _fuse_info_diag_with_cov(xhat, pdiag, masked_weights)

        self.assertTrue(torch.equal(pred, legacy_pred))
        self.assertTrue(torch.allclose(pred, xhat[:, 0, :]))
        self.assertTrue(torch.allclose(covariance, pdiag[:, 0, :]))
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertTrue((covariance > 0.0).all())

    def test_info_diag_model_exposes_covariance_only_with_weights(self):
        model = OriginalGNNFusion(hidden_dim=4, output_fusion_mode="info_diag")
        post_feat = torch.zeros((1, 2, 9))
        post_feat[0, :, 0:4] = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        post_feat[0, :, 4:8] = torch.log1p(torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]))
        post_feat[..., 8] = 1.0

        plain = model(post_feat, return_weights=False)
        weighted = model(post_feat, return_weights=True)

        self.assertTrue(torch.equal(plain.pred, weighted.pred))
        self.assertIn("fused_cov_diag", weighted.aux)
        self.assertTrue(torch.isfinite(weighted.aux["fused_cov_diag"]).all())
        self.assertTrue((weighted.aux["fused_cov_diag"] > 0.0).all())
