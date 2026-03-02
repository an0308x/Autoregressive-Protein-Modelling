"""
Unit tests for PAR components.

Run: python tests.py
"""

import sys
import torch
import unittest

sys.path.insert(0, ".")


class TestDownsampling(unittest.TestCase):
    def test_interpolate_coords_shape(self):
        from models.downsampling import interpolate_coords
        coords = torch.randn(4, 200, 3)
        out = interpolate_coords(coords, 64)
        self.assertEqual(out.shape, (4, 64, 3))

    def test_upsample_shape(self):
        from models.downsampling import upsample_coords
        x = torch.randn(2, 64, 3)
        out = upsample_coords(x, 128)
        self.assertEqual(out.shape, (2, 128, 3))

    def test_multiscale_downsample_n_scales(self):
        from models.downsampling import multiscale_downsample
        B, L = 2, 200
        backbone = torch.randn(B, L, 3)
        lengths  = torch.tensor([L, L])
        scales   = multiscale_downsample(backbone, [64, 128, 256], lengths)
        # L=200: sizes become [64, 128, 200]
        self.assertEqual(len(scales), 3)
        self.assertEqual(scales[0].shape, (B, 64, 3))
        self.assertEqual(scales[1].shape, (B, 128, 3))
        self.assertEqual(scales[2].shape, (B, 200, 3))

    def test_spatial_preservation(self):
        from models.downsampling import verify_downsampling_preserves_spatial_relations
        coords = torch.randn(200, 3).cumsum(0)
        rmse, lddt = verify_downsampling_preserves_spatial_relations(coords, 64)
        # Paper claims LDDT=1 across all scales; in practice lddt should be high
        self.assertGreater(lddt, 0.9, f"lDDT too low: {lddt}")
        self.assertLess(rmse, 5.0, f"RMSE too high: {rmse}")


class TestARTransformer(unittest.TestCase):
    def setUp(self):
        self.ar = __import__(
            "models.ar_transformer", fromlist=["ARTransformer"]
        ).ARTransformer(
            d_model=64, d_cond=32, n_heads=4, n_layers=2, n_scales=3
        )
        self.ar.eval()

    def test_first_scale_output_shape(self):
        """z^1 should have shape (B, bos_size, d_cond)."""
        B, bos_size = 2, 64
        z1, _ = self.ar.forward(
            prior_scales=[], target_scale_size=bos_size,
            protein_length=200, scale_idx=0, bos_size=bos_size,
        )
        self.assertEqual(z1.shape, (B, bos_size, 32))

    def test_second_scale_output_shape(self):
        B, size1, size2 = 2, 64, 128
        x1 = torch.randn(B, size1, 3)
        z2, _ = self.ar.forward(
            prior_scales=[x1], target_scale_size=size2,
            protein_length=200, scale_idx=1, bos_size=size1,
        )
        self.assertEqual(z2.shape, (B, size2, 32))

    def test_forward_all_scales(self):
        B = 2
        scales = [torch.randn(B, s, 3) for s in [64, 128, 200]]
        zs = self.ar.forward_all_scales(scales, protein_length=200, bos_size=64)
        self.assertEqual(len(zs), 3)
        for z, s in zip(zs, [64, 128, 200]):
            self.assertEqual(z.shape, (B, s, 32))


class TestFlowDecoder(unittest.TestCase):
    def setUp(self):
        self.dec = __import__(
            "models.flow_decoder", fromlist=["FlowDecoder"]
        ).FlowDecoder(
            d_model=64, d_cond=32, d_time=32, n_heads=4, n_layers=2, n_scales=3
        )
        self.dec.eval()

    def test_velocity_shape(self):
        B, S = 2, 128
        x_t = torch.randn(B, S, 3)
        t   = torch.rand(B)
        z   = torch.randn(B, S, 32)
        v   = self.dec(x_t, t, z, scale_idx=1)
        self.assertEqual(v.shape, (B, S, 3))

    def test_self_conditioning(self):
        B, S = 2, 64
        x_t = torch.randn(B, S, 3)
        t   = torch.rand(B)
        z   = torch.randn(B, S, 32)
        sc  = torch.randn(B, S, 3)
        v   = self.dec(x_t, t, z, scale_idx=0, self_cond=sc)
        self.assertEqual(v.shape, (B, S, 3))


class TestLoss(unittest.TestCase):
    def test_flow_matching_loss(self):
        from training.loss import flow_matching_loss_scale, build_noisy_batch
        B, S = 4, 128
        v    = torch.randn(B, S, 3)
        x    = torch.randn(B, S, 3)
        eps  = torch.randn(B, S, 3)
        loss = flow_matching_loss_scale(v, x, eps)
        self.assertTrue(loss.item() > 0)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_build_noisy_batch(self):
        from training.loss import build_noisy_batch
        scales = [torch.randn(2, s, 3) for s in [64, 128]]
        x_t, eps, t = build_noisy_batch(scales, torch.device("cpu"))
        self.assertEqual(len(x_t), 2)
        self.assertEqual(x_t[0].shape, (2, 64, 3))


class TestExposureBias(unittest.TestCase):
    def test_ncl_shape(self):
        from training.exposure_bias import noisy_context_learning
        x = torch.randn(4, 64, 3)
        x_ncl = noisy_context_learning(x)
        self.assertEqual(x_ncl.shape, x.shape)

    def test_ebm_teacher_forcing(self):
        from training.exposure_bias import ExposureBiasMitigation
        ebm = ExposureBiasMitigation(use_ncl=False, use_ss=False)
        scales = [torch.randn(2, s, 3) for s in [64, 128, 256]]
        ctx    = ebm(scales)
        for gt, c in zip(scales, ctx):
            self.assertTrue(torch.allclose(gt, c))

    def test_ebm_ncl(self):
        from training.exposure_bias import ExposureBiasMitigation
        ebm = ExposureBiasMitigation(use_ncl=True, use_ss=False)
        scales = [torch.randn(2, s, 3) for s in [64, 128, 256]]
        ctx    = ebm(scales)
        # Should be different from gt due to NCL noise
        self.assertFalse(torch.allclose(scales[0], ctx[0]))


class TestSampling(unittest.TestCase):
    def test_ode_shape(self):
        from utils.sampling import ode_sample
        B, S = 2, 64
        def vfn(x, t, sc=None): return torch.randn_like(x)
        x = torch.randn(B, S, 3)
        out = ode_sample(vfn, x, n_steps=5)
        self.assertEqual(out.shape, (B, S, 3))

    def test_sde_shape(self):
        from utils.sampling import sde_sample
        B, S = 2, 64
        def vfn(x, t, sc=None): return torch.randn_like(x)
        x = torch.randn(B, S, 3)
        out = sde_sample(vfn, x, n_steps=5, gamma=0.3)
        self.assertEqual(out.shape, (B, S, 3))


class TestPARModel(unittest.TestCase):
    """Smoke test for the full PAR model (tiny config)."""

    def setUp(self):
        from models.par import PAR
        self.model = PAR(
            scale_sizes  = [16, 32, 64],
            ar_d_model   = 64,
            ar_d_cond    = 32,
            ar_n_heads   = 4,
            ar_n_layers  = 2,
            dec_d_model  = 64,
            dec_n_heads  = 4,
            dec_n_layers = 2,
            n_scales     = 3,
            use_ncl      = True,
            use_ss       = False,   # skip SS for speed in unit tests
        )
        self.model.eval()

    def test_forward_pass(self):
        B, L = 2, 80
        backbone = torch.randn(B, L, 3)
        lengths  = torch.tensor([L, L])
        out = self.model(backbone, lengths)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].item() > 0)
        print(f"  [OK] loss={out['loss'].item():.4f}")

    def test_generation(self):
        backbone, intermediates = self.model.generate(
            target_length=64, n_samples=2,
            sampling_modes=["ode", "ode", "ode"],
            n_steps_per_scale=[3, 3, 3],
            gamma=0.3,
        )
        self.assertEqual(backbone.shape, (2, 64, 3))
        self.assertEqual(len(intermediates), 3)
        print(f"  [OK] generated {backbone.shape}")

    def test_prompted_generation(self):
        prompt = torch.randn(1, 16, 3)
        out = self.model.prompted_generation(
            prompt=prompt, target_length=64, n_samples=2, n_steps=3
        )
        self.assertEqual(out.shape, (2, 64, 3))
        print(f"  [OK] prompted {out.shape}")

    def test_parameter_count(self):
        counts = self.model.count_parameters()
        print(f"  [OK] params: {counts}")
        self.assertGreater(counts["total"], 0)


if __name__ == "__main__":
    print("=" * 60)
    print("PAR Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
