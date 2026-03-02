"""
PAR Sampling Script

Usage:
    # Unconditional generation (100 samples each at L=50,100,150,200,250)
    python sample.py --checkpoint checkpoints/par_400m.pt \
                     --lengths 50 100 150 200 250 --n_samples 100

    # Prompted generation (from a .npy point cloud)
    python sample.py --checkpoint checkpoints/par_400m.pt \
                     --mode prompted --prompt prompt_16pts.npy \
                     --target_length 200 --n_samples 4

    # Motif scaffolding
    python sample.py --checkpoint checkpoints/par_400m.pt \
                     --mode motif --motif_coords motif.npy \
                     --motif_mask motif_mask.npy \
                     --target_length 200 --n_samples 4
"""

import argparse
import os
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--output_dir",     type=str, default="outputs")
    parser.add_argument("--mode",           type=str, default="unconditional",
                        choices=["unconditional", "prompted", "motif"])
    # Unconditional
    parser.add_argument("--lengths",        type=int, nargs="+",
                        default=[50, 100, 150, 200, 250])
    parser.add_argument("--n_samples",      type=int, default=100)
    # Prompted
    parser.add_argument("--prompt",         type=str, default=None,
                        help=".npy file with (P, 3) prompt coords")
    parser.add_argument("--target_length",  type=int, default=200)
    # Motif scaffolding
    parser.add_argument("--motif_coords",   type=str, default=None)
    parser.add_argument("--motif_mask",     type=str, default=None)
    # Sampling params
    parser.add_argument("--gamma",          type=float, default=0.30)
    parser.add_argument("--n_steps_sde",    type=int,   default=400)
    parser.add_argument("--n_steps_ode",    type=int,   default=2)
    parser.add_argument("--sampling_modes", type=str,   nargs="+",
                        default=["sde", "ode", "ode"],
                        help="Per-scale sampling mode: sde or ode")
    parser.add_argument("--no_self_cond",   action="store_true")
    parser.add_argument("--device",         type=str, default="cuda"
                        if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    from models.par import PAR
    print(f"[Sample] Loading model from {args.checkpoint}...")
    model = PAR.from_pretrained(args.checkpoint).to(device)
    model.eval()
    print(f"[Sample] Model loaded.")

    n_steps_per_scale = []
    for m in args.sampling_modes:
        n_steps_per_scale.append(args.n_steps_sde if m == "sde" else args.n_steps_ode)

    # ---- Unconditional generation ------------------------------------------
    if args.mode == "unconditional":
        for L in args.lengths:
            print(f"[Sample] Generating {args.n_samples} backbones at L={L}...")
            backbone, intermediates = model.generate(
                target_length        = L,
                n_samples            = args.n_samples,
                device               = device,
                gamma                = args.gamma,
                sampling_modes       = args.sampling_modes,
                n_steps_per_scale    = n_steps_per_scale,
                use_self_conditioning= not args.no_self_cond,
            )
            out_path = os.path.join(args.output_dir, f"backbones_L{L}.npy")
            np.save(out_path, backbone.cpu().numpy())
            print(f"  → saved {backbone.shape} to {out_path}")

    # ---- Prompted generation -----------------------------------------------
    elif args.mode == "prompted":
        assert args.prompt is not None, "--prompt required for prompted mode"
        prompt = torch.from_numpy(np.load(args.prompt)).float().to(device)
        print(f"[Sample] Prompted generation from {prompt.shape} → L={args.target_length}")
        backbones = model.prompted_generation(
            prompt                = prompt,
            target_length         = args.target_length,
            n_samples             = args.n_samples,
            gamma                 = args.gamma,
            use_self_conditioning = not args.no_self_cond,
        )
        out_path = os.path.join(args.output_dir, "prompted_backbones.npy")
        np.save(out_path, backbones.cpu().numpy())
        print(f"  → saved {backbones.shape} to {out_path}")

    # ---- Motif scaffolding -------------------------------------------------
    elif args.mode == "motif":
        assert args.motif_coords is not None and args.motif_mask is not None
        motif_coords = torch.from_numpy(np.load(args.motif_coords)).float().to(device)
        motif_mask   = torch.from_numpy(np.load(args.motif_mask)).bool().to(device)
        print(f"[Sample] Motif scaffolding: {motif_coords.shape} motif → L={args.target_length}")
        backbones = model.motif_scaffolding(
            motif_coords   = motif_coords,
            motif_mask     = motif_mask,
            target_length  = args.target_length,
            n_samples      = args.n_samples,
            gamma          = args.gamma,
        )
        out_path = os.path.join(args.output_dir, "scaffolded_backbones.npy")
        np.save(out_path, backbones.cpu().numpy())
        print(f"  → saved {backbones.shape} to {out_path}")

    print("[Sample] Done.")


if __name__ == "__main__":
    main()
