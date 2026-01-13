#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "torch==2.3.*",
#   "torchaudio",
#   "funasr>=1.1.3",
#   "modelscope",
#   "huggingface_hub",
#   "numpy<=1.26.4",
#   "coremltools",
# ]
# ///

"""Numeric sanity-check: PyTorch vs CoreML.

This script is vendored into SenseVoiceSmall-coreml so we don't depend on custom tools living
inside the upstream SenseVoice repo. We *do* still depend on upstream SenseVoice source code
for the model definition (via --sensevoice-repo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def _default_model_dir() -> str:
    return os.environ.get("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sensevoice-repo",
        default=os.environ.get("SENSEVOICE_REPO", "./.upstream/SenseVoice"),
        help="Path to upstream SenseVoice repo (for importing model.py)",
    )
    p.add_argument("--model", default=_default_model_dir())
    p.add_argument("--device", default=os.environ.get("SENSEVOICE_DEVICE", "cpu"))
    p.add_argument("--mlpackage", required=True)
    p.add_argument("--frames", type=int, default=300)
    args = p.parse_args()

    repo = Path(args.sensevoice_repo).resolve()
    if not repo.exists():
        raise SystemExit(f"Missing --sensevoice-repo: {repo}")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        from model import SenseVoiceSmall  # type: ignore
    except Exception as e:
        raise SystemExit(f"Failed to import SenseVoiceSmall from {repo}/model.py: {e}")

    model, _kwargs = SenseVoiceSmall.from_pretrained(model=args.model, device=args.device)
    model.eval()
    rebuilt = model.export(type="torch", quantize=False)
    rebuilt.eval()

    B = 1
    T = int(args.frames)
    F = 560

    speech = torch.randn(B, T, F, dtype=torch.float32)
    speech_lengths = torch.tensor([T], dtype=torch.int32)
    language = torch.tensor([0], dtype=torch.int32)
    textnorm = torch.tensor([15], dtype=torch.int32)

    with torch.no_grad():
        pt_logits, pt_lens = rebuilt(speech, speech_lengths, language, textnorm)

    mlmodel = ct.models.MLModel(args.mlpackage)
    out = mlmodel.predict(
        {
            "speech": speech.numpy(),
            "speech_lengths": speech_lengths.numpy(),
            "language": language.numpy(),
            "textnorm": textnorm.numpy(),
        }
    )

    cm_logits = np.array(out["ctc_logits"], dtype=np.float32)
    cm_lens = np.array(out["encoder_out_lens"], dtype=np.int64)

    pt_logits_np = pt_logits.detach().cpu().numpy().astype(np.float32)
    pt_lens_np = pt_lens.detach().cpu().numpy().astype(np.int64)

    print(f"logits shape: pt={pt_logits_np.shape} coreml={cm_logits.shape}")
    print(f"lens shape:   pt={pt_lens_np.shape} coreml={cm_lens.shape}")

    max_abs_diff = float(np.max(np.abs(pt_logits_np - cm_logits)))
    print(f"max_abs_diff(logits)={max_abs_diff}")

    if not np.array_equal(pt_lens_np, cm_lens):
        print("WARN: encoder_out_lens mismatch")
        print(f"pt_lens={pt_lens_np} (dtype={pt_lens_np.dtype})")
        print(f"cm_lens={cm_lens} (dtype={cm_lens.dtype}) raw={out['encoder_out_lens']}")


if __name__ == "__main__":
    main()
