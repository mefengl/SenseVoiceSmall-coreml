#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "torch==2.3.*",
#   "funasr>=1.1.3",
#   "modelscope",
#   "huggingface_hub",
#   "numpy<=1.26.4",
#   "coremltools",
# ]
# ///

"""Convert SenseVoiceSmall (PyTorch) -> CoreML .mlpackage.

Vendored into SenseVoiceSmall-coreml so we don't rely on custom tools inside upstream.
We *do* import the upstream SenseVoice model code via --sensevoice-repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sensevoice-repo",
        default=os.environ.get("SENSEVOICE_REPO", "./.upstream/SenseVoice"),
        help="Path to upstream SenseVoice repo (for importing model.py)",
    )
    p.add_argument("--model", default=os.environ.get("SENSEVOICE_MODEL", "iic/SenseVoiceSmall"))
    p.add_argument("--device", default=os.environ.get("SENSEVOICE_DEVICE", "cpu"))
    p.add_argument("--out", default=os.environ.get("SENSEVOICE_MLPACKAGE_OUT", "./.coreml-build/SenseVoiceSmall.mlpackage"))
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--default-frames", type=int, default=300)
    p.add_argument(
        "--deployment-target",
        default=os.environ.get("SENSEVOICE_DEPLOYMENT_TARGET", "macOS15"),
        help="Minimum deployment target (e.g. macOS15, iOS15)",
    )
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
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
    T = int(args.default_frames)
    F = 560
    speech = torch.randn(B, T, F, dtype=torch.float32)
    speech_lengths = torch.tensor([T], dtype=torch.int32)
    language = torch.tensor([0], dtype=torch.int32)
    textnorm = torch.tensor([15], dtype=torch.int32)

    # Temporary workaround: upstream `export_meta.export_forward` mutates `speech_lengths` in-place,
    # which can break TorchScript trace-check by reusing a mutated input tensor.
    # Can be removed once FunAudioLLM/SenseVoice PR #275 is merged:
    #   https://github.com/FunAudioLLM/SenseVoice/pull/275
    class _TraceWrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, speech, speech_lengths, language, textnorm):
            return self.m(speech, speech_lengths.clone(), language, textnorm)

    with torch.no_grad():
        traced = torch.jit.trace(
            _TraceWrapper(rebuilt), (speech, speech_lengths, language, textnorm), strict=False
        )

    target = getattr(ct.target, args.deployment_target, None)
    if target is None:
        avail = sorted([n for n in dir(ct.target) if n.startswith(("macOS", "iOS"))])
        raise SystemExit(
            f"Unknown deployment target: {args.deployment_target}. Available: {', '.join(avail)}"
        )

    compute_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32

    speech_shape = ct.Shape(shape=(B, ct.RangeDim(lower_bound=1, upper_bound=int(args.max_frames), default=T), F))

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=compute_precision,
        inputs=[
            ct.TensorType(name="speech", shape=speech_shape, dtype=np.float32),
            ct.TensorType(name="speech_lengths", shape=(B,), dtype=np.int32),
            ct.TensorType(name="language", shape=(B,), dtype=np.int32),
            ct.TensorType(name="textnorm", shape=(B,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="ctc_logits"),
            ct.TensorType(name="encoder_out_lens"),
        ],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        # mlpackage is a directory
        import shutil

        shutil.rmtree(out)
    mlmodel.save(str(out))
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
