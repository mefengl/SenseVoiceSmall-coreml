#!/usr/bin/env python3
"""repo.py

Two things only:
- validate: ensure manifest/checksums/artifact are self-consistent.
- pin: update config.json upstream pins + asset checksums.

Usage:
  uv run scripts/repo.py validate --root .
  uv run scripts/repo.py pin --manifest config.json --model FunAudioLLM/SenseVoiceSmall --model-revision <sha> \
    --asset-url cmvn_am.mvn=<url> --asset-url spm=<url> [--sensevoice-repo ./.upstream/SenseVoice]
"""

# /// script
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sensevoice-coreml"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _git_head(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def validate(root: Path) -> None:
    manifest_path = root / "config.json"
    checksums_path = root / "checksums.sha256"

    if not manifest_path.exists():
        raise SystemExit("Missing config.json")
    if not checksums_path.exists():
        raise SystemExit("Missing checksums.sha256")

    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    zip_rel = (m.get("artifacts", {}) or {}).get("coreml_zip")
    if not zip_rel:
        raise SystemExit("config.json missing artifacts.coreml_zip")

    zip_path = root / zip_rel
    if not zip_path.exists():
        raise SystemExit(f"Missing artifact: {zip_rel}")

    actual_sha = _sha256_file(zip_path)
    actual_bytes = zip_path.stat().st_size

    build = m.get("build", {}) or {}
    if build.get("artifact") and build["artifact"] != zip_rel:
        raise SystemExit("manifest build.artifact mismatch")
    if build.get("sha256") and build["sha256"] != actual_sha:
        raise SystemExit("manifest build.sha256 mismatch")
    if build.get("bytes") and int(build["bytes"]) != actual_bytes:
        raise SystemExit("manifest build.bytes mismatch")

    want = f"{actual_sha}  {zip_rel}"
    lines = [ln.strip() for ln in checksums_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if want not in lines:
        raise SystemExit("checksums.sha256 missing/incorrect artifact line")

    # Tiny schema sanity (helps catch accidental drift).
    dec = m.get("decoding", {}) or {}
    if dec.get("ctc_blank_id") != 0:
        raise SystemExit("manifest decoding.ctc_blank_id must be 0")
    if dec.get("token_offset") != 0:
        raise SystemExit("manifest decoding.token_offset must be 0")

    print("OK")


def pin(manifest: Path, model: str, model_revision: str, asset_urls: list[str], sensevoice_repo: Path | None) -> None:
    m = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}

    up = m.get("upstream", {}) or {}
    up["model"] = model
    up["code"] = up.get("code") or "https://github.com/FunAudioLLM/SenseVoice"
    up["model_revision"] = model_revision

    if sensevoice_repo is not None:
        head = _git_head(sensevoice_repo)
        if head:
            up["code_commit"] = head

    assets = []
    for spec in asset_urls:
        if "=" not in spec:
            raise SystemExit(f"Bad --asset-url {spec!r} (expected name=url)")
        name, url = spec.split("=", 1)
        data = _download(url)
        assets.append({"name": name, "url": url, "sha256": _sha256_bytes(data), "bytes": len(data)})

    up["assets"] = assets
    m["upstream"] = up

    manifest.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print("OK")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate")
    v.add_argument("--root", type=Path, default=Path("."))

    p = sub.add_parser("pin")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--asset-url", action="append", default=[])
    p.add_argument("--sensevoice-repo", type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "validate":
        validate(args.root)
    elif args.cmd == "pin":
        pin(args.manifest, args.model, args.model_revision, args.asset_url, args.sensevoice_repo)


if __name__ == "__main__":
    main(sys.argv[1:])
