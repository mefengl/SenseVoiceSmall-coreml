#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'TXT'
Usage:
  build_coreml_zip.sh --mlpackage /path/to/SenseVoiceSmall.mlpackage --out coreml/SenseVoiceSmall.mlmodelc.zip

Notes:
- Uses /usr/bin/xcrun coremlcompiler to compile .mlpackage -> .mlmodelc
- Produces a zip that contains the .mlmodelc bundle (keeps parent dir)
TXT
}

MLPACKAGE=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mlpackage) MLPACKAGE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$MLPACKAGE" || -z "$OUT" ]]; then
  usage; exit 2
fi

# Resolve OUT to absolute path to handle cd later
if [[ "$OUT" != /* ]]; then
  OUT="$(pwd)/$OUT"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d "$MLPACKAGE" ]]; then
  echo "ERROR: mlpackage not found: $MLPACKAGE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

TMP="$ROOT/.coreml-tmp"
rm -rf "$TMP" && mkdir -p "$TMP/compiled"

echo "Compiling .mlpackage -> .mlmodelc ..."
/usr/bin/xcrun coremlcompiler compile "$MLPACKAGE" "$TMP/compiled" >/dev/null

MLMODELC=$(find "$TMP/compiled" -maxdepth 2 -type d -name "*.mlmodelc" | head -n 1 || true)
if [[ -z "$MLMODELC" ]]; then
  echo "ERROR: coremlcompiler did not produce a .mlmodelc" >&2
  exit 1
fi

rm -f "$OUT"
echo "Zipping: $OUT (deterministic)"
# Use standard zip with -X (no extra file attributes/timestamps) and -r (recursive)
# We also touch the files to a fixed timestamp before zipping to ensure content mtime consistency if zip leaks it
find "$MLMODELC" -exec touch -t 198001010000 {} +
(cd "$(dirname "$MLMODELC")" && /usr/bin/zip -q -X -r "$OUT" "$(basename "$MLMODELC")")

echo "Updating checksums.sha256 + manifest.json"
python3 - "$OUT" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

root = Path.cwd()
out_arg = Path(sys.argv[1])
out = out_arg if out_arg.is_absolute() else (root / out_arg)
out = out.resolve()
manifest = root / "manifest.json"
checksums = root / "checksums.sha256"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

zip_sha = sha256(out)
zip_size = out.stat().st_size

try:
    xcode = subprocess.check_output(["/usr/bin/xcodebuild", "-version"], text=True).strip()
except Exception:
    xcode = None

rel = out.relative_to(root)

m = json.loads(manifest.read_text())
m.setdefault("build", {})
m["build"].update({
    "artifact": rel.as_posix(),
    "sha256": zip_sha,
    "bytes": zip_size,
    "xcodebuild_version": xcode,
})
manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")

checksums.write_text(f"{zip_sha}  {rel.as_posix()}\n")

print("sha256", zip_sha)
print("bytes", zip_size)
print("path", rel.as_posix())
PY

echo "Done."
