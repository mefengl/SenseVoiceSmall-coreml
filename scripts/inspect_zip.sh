#!/usr/bin/env bash
set -euo pipefail

ZIP=${1:-}
if [[ -z "$ZIP" ]]; then
  echo "Usage: inspect_zip.sh coreml/SenseVoiceSmall.mlmodelc.zip" >&2
  exit 2
fi

if [[ ! -f "$ZIP" ]]; then
  echo "Missing zip: $ZIP" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

/usr/bin/ditto -x -k "$ZIP" "$TMP"

echo "--- zip contains (top) ---"
find "$TMP" -maxdepth 3 -print | sed "s|^$TMP/||" | head -n 80 | cat

MLMODELC=$(find "$TMP" -type d -name "*.mlmodelc" | head -n 1 || true)
if [[ -z "$MLMODELC" ]]; then
  echo "ERROR: no .mlmodelc found inside zip" >&2
  exit 1
fi

echo "--- .mlmodelc metadata.json (if present) ---"
if [[ -f "$MLMODELC/metadata.json" ]]; then
  sed -n '1,160p' "$MLMODELC/metadata.json" | cat
else
  echo "(no metadata.json at root of bundle; ok for some models)"
fi

echo "OK"
