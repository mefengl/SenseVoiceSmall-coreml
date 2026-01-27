---
library_name: coreml
pipeline_tag: automatic-speech-recognition
tags:
- coreml
- speech-recognition
- sensevoice
- asr
license: other
license_name: funasr-model-license
license_link: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
language:
- zh
- en
- ja
- ko
- yue
---

# SenseVoiceSmall-coreml

[Hugging Face](https://huggingface.co/mefengl/SenseVoiceSmall-coreml) | [GitHub](https://github.com/mefengl/SenseVoiceSmall-coreml)

CoreML artifact for **SenseVoiceSmall**.

- Upstream model: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- Upstream code: https://github.com/FunAudioLLM/SenseVoice

## Download

```bash
uvx hf download mefengl/SenseVoiceSmall-coreml --local-dir . \
  --include "coreml/SenseVoiceSmall.mlmodelc.zip" \
  --include "config.json" \
  --include "checksums.sha256"
```

## Verify

```bash
make verify
```

## Optional: generate CoreML `.mlpackage`

A CoreML conversion of SenseVoiceSmall, using the upstream model definition.

```bash
# clones upstream code into ./.upstream/SenseVoice automatically
make convert DEPLOYMENT_TARGET=macOS15
# output: ./.coreml-build/SenseVoiceSmall.mlpackage
```

## Optional: rebuild `coreml/SenseVoiceSmall.mlmodelc.zip` from a `.mlpackage`

macOS + Xcode required.

```bash
make build
make inspect
make verify
```

## Optional: compare (numeric)

Compare the CoreML output against the original PyTorch model.

```bash
make compare \
  MLPACKAGE=.coreml-build/SenseVoiceSmall.mlpackage
# SENSEVOICE_REPO defaults to ./.upstream/SenseVoice (auto-cloned by `make upstream`)
```
