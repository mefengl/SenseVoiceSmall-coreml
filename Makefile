ZIP := coreml/SenseVoiceSmall.mlmodelc.zip

# Imports upstream model code from this repo.
SENSEVOICE_REPO ?= ./.upstream/SenseVoice

# Default output location for a generated mlpackage.
MLPACKAGE_OUT ?= ./.coreml-build/SenseVoiceSmall.mlpackage

# Default deployment target.
DEPLOYMENT_TARGET ?= macOS15

# Hugging Face repository to upload to.
HF_REPO ?= mefengl/SenseVoiceSmall-coreml

.PHONY: upstream convert build pin inspect verify compare upload

upstream:
	@test -d "$(SENSEVOICE_REPO)/.git" || (mkdir -p "$(dir $(SENSEVOICE_REPO))" && git clone https://github.com/FunAudioLLM/SenseVoice "$(SENSEVOICE_REPO)")

convert: upstream
	uv run scripts/convert_coreml.py --sensevoice-repo "$(SENSEVOICE_REPO)" --out "$(MLPACKAGE_OUT)" --deployment-target "$(DEPLOYMENT_TARGET)"

build:
	@test -d "$(MLPACKAGE_OUT)" || (echo "Missing $(MLPACKAGE_OUT). Run 'make convert' first."; exit 2)
	./scripts/build_coreml_zip.sh --mlpackage "$(MLPACKAGE_OUT)" --out "$(ZIP)"

pin:
	@HF_SHA=$$(uvx hf models info "FunAudioLLM/SenseVoiceSmall" --expand sha | python3 -c 'import sys,json;print(json.load(sys.stdin)["sha"])'); \
	CMVN_URL='https://modelscope.cn/api/v1/models/iic/SenseVoiceSmall/repo?Revision=master&FilePath=am.mvn'; \
	SPM_URL='https://modelscope.cn/api/v1/models/iic/SenseVoiceSmall/repo?Revision=master&FilePath=chn_jpn_yue_eng_ko_spectok.bpe.model'; \
	ARGS="pin --manifest config.json --model FunAudioLLM/SenseVoiceSmall --model-revision $$HF_SHA --asset-url cmvn_am.mvn=$$CMVN_URL --asset-url spm=$$SPM_URL"; \
	if [ -d "$(SENSEVOICE_REPO)" ]; then ARGS="$$ARGS --sensevoice-repo $(SENSEVOICE_REPO)"; fi; \
	uv run scripts/repo.py $$ARGS

inspect:
	./scripts/inspect_zip.sh "$(ZIP)"

verify:
	uv run scripts/repo.py validate --root .
	shasum -a 256 -c checksums.sha256

compare:
	@test -d "$(MLPACKAGE_OUT)" || (echo "Missing $(MLPACKAGE_OUT). Run 'make convert' first."; exit 2)
	@test -d "$(SENSEVOICE_REPO)" || (echo "Missing $(SENSEVOICE_REPO). Set SENSEVOICE_REPO=..."; exit 2)
	uv run scripts/compare_torch_coreml.py --sensevoice-repo "$(SENSEVOICE_REPO)" --mlpackage "$(MLPACKAGE_OUT)"

upload:
	uvx hf upload "$(HF_REPO)" "$(ZIP)" "$(ZIP)"
	uvx hf upload "$(HF_REPO)" config.json config.json
	uvx hf upload "$(HF_REPO)" checksums.sha256 checksums.sha256
	uvx hf upload "$(HF_REPO)" README.md README.md
