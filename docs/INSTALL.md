# Pinned installation (Linux ARM64 / GB10)

Prerequisites: Docker with NVIDIA Container Toolkit and a working NVIDIA driver. Installation does not replace host CUDA/Python. The runtime image is third-party software, not built or redistributed here. Review its upstream project and licenses before use.

```sh
export IMAGE=ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x@sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931
mkdir state
```

Use a fresh `state` directory. Existing state/checkpoints are never removed automatically. Image branding says v0.26.0-gb10.2, but the **actual installed vLLM version is 0.26.1.dev0+gf2654939e.d20260726**, PyTorch2.11.0+cu130. The speed adapter deliberately requires that exact version and checks patch sites. Do not replace the image with `vllm:latest`.

## Prepare the package

Run from the repository root. Package build metadata is written into this checkout; model files are not changed. Container names must be unused; use another name for another attempt.

```sh
docker run --name gemma-kit-setup --user "$(id -u):$(id -g)" \
  -e HOME=/state -v "$PWD":/app -v "$PWD/state":/state -w /app \
  --entrypoint bash "$IMAGE" -lc '
    python3 -m pip install --no-deps --no-build-isolation --target /state/package /app
  '
```


## Obtain weights separately

Read [model licenses](LICENSES.md), then explicitly download the NVFP4 checkpoint. Requires Internet during download; runtime commands can be offline. The following command uses the image's Hugging Face client and does not silently accept gated-model agreements:

```sh
docker run --name gemma-kit-download --user "$(id -u):$(id -g)" \
  -e HOME=/state -e PYTHONPATH=/state/package \
  -v "$PWD":/app:ro -v "$PWD/state":/state -w /app \
  --entrypoint python3 "$IMAGE" scripts/download-model.py speed /state/model-nvfp4
```

The script pins model revisions from `gemma_decision.profiles` and refuses an existing target. Do not point a profile at a different checkpoint and assume equivalent predictions. Keep enough disk space for the model and caches; downloads are not included in the package. Supply credentials through your own local Hugging Face setup if upstream access requires them; never put tokens into this repository.

## Predict (offline)

```sh
docker run --name gemma-kit-predict --gpus all --network none \
  --user "$(id -u):$(id -g)" --shm-size 4g \
  -e HOME=/state -e PYTHONPATH=/state/package \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -e TORCH_EXTENSIONS_DIR=/state/extensions -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e MAX_JOBS=4 -e OMP_NUM_THREADS=4 -e TOKENIZERS_PARALLELISM=true -e RAYON_NUM_THREADS=4 \
  -v "$PWD":/app:ro -v "$PWD/state":/state -w /app \
  --entrypoint python3 "$IMAGE" -m gemma_decision.cli predict \
  --profile speed --model-path /state/model-nvfp4 --input examples/request.json
```

Add `--media` to predict images/videos, and choose `examples/image-request.json` or `examples/video-request.json`. Only one NVFP4 model loads. Startup and first-use kernel compilation are excluded from warm speed claims. Preserve your existing model files when upgrading the package; EXL3 installation is no longer needed.

## Serve

Use the same runtime arguments, change `predict` to `serve --profile speed --model-path /state/model-nvfp4 --port 8765`, and omit `--input`. On Linux use `--network host` instead of `--network none`; the application **still binds only 127.0.0.1**. Docker `-p` alone cannot reach a loopback-only service inside an ordinary bridge network. Use an SSH tunnel for authenticated remote access; no unauthenticated LAN listener is provided.

`GET /health`; `POST /v1/decisions` with `Content-Type: application/json`. Stop this owned container normally with `docker stop gemma-kit-serve` (give it a sufficient timeout for an active request); do not kill other workloads. Start with `--media` to accept images/videos. Restart to change media mode. There is no automatic simultaneous model residency.
