# AMD Developer Cloud Deployment Instructions — Phase 4

**Status: not executed.** These are the exact steps a human operator would
run once AMD Developer Cloud credits are approved. Nothing in this
document has been run against real AMD hardware — this development
sandbox is confirmed Intel-only with no ROCm stack and no AMD cloud
credentials anywhere in the repo (`phase3/reports/amd_investigation.md`).
`LLM_PROVIDER=vllm` / `VLLMProvider` (added in `phase3/llm_provider.py`
for Phase 4) is a skeleton built to this plan; it has never connected to a
real vLLM/ROCm endpoint.

This formalizes the deployment path already designed (not executed) in
`phase3/reports/llm_model_selection.md` (model choice: Qwen2.5-7B-Instruct)
and `phase3/reports/amd_investigation.md` (what's missing today) into
runnable commands, and wires it to this Phase 4 FastAPI service
(`backend/main.py`).

## 0. Prerequisites before starting

- Approved AMD Developer Cloud credits / account, OR on-prem access to a
  machine with an AMD Instinct (MI210/MI250/MI300) or Radeon GPU.
- SSH access to that instance.
- This repository checked out on that instance (or reachable over the
  network from it), with `backend/` and `phase3/` present.

## 1. Provision the AMD instance

1. Log in to the AMD Developer Cloud console and launch an instance with
   at least one AMD Instinct GPU (MI210 64GB HBM2e or better is sufficient
   for a 7B model; MI250/MI300 give more headroom).
2. Choose an image/template that already ships ROCm if one is offered
   (fastest path); otherwise use a plain Ubuntu 22.04 image and install
   ROCm manually (step 2).
3. Note the instance's public/private IP or hostname — this becomes part
   of `VLLM_ENDPOINT` in step 5.
4. Open (or tunnel) port `8000` (or whichever port you choose for vLLM's
   server) between this API's host and the AMD instance.

## 2. Install ROCm 6.x (skip if the chosen image already includes it)

Run on the AMD instance:

```bash
# Ubuntu 22.04 example — see AMD's current ROCm install docs for your
# exact OS/version, this is the documented pattern as of ROCm 6.x.
sudo apt-get update
sudo apt-get install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo apt-get install -y ./amdgpu-install_6.2.60200-1_all.deb
sudo amdgpu-install -y --usecase=rocm

# Add your user to the required groups, then re-login (or reboot)
sudo usermod -aG render,video "$LOGNAME"
sudo reboot
```

After reboot, verify the GPU is visible:

```bash
rocm-smi
# Expect at least one AMD Instinct device listed. If this errors or shows
# nothing, stop here — vLLM will not work until this passes.
```

## 3. Install a ROCm build of PyTorch and verify it

```bash
python3 -m venv ~/aman-vllm-env
source ~/aman-vllm-env/bin/activate
pip install --upgrade pip

# Use the ROCm wheel index matching your installed ROCm version
# (6.2 shown — adjust to match `rocm-smi --showdriverversion` output).
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# Expect: a version string for torch.version.hip (NOT None), and
# torch.cuda.is_available() == True. If either is wrong, the ROCm PyTorch
# build did not install correctly — fix this before proceeding.
```

## 4. Install vLLM with its ROCm backend and serve the model

```bash
# Official ROCm vLLM Docker image is the most reliable path; pip install
# of vLLM's ROCm backend also works but is more sensitive to exact
# ROCm/PyTorch version alignment. Docker example:
docker pull rocm/vllm:latest

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 8g \
  -p 8000:8000 \
  rocm/vllm:latest \
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype float16

# Pure pip alternative, inside the venv from step 3:
#   VLLM_TARGET_DEVICE=rocm pip install vllm
#   vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --dtype float16
```

Confirm the OpenAI-compatible endpoint is live from the instance itself:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Say OK"}]}'
```

If VRAM is tight, add `--quantization awq` or `--quantization gptq` with an
AWQ/GPTQ-quantized checkpoint of the same model (see
`phase3/reports/llm_model_selection.md` for why int4/int8 quantization was
flagged as an option if latency/VRAM budget requires it).

## 5. Point this API at the new endpoint

On the machine running this FastAPI backend (`backend/main.py`), set (in
`backend/.env`, copied from `backend/.env.example` — never commit the real
file):

```bash
LLM_PROVIDER=vllm
VLLM_ENDPOINT=http://<amd-instance-ip-or-hostname>:8000
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# Only if the vLLM server was started with an API-key/auth proxy in front of it:
# VLLM_API_KEY=<real key, local .env only, never committed>
```

`phase3.llm_provider.get_default_provider()` reads `LLM_PROVIDER` and
constructs `VLLMProvider(endpoint=VLLM_ENDPOINT, api_key=VLLM_API_KEY,
model_name=VLLM_MODEL)`, which calls `POST {VLLM_ENDPOINT}/v1/chat/completions`.
No code change is needed in `phase3/verification_llm.py` or
`phase3/aman_pipeline.py` — the provider abstraction is the seam this swap
was designed for (`phase3/reports/llm_model_selection.md`, "Interface"
section).

## 6. Restart this API

```bash
# From the repo root, with the new env vars exported/sourced:
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

`GET /health` will report `pipeline_loaded: true` once the SVM/BiLSTM/
MARBERT/novelty-detector warm-up finishes (unrelated to the LLM provider
switch — that warm-up never touches the LLM provider; see
`backend/pipeline_wrapper.py`). The LLM provider itself is only invoked on
requests that the router escalates (`VERIFIED_WITH_EVIDENCE`/`UNCERTAIN`).

## 7. Validate before calling this "production"

Re-run the same validation `phase3/reports/llm_model_selection.md` already
calls for: re-run the smoke test / held-out evaluation
(`phase3/evaluate_deep_verification.py`, 25 held-out examples, see
`phase3/reports/deep_verification_eval_summary.json` for the local-Ollama
baseline numbers) against the real vLLM/ROCm endpoint and confirm
schema-validity and citation-validity rates match or exceed the baseline
observed with `OllamaProvider` locally. Do not consider the AMD path
production-ready until that comparison has actually been run and recorded
— this document only makes the deployment steps runnable, it does not
claim the result.

## What has NOT been done

- No AMD instance has been provisioned.
- No ROCm install, PyTorch-ROCm build, or vLLM-ROCm install has been run.
- `VLLMProvider` (`phase3/llm_provider.py`) has never made a real network
  call to a vLLM server; it has only been unit-tested with
  `VLLM_ENDPOINT` unset/fake, confirming it fails honestly (raises)
  instead of silently returning a fabricated response (see
  `backend/tests/test_api.py::TestLLMProviderEnvSwitch`).
