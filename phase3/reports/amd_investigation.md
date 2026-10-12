# AMD / ROCm Hardware Investigation — Phase 3

**Date:** 2026-10-12
**Scope:** Verify, with no assumptions either way, whether this development machine has real AMD GPU / ROCm / AMD Developer Cloud access available for the LLM verification layer.

## Finding: No AMD hardware, driver stack, or cloud access present

This is an **Intel-only machine**. There is no AMD hardware anywhere in the stack, and no AMD cloud credentials of any kind were found.

### 1. GPU / CPU hardware (via Windows WMI)

```
GPU:  Intel(R) Iris(R) Xe Graphics   (AdapterCompatibility: Intel Corporation)
CPU:  12th Gen Intel(R) Core(TM) i5-1235U   (Manufacturer: GenuineIntel)
```

No AMD Radeon or AMD Instinct device is present. `rocm-smi` is **not installed** (`command not found`).

### 2. Environment variables

Checked for `HIP_VISIBLE_DEVICES`, `ROCM_HOME`, `ROCM_PATH`, and any variable matching `^(HIP_|ROCM_|AMD_)` (both in the Bash shell env and via `Get-ChildItem Env:` in PowerShell, matched by **name only**, no values printed). Result: **none set**. No CUDA-related variables either.

### 3. Torch build

```
torch.__version__        = 2.12.0+cpu
torch.cuda.is_available() = False
torch.version.hip         = None
```

`torch.__config__.show()` confirms the build uses Intel MKL / MKL-DNN + AVX2, with no CUDA or ROCm backend compiled in. This matches and reconfirms the earlier-phase finding: the existing torch install in this repo's environment is **CPU-only**, with **no ROCm build either** — not just "no CUDA."

### 4. vLLM

```
pip show vllm  →  WARNING: Package(s) not found: vllm
```

vLLM is **not installed** at all, so there is no ROCm backend to check within it.

### 5. AMD Developer Cloud credentials / config

Searched the full repository tree for filenames/content matching `AMD_API`, `AMD_DEVELOPER`, `ROCM_API`, `amd.cloud`, `developer.amd.com`, `AMD_CLOUD` across `.py/.env/.json/.yaml/.yml/.md/.txt` files, and searched for any `.env*` file anywhere in the repo. Result: **nothing found**. No AMD cloud endpoint URLs, API keys, or config files exist anywhere in this project.

### 6. What IS actually available for LLM inference here

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`: **not set** in the environment (only `CLAUDE_CODE_EXECPATH` is present, which is the Claude Code CLI's own harness path, not a model API key).
- **Ollama is reachable** at `http://localhost:11434` and responded successfully to `/api/tags` and `/api/generate`. Models already pulled and available:
  - `qwen2.5:1.5b` (1.5B params, Q4_K_M, 32K ctx)
  - `qwen3:8b` (8.2B params, Q4_K_M, 40K ctx)
  - `nomic-embed-text:latest` (embedding model)

  A live test call (`qwen2.5:1.5b`, prompt "Say OK") completed in ~7.6s (first call includes ~6.6s model load; subsequent calls are fast).
- `transformers 4.57.6` + `torch 2.12.0+cpu` are installed and importable, confirming the earlier-phase finding still holds. Outbound network to `huggingface.co` returned HTTP 200 (reachable), but no Qwen/small-instruct model is currently present in the local HF cache (`~/.cache/huggingface`) — a `transformers`-based load would require a fresh download.

This means the honest provider decision for the smoke test in this investigation is: no AMD path exists, but a real (non-AMD) local small-model inference path (Ollama) does exist and was used — see `phase3/llm_provider.py` and the smoke-test report for exactly which provider ran.

## Exact action a human would need to take for real AMD execution

None of the following exist today in this sandbox, and all of them would be required:

1. **Hardware/cloud access**: Provision either (a) an **AMD Developer Cloud** instance (or any cloud offering AMD Instinct MI-series GPUs, e.g. MI210/MI250/MI300), and obtain its endpoint URL and an API key or SSH credential, or (b) physical/on-prem access to a machine with an **AMD Instinct or Radeon GPU** and the **ROCm 6.x** driver stack installed.
2. **Credentials as environment variables**: if using a cloud endpoint, the human must supply something like `AMD_CLOUD_ENDPOINT`, `AMD_CLOUD_API_KEY` (or equivalent SSH access) — none of these currently exist anywhere in this environment or repo.
3. **ROCm-enabled software stack** on that target machine/instance:
   - `rocm-smi` installed and showing at least one AMD device.
   - A ROCm build of PyTorch (`pip install torch --index-url <rocm-wheel-index>`) such that `torch.version.hip` returns a real version string (not `None`) and `torch.cuda.is_available()` returns `True` (ROCm exposes itself through the CUDA-compatible HIP API).
   - `vllm` installed with its ROCm backend (`VLLM_TARGET_DEVICE=rocm` build, or the official ROCm vLLM Docker image).
4. Re-run the exact same verification scripts (`phase3/llm_provider.py`, `phase3/verification_llm.py`) pointed at the production model (see `phase3/reports/llm_model_selection.md`) served via vLLM on that ROCm instance.

**Until all four of the above are provided, no AMD execution is possible in this environment, and none was performed or claimed in this work.** The smoke test in Part 3 below runs on a genuinely available local alternative (Ollama, CPU) instead, and is reported as such.
