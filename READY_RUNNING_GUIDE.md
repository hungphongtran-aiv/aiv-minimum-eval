# Ready-Running Guide: vLLM Serving + AIV/Official MMLU

This guide gives a practical, copy-paste flow to:

1. host a LlamaFactory-exported HuggingFace-style checkpoint with vLLM,
2. run `aiv-benchmark-harness` on `mmlu`, and
3. run `lm-evaluation-harness` on official `mmlu` only.

## 0) Chosen benchmark in `aiv-benchmark-harness`

From the available tasks on the current `deepedu-runner` branch, this guide uses:

- benchmark: `mmlu`
- tasks: `hle_mc`, `vlmu_extend`

Reason: it aligns best with your requirement to run official MMLU in `lm-evaluation-harness`.

## 1) Prerequisites

- Linux/macOS machine with NVIDIA GPU and CUDA compatible with your `vllm` build
- `uv` installed
- A HuggingFace-format checkpoint exported from LlamaFactory

Expected checkpoint files (minimum):

- `config.json`
- model weights (`*.safetensors` or `pytorch_model*.bin`)
- tokenizer files (`tokenizer.json` or `tokenizer.model`, `tokenizer_config.json`, special token files)

If your result is LoRA-only adapters, merge adapters into full weights before serving.

## 2) Set shared environment variables

Open a shell and set these once:

```bash
export MODEL_PATH="/ABSOLUTE/PATH/TO/YOUR/HF_CHECKPOINT"
export SERVED_MODEL_NAME="your-model-name"
export VLLM_PORT="8000"
export VLLM_API_KEY="local-dev-key"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export BASE_URL="$OPENAI_BASE_URL"
```

Notes:

- Use an absolute path for `MODEL_PATH`.
- Keep `SERVED_MODEL_NAME` stable; both harnesses will call this name.
- `OPENAI_BASE_URL` is required by this branch's `config.yaml` env expansion.

## 3) Start vLLM OpenAI-compatible server

Use a dedicated environment for serving (recommended):

```bash
# from any directory
uv venv ~/.venvs/vllm-serve
source ~/.venvs/vllm-serve/bin/activate
uv pip install vllm

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --api-key "$VLLM_API_KEY" \
  --dtype auto
```

Optional flags you may add for memory/perf tuning:

- `--gpu-memory-utilization 0.9`
- `--max-model-len 4096`
- `--tensor-parallel-size <num_gpus>`

Keep this terminal running.

## 4) Smoke-test the vLLM server

In a new terminal:

```bash
export VLLM_PORT="8000"
export SERVED_MODEL_NAME="your-model-name"
export VLLM_API_KEY="local-dev-key"

curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  -H "Authorization: Bearer ${VLLM_API_KEY}"
```

Then test one chat completion (used by `aiv-benchmark-harness`):

```bash
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Trả lời đúng định dạng: Câu trả lời: A\"}],\"temperature\":0}" 
```

## 5) Run `aiv-benchmark-harness` on `mmlu`

In a new terminal:

```bash
cd /Users/phong/Workspace/aiv-eval/aiv-benchmark-harness

uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

export OPENAI_API_KEY="$VLLM_API_KEY"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export BASE_URL="$OPENAI_BASE_URL"
```

Note: on this branch, prefer module entrypoint (`python -m src.main`) for CLI execution.

Quick sanity run:

```bash
uv run python -m src.main run \
  --model openai \
  --model-name "$SERVED_MODEL_NAME" \
  --benchmark mmlu \
  --tasks "hle_mc" \
  --limit 5 \
  --output-dir ./results
```

Full run:

```bash
uv run python -m src.main run \
  --model openai \
  --model-name "$SERVED_MODEL_NAME" \
  --benchmark mmlu \
  --output-dir ./results
```

Run only one task (optional):

```bash
uv run python -m src.main run \
  --model openai \
  --model-name "$SERVED_MODEL_NAME" \
  --benchmark mmlu \
  --tasks "vlmu_extend" \
  --output-dir ./results
```

Results are written under:

- `aiv-benchmark-harness/results/`

## 6) Run official `lm-evaluation-harness` on `mmlu` only

In another terminal:

```bash
cd /Users/phong/Workspace/aiv-eval/lm-evaluation-harness

uv venv .venv
source .venv/bin/activate
uv pip install -e ".[api]"

export OPENAI_API_KEY="$VLLM_API_KEY"
export TOKENIZER_PATH="$MODEL_PATH"
```

Quick sanity run:

```bash
uv run lm-eval run \
  --model local-completions \
  --model_args model=$SERVED_MODEL_NAME,base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,tokenizer=$TOKENIZER_PATH,tokenized_requests=False,num_concurrent=1,max_retries=3 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 1 \
  --limit 10 \
  --output_path ./results/mmlu_smoke
```

Full run:

```bash
uv run lm-eval run \
  --model local-completions \
  --model_args model=$SERVED_MODEL_NAME,base_url=http://127.0.0.1:${VLLM_PORT}/v1/completions,tokenizer=$TOKENIZER_PATH,tokenized_requests=False,num_concurrent=1,max_retries=3 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path ./results/mmlu_full \
  --log_samples
```

Outputs are written under:

- `lm-evaluation-harness/results/`

## 7) What to record for reproducibility

For each run, log:

- checkpoint path and checksum
- `SERVED_MODEL_NAME`
- exact vLLM launch command
- commit SHA of each harness repo
- exact eval commands
- output directory paths

## 8) Common issues and fixes

- **401/403 from vLLM**: `OPENAI_API_KEY` does not match `--api-key`.
- **`deepedu-runner` command not found / import error**: use `uv run python -m src.main ...` on this branch.
- **Tokenizer loading failure in official harness**: set `tokenizer=<local checkpoint path or HF tokenizer id>` in `--model_args`.
- **OOM on serve**: reduce `--max-model-len`, lower `--gpu-memory-utilization`, or increase tensor parallel size.
- **Very slow official run**: increase `num_concurrent`, tune `--batch_size`, and check GPU utilization.
- **Format mismatch in AIV MMLU**: ensure model follows `Câu trả lời: <LETTER>`; keep `temperature=0` behavior.
