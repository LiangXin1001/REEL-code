# vLLM encoder-cache instrumentation

`sitecustomize.py` is the exact runtime patch used by the accepted REEL runs.
Its SHA-256 is
`211d6525ce9bb446fd900c791473dcdb1c82dcdeca586b06a5ea6584e2d9430c`.
It was tested only with the vLLM 0.11.0 and 0.15.0 environments pinned in
`configs/environments/`.

The Visual condition needs this patch because the public vLLM command-line
interface does not expose the required minimum encoder-output cache size. The
patch changes only that capacity and records encoder executions; prefix
caching remains controlled by vLLM's normal command-line flag.

Load it only for a Visual service:

```bash
export PESTATE_ENCODER_TRACE=/absolute/path/to/run/encoder_trace.jsonl
export PESTATE_ENCODER_CACHE_TOKENS=65536
export PYTHONPATH="$PWD/instrumentation/vllm_encoder_trace${PYTHONPATH:+:$PYTHONPATH}"
vllm serve ...
```

The accepted main runs set this minimum to 65,536 tokens for every model and
dataset. This is a lower bound, not an exact capacity: the patch uses the
larger of vLLM's profiled native budget and this value. For example, the
accepted Qwen3-VL HD service reported an effective 153,600-token budget. The
server log must contain both `[pestate-encoder-cache] minimum cache size=...`
and
`[pestate-encoder-trace] GPUModelRunner instrumentation installed`. Merely
setting `PESTATE_ENCODER_CACHE_TOKENS` without putting this directory first on
`PYTHONPATH` has no effect. Record both the requested minimum and vLLM's
reported effective budget.

The optional `PESTATE_RESOURCE_TRACE=1` mode synchronizes CUDA operations and
adds tensor-size and prefix-page measurements. It is for a separate resource
diagnostic, not a timing run. Trace files can contain request identifiers and
must be reviewed before sharing.

This patch imports private vLLM interfaces and intentionally fails at startup
when those interfaces are incompatible. Do not silently use it with a newer
vLLM release; port and revalidate it first.
