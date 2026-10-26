"""Optional vLLM multimodal-encoder tracing for controlled diagnostics.

This module is loaded only when its directory is explicitly added to
PYTHONPATH. It has no effect unless PESTATE_ENCODER_TRACE names an output file.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


TRACE_PATH = os.environ.get("PESTATE_ENCODER_TRACE", "").strip()
CACHE_TOKENS = int(os.environ.get("PESTATE_ENCODER_CACHE_TOKENS", "0") or 0)
RESOURCE_TRACE = os.environ.get("PESTATE_RESOURCE_TRACE", "0").strip() == "1"


def _append_trace(payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    fd = os.open(TRACE_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)


def _tensor_payload(value: Any) -> dict[str, Any]:
    """Count logical tensor bytes and unique backing-storage bytes.

    Logical bytes are the representation elements visible through the cached
    tensor views. Unique-storage bytes count each backing allocation once
    within this event. Both intentionally exclude Python/container and CUDA
    allocator overhead.
    """

    try:
        import torch
    except ImportError:
        return {
            "logical_tensor_bytes": 0,
            "unique_storage_bytes": 0,
            "tensor_count": 0,
            "dtype_counts": {},
        }

    tensors: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child)

    visit(value)
    logical_bytes = sum(int(item.numel()) * int(item.element_size()) for item in tensors)
    storages: dict[tuple[str, int], int] = {}
    for item in tensors:
        storage = item.untyped_storage()
        key = (str(item.device), int(storage.data_ptr()))
        storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
    dtype_counts = Counter(str(item.dtype) for item in tensors)
    return {
        "logical_tensor_bytes": logical_bytes,
        "unique_storage_bytes": sum(storages.values()),
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }


def _request_encoder_payloads(
    runner: Any, request_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Snapshot the complete encoder-output payload for each request.

    ``scheduled_encoder_inputs`` contains only cache misses.  Measuring those
    outputs therefore makes a video's reported size depend on which videos
    happened to run before it.  A request's reusable visual state is instead
    the set of cached outputs referenced by *all* of its multimodal features.

    Identical feature identifiers are retained once, matching vLLM's
    content-addressed encoder cache.  ``_tensor_payload`` additionally counts
    a shared tensor backing storage once even if distinct output views refer
    to it.  During chunked prefill an early snapshot can be incomplete; the
    summarizer accepts only the final complete snapshot.
    """

    snapshots: list[dict[str, Any]] = []
    for request_id in request_ids:
        req_state = runner.requests[str(request_id)]
        identifiers = [feature.identifier for feature in req_state.mm_features]
        unique_identifiers = list(dict.fromkeys(identifiers))
        available_identifiers = [
            identifier
            for identifier in unique_identifiers
            if identifier in runner.encoder_cache
        ]
        payload = _tensor_payload([
            runner.encoder_cache[identifier]
            for identifier in available_identifiers
        ])
        snapshots.append({
            "request_id": str(request_id),
            "request_mm_feature_count": len(identifiers),
            "request_identifier_count": len(identifiers),
            "request_unique_identifier_count": len(unique_identifiers),
            "request_unique_encoder_output_count": len(available_identifiers),
            "request_payload_complete": (
                len(available_identifiers) == len(unique_identifiers)
            ),
            **payload,
        })
    return snapshots


def _aggregate_request_payloads(
    snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Provide aggregate compatibility fields for a trace event.

    Per-request snapshots remain authoritative.  The aggregate sums each
    request independently, so storage shared by two concurrently batched
    requests is intentionally charged to both requests rather than making
    either request's size depend on batching.
    """

    dtype_counts: Counter[str] = Counter()
    for snapshot in snapshots:
        dtype_counts.update(snapshot.get("dtype_counts") or {})
    return {
        "request_mm_feature_count": sum(
            int(snapshot.get("request_mm_feature_count") or 0)
            for snapshot in snapshots
        ),
        "request_identifier_count": sum(
            int(snapshot.get("request_identifier_count") or 0)
            for snapshot in snapshots
        ),
        "request_unique_identifier_count": sum(
            int(snapshot.get("request_unique_identifier_count") or 0)
            for snapshot in snapshots
        ),
        "request_unique_encoder_output_count": sum(
            int(snapshot.get("request_unique_encoder_output_count") or 0)
            for snapshot in snapshots
        ),
        "request_payload_complete": all(
            bool(snapshot.get("request_payload_complete"))
            for snapshot in snapshots
        ),
        "logical_tensor_bytes": sum(
            int(snapshot.get("logical_tensor_bytes") or 0)
            for snapshot in snapshots
        ),
        "unique_storage_bytes": sum(
            int(snapshot.get("unique_storage_bytes") or 0)
            for snapshot in snapshots
        ),
        "tensor_count": sum(
            int(snapshot.get("tensor_count") or 0)
            for snapshot in snapshots
        ),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }

if TRACE_PATH or CACHE_TOKENS:
    try:
        if CACHE_TOKENS:
            from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

            _original_encoder_cache_init = EncoderCacheManager.__init__

            @functools.wraps(_original_encoder_cache_init)
            def _sized_encoder_cache_init(self, cache_size):
                return _original_encoder_cache_init(
                    self, max(int(cache_size), CACHE_TOKENS)
                )

            EncoderCacheManager.__init__ = _sized_encoder_cache_init
            print(
                f"[pestate-encoder-cache] minimum cache size={CACHE_TOKENS} tokens",
                file=sys.stderr,
            )

        if not TRACE_PATH:
            raise RuntimeError("encoder tracing path is required for cache acceptance")

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        _original_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

        @functools.wraps(_original_execute_mm_encoder)
        def _traced_execute_mm_encoder(self, scheduler_output):
            scheduled = scheduler_output.scheduled_encoder_inputs or {}
            request_count = len(scheduled)
            item_count = sum(len(item_ids) for item_ids in scheduled.values())
            request_ids = sorted(str(request_id) for request_id in scheduled)
            cuda_started = None
            cuda_finished = None
            if RESOURCE_TRACE and item_count:
                import torch

                cuda_started = torch.cuda.Event(enable_timing=True)
                cuda_finished = torch.cuda.Event(enable_timing=True)
                cuda_started.record()
            started = time.perf_counter()
            try:
                return _original_execute_mm_encoder(self, scheduler_output)
            finally:
                if item_count:
                    cuda_elapsed_s = None
                    tensor_payload = {
                        "request_mm_feature_count": None,
                        "request_identifier_count": None,
                        "request_unique_identifier_count": None,
                        "request_unique_encoder_output_count": None,
                        "request_payload_complete": None,
                        "logical_tensor_bytes": None,
                        "unique_storage_bytes": None,
                        "tensor_count": None,
                        "dtype_counts": {},
                    }
                    request_payloads: list[dict[str, Any]] = []
                    if RESOURCE_TRACE:
                        assert cuda_started is not None and cuda_finished is not None
                        cuda_finished.record()
                        cuda_finished.synchronize()
                        cuda_elapsed_s = float(
                            cuda_started.elapsed_time(cuda_finished)
                        ) / 1000.0
                        request_payloads = _request_encoder_payloads(
                            self, request_ids
                        )
                        tensor_payload = _aggregate_request_payloads(
                            request_payloads
                        )
                    payload = {
                        "kind": "mm_encoder_execution",
                        "time_ns": time.time_ns(),
                        "pid": os.getpid(),
                        "request_ids": request_ids,
                        "request_count": request_count,
                        "item_count": item_count,
                        "elapsed_s": time.perf_counter() - started,
                        "cuda_elapsed_s": cuda_elapsed_s,
                        "resource_trace": RESOURCE_TRACE,
                        "scheduled_encoder_item_count": item_count,
                        "request_payloads": request_payloads,
                        **tensor_payload,
                    }
                    _append_trace(payload)

        GPUModelRunner._execute_mm_encoder = _traced_execute_mm_encoder
        print("[pestate-encoder-trace] GPUModelRunner instrumentation installed", file=sys.stderr)

        if RESOURCE_TRACE:
            import torch

            _original_execute_model = GPUModelRunner.execute_model

            @functools.wraps(_original_execute_model)
            def _traced_execute_model(self, scheduler_output, *args, **kwargs):
                request_ids = sorted(
                    str(request_id)
                    for request_id in (scheduler_output.num_scheduled_tokens or {})
                )
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                result = _original_execute_model(
                    self, scheduler_output, *args, **kwargs
                )
                finished.record()
                finished.synchronize()
                _append_trace({
                    "kind": "model_runner_execution",
                    "time_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "request_ids": request_ids,
                    "scheduled_tokens": {
                        str(key): int(value)
                        for key, value in (
                            scheduler_output.num_scheduled_tokens or {}
                        ).items()
                    },
                    "encoder_item_count": sum(
                        len(item_ids)
                        for item_ids in (
                            scheduler_output.scheduled_encoder_inputs or {}
                        ).values()
                    ),
                    "cuda_elapsed_s": float(started.elapsed_time(finished)) / 1000.0,
                })
                return result

            GPUModelRunner.execute_model = _traced_execute_model

            from vllm.v1.core.kv_cache_manager import KVCacheManager

            _original_get_computed_blocks = KVCacheManager.get_computed_blocks

            @functools.wraps(_original_get_computed_blocks)
            def _traced_get_computed_blocks(self, request):
                computed_blocks, cached_tokens = _original_get_computed_blocks(
                    self, request
                )
                if cached_tokens:
                    group_rows = []
                    payload_bytes = 0
                    for index, (blocks, group) in enumerate(zip(
                        computed_blocks.blocks,
                        self.kv_cache_config.kv_cache_groups,
                    )):
                        resident_blocks = [
                            block for block in blocks
                            if not bool(getattr(block, "is_null", False))
                        ]
                        page_bytes = int(group.kv_cache_spec.page_size_bytes)
                        layer_count = len(group.layer_names)
                        # UniformTypeKVCacheSpecs.page_size_bytes already sums
                        # all member layers; ordinary AttentionSpec page size
                        # describes one layer and therefore needs the layer
                        # multiplier.
                        page_includes_all_layers = hasattr(
                            group.kv_cache_spec, "kv_cache_specs"
                        )
                        layer_multiplier = 1 if page_includes_all_layers else layer_count
                        group_bytes = (
                            len(resident_blocks) * page_bytes * layer_multiplier
                        )
                        payload_bytes += group_bytes
                        group_rows.append({
                            "group_index": index,
                            "block_table_entries": len(blocks),
                            "resident_block_count": len(resident_blocks),
                            "null_block_count": len(blocks) - len(resident_blocks),
                            "block_size_tokens": int(
                                group.kv_cache_spec.block_size
                            ),
                            "page_size_bytes": page_bytes,
                            "layer_count": layer_count,
                            "page_includes_all_layers": page_includes_all_layers,
                            "layer_multiplier": layer_multiplier,
                            "payload_bytes": group_bytes,
                            "spec_type": type(group.kv_cache_spec).__name__,
                        })
                    _append_trace({
                        "kind": "kv_prefix_hit",
                        "time_ns": time.time_ns(),
                        "pid": os.getpid(),
                        "request_id": str(request.request_id),
                        "cached_tokens": int(cached_tokens),
                        "logical_payload_bytes": payload_bytes,
                        "groups": group_rows,
                        "measurement_scope": (
                            "KV tensor pages reached by this prefix hit; excludes "
                            "the server-wide preallocated free pool, allocator "
                            "fragmentation, and block-hash metadata."
                        ),
                    })
                return computed_blocks, cached_tokens

            KVCacheManager.get_computed_blocks = _traced_get_computed_blocks
            print(
                "[pestate-resource-trace] synchronized CUDA and KV payload tracing installed",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[pestate-encoder-trace] installation failed: {exc!r}", file=sys.stderr)
        raise
