# Agent notes for FLA_KDA

## What this repo actually is

A partial fork of vLLM whose only purpose is porting the **KDA (Kimi Delta
Attention) FLA Triton kernels to Ascend NPU** via `triton-ascend`. The full
`vllm/` and `tests/` trees from upstream are checked in, but only a tiny slice
is in active scope. Treat the rest as read-only context.

- Origin: `https://github.com/XyHalcyon/FLA_KDA.git`, single commit, forked from
  `ChenxiQ/vllm` branch `migrate_kda_to_npu` (upstream `d8779c5`).
- No `README`, no `pyproject.toml`, no `setup.py`, no `requirements*.txt`, no
  CI config at root. There is no build here — vLLM must already be installed
  in the environment. Do not try to `pip install -e .` this tree.
- Many paths in `.gitignore` are *generated at build time* by upstream vLLM
  (e.g. `vllm/_version.py`, `vllm/vllm_flash_attn/*`,
  `vllm/third_party/triton_kernels/*`, `vllm/third_party/deep_gemm/`,
  `vllm/grpc/vllm_engine_pb2*`). Do not re-generate these here.

## Files in active scope

- `vllm/model_executor/layers/fla/ops/kda.py` — the kernels under test
  (`chunk_kda`, `fused_recurrent_kda`, `fused_kda_gate`, `FusedRMSNormGated`).
  Has NPU-specific scheduling rewrites:
  - `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` uses a `(NT, B*H)` grid
    and contiguous T-major tiles instead of upstream's `(NT, NC, B*H)` grid with
    transposed strides.
  - `chunk_gla_fwd_kernel_o` replaces in-kernel `tl.arange` mask computation with
    a pre-computed `torch.tril` buffer passed as a kernel argument, reducing AIV
    scalar占比 and achieving ~11.6x speedup on Ascend.
- `vllm/model_executor/layers/fla/ops/cumsum.py` — `chunk_local_cumsum` modified
  to use `tl.cumsum` instead of `tl.dot` with a lower-triangular mask, reducing
  scalar operations from ~97% to minimal on Ascend.
- `vllm/model_executor/layers/kda.py` — `KimiDeltaAttention` layer that wires
  the kernels into vLLM's prefill (`chunk_kda`) and decode
  (`fused_recurrent_kda`) paths.
- `tests/kernels/test_chunk_kda_npu.py` — prefill kernel precision test.
  **Currently only 1 parametrization is active** (`H32-D128-cu[0,8192]-float16`);
  the other 8 are commented out (commit `2f0251e`). The report lists 9 but the
  file only runs the 8K case. Uncomment before adding new cases.
- `tests/kernels/test_fused_recurrent_kda_npu.py` — decode kernel precision test
  (3 functions: non-inplace varlen, inplace + `ssm_state_indices`, fp32).
  All parametrizations are active (10 + 5 + 4 = 19 cases).
- `tests/kernels/test_kda_npu_report.md` — coverage-vs-vLLM-call-site matrix.
  Read this before changing test parametrizations or adding scenarios; it is
  the spec for what these tests must cover.

Anything else under `vllm/` or `tests/` is upstream vLLM and not the target of
work in this repo.

## Running the KDA tests

These tests **only run on Ascend NPU**. They `import torch_npu` and use
`DEVICE = "npu"`; they will hard-fail on CUDA/CPU hosts. Triton calls go
through `triton-ascend`.

```
pytest tests/kernels/test_chunk_kda_npu.py -v
pytest tests/kernels/test_fused_recurrent_kda_npu.py -v
```

Pass criteria is **RMSE relative error < 0.005** vs a float32 naive recurrent
reference, with absolute-error short-circuit at `atol=1e-6`. Both `o` and `ht`
are checked. Any NaN in kernel output fails the test.

Both files carry `@pytest.mark.skip_global_cleanup`, which is honored by the
top-level `tests/conftest.py` to skip the post-test distributed cleanup
(see `should_do_global_cleanup_after_test` at `tests/conftest.py:255`). Do
not remove that marker — it is required to avoid initializing the full vLLM
distributed env for kernel-only tests.

`tests/conftest.py` also gates `@pytest.mark.optional` behind a `--optional`
flag (`tests/conftest.py:1454`); KDA tests do not use it, but other tests
under `tests/` will silently skip without the flag.

## Non-obvious correctness gotchas

- **`chunk_kda` initial-state layout differs from the naive reference.** The
  kernel expects `initial_state` shaped as `(N, H, V, K)` — i.e. the K/V dims
  are transposed relative to the naive `(N, H, K, V)` form. Tests pass
  `h0.transpose(-1, -2).contiguous()` in and transpose `tri_ht` back before
  comparison. See `tests/kernels/test_chunk_kda_npu.py:140-152`. This applies
  to `fused_recurrent_kda`'s state buffers too (see
  `test_fused_recurrent_kda_decode_inplace`).
- **`cu_seqlens` (varlen) mode requires `B == 1`.** Inputs must already be
  flattened. `fused_recurrent_kda` enforces this with a `ValueError`
  (`vllm/model_executor/layers/fla/ops/kda.py:123`). All KDA tests use `B = 1`.
- **In vLLM's decode path, `inplace_final_state=True` is mandatory** and the
  kernel writes back into `recurrent_state` via `ssm_state_indices`. Slot
  index `0` is the NULL slot and must remain untouched — `Test 2` asserts
  this. New tests touching the decode path must preserve that contract.
- **`use_qk_l2norm_in_kernel=True` is the only mode vLLM uses.** Tests pre-apply
  `l2norm_fwd` to Q/K for the reference but pass raw Q/K to the kernel.
- `FLA_CHUNK_SIZE = 64` (`vllm/model_executor/layers/fla/ops/utils.py:31`) and
  the autotune warp grid in `kda.py:29` branches on `is_amd`; there is no
  dedicated NPU branch — correctness on Ascend is achieved through
  `triton-ascend` lowering, not source forks.

## Coverage gap to remember

Per `test_kda_npu_report.md`, `fused_kda_gate` and `FusedRMSNormGated` are
called from `KimiDeltaAttention.forward` but have **no precision tests**. If
asked to extend KDA test coverage, that gate test is the highest-priority
addition, followed by `T=1` prefill and zero-initial-state cases.

## Things not to do

- Do not edit upstream-vLLM files outside the active-scope list above unless
  the task explicitly calls for it; this repo's diff against upstream is
  intentionally minimal.
- Do not add a `README`, `pyproject.toml`, or build scripts speculatively.
  Nothing in the tree expects them.
- Do not run the broader `tests/` suites as a smoke check — most require a
  full vLLM engine, GPUs, model downloads, or network access, and many are
  irrelevant to the NPU/KDA work.
