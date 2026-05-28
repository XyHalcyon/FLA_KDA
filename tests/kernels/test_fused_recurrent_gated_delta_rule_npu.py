# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for `fused_recurrent_gated_delta_rule_fwd_kernel` on NPU,
exercised through the KDA production call chain.

Production wiring in this repo:

    KimiDeltaAttention._forward       (vllm/model_executor/layers/kda.py:440)
        |
        v
    fused_recurrent_kda               (.../fla/ops/kda.py:109)
        |
        v
    fused_recurrent_kda_fwd           (.../fla/ops/kda.py:32)
        |   IS_KDA=True
        |   BV = min(next_pow2(V), 8) = 8
        |   num_warps=1, num_stages=3
        v
    fused_recurrent_gated_delta_rule_fwd_kernel   (.../fla/ops/fused_recurrent.py:27)

The kernel itself is defined exactly once (`fused_recurrent.py:27`); `kda.py`
imports the same `@triton.jit` object and dispatches it with `IS_KDA=True`
and `BV=8`. The sibling launcher `fused_recurrent_gated_delta_rule_fwd`
(`fused_recurrent.py:178`) sets `IS_KDA=False` and `BV=32` and is only used
by `olmo_hybrid` -- not in scope for this repo (see CLAUDE.md). We therefore
hit the kernel only via the KDA launcher, matching what runs in production.

Constexpr branches we cover via this path:
- IS_KDA=True              -> per-K-channel gate `g` of shape (B, T, H, K),
                              `b_h *= exp(b_gk[None, :])` rather than scalar.
- USE_QK_L2NORM_IN_KERNEL=True
- IS_BETA_HEADWISE=False   -> beta is scalar per head, shape (B, T, H).
- IS_VARLEN=True           -> `cu_seqlens` provided, B == 1.
- IS_CONTINUOUS_BATCHING=True / INPLACE_FINAL_STATE=True
                           -> decode pattern with `ssm_state_indices`; slot 0
                              is the NULL slot and must remain untouched.
- IS_SPEC_DECODING=False   -> `num_accepted_tokens` is None
                              (`fused_recurrent_kda` hard-codes None at
                              `kda.py:142`).
- HV == H                  -> KDA has no GVA; `i_h = i_hv // (HV // H) = i_hv`.

Production shapes from runtime profiling (KDA-32 + TP4):
- B = 1 (varlen requires it)
- H = HV = local_num_heads = 8        (32 / 4)
- K = V = head_dim = 128
- T = num_decodes (one token per sequence in the decode path)

Pass criteria: RMSE relative error < 0.005, with absolute-error short-circuit
at atol=1e-6.
"""

import pytest
import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

from vllm.model_executor.layers.fla.ops.kda import fused_recurrent_kda
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

DEVICE = "npu"

NPU_RMSE_RATIO_O = 0.005
NPU_RMSE_RATIO_HT = 0.005


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent reference for the IS_KDA=True branch of the kernel.

    Mirrors `fused_recurrent_gated_delta_rule_fwd_kernel` with `IS_KDA=True`:
    `g` is per-K-channel (shape `[B, T, H, K]`), so the state decay is
    `S *= exp(g[..., :, None])` along the K axis (broadcast across V).
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state

    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        g_i, b_i = g[:, i], beta[:, i]
        # Per-K-channel decay (KDA). Broadcast across V.
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)

    if not output_final_state:
        S = None
    return o.to(dtype), S


def assert_close(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    ratio: float,
    err_atol: float = 1e-6,
):
    """RMSE-based relative error comparison."""
    abs_err = (ref.detach() - tri.detach()).flatten().abs().max().item()
    rmse_diff = (
        (ref.detach() - tri.detach()).flatten().square().mean().sqrt().item()
    )
    rmse_base = ref.detach().flatten().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    print(
        f"{name:>8} | max abs err: {abs_err:.6f}"
        f" | rmse ratio: {rel_err:.6f} | threshold: {ratio}"
    )
    if abs_err <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert rel_err < ratio, (
        f"{name}: max abs err {abs_err:.6f},"
        f" rmse ratio {rel_err:.6f} >= {ratio}"
    )


# ---------------------------------------------------------------------------
# Production decode call chain: varlen + ssm_state_indices + inplace state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("H", "K", "V", "N", "dtype"),
    [
        pytest.param(
            *test,
            id="H{}-K{}-V{}-N{}-{}".format(*test),
        )
        for test in [
            # Production shape (KDA-32 + TP4 -> H=8 per rank), varying batch.
            (8, 128, 128, 1, torch.float16),
            (8, 128, 128, 4, torch.float16),
            (8, 128, 128, 16, torch.float16),
            (8, 128, 128, 32, torch.float16),
            (8, 128, 128, 4, torch.bfloat16),
            (8, 128, 128, 16, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kernel_via_kda_decode_inplace(
    H: int,
    K: int,
    V: int,
    N: int,
    dtype: torch.dtype,
):
    """Decode through `fused_recurrent_kda` with inplace + ssm_state_indices.

    Matches the only call site in `KimiDeltaAttention._forward` that reaches
    `fused_recurrent_gated_delta_rule_fwd_kernel` in production.
    """
    B = 1
    T = N  # one token per sequence
    cu_seqlens = list(range(N + 1))

    torch.manual_seed(42)
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)

    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=dtype, device=DEVICE)
    # KDA gate: per-K-channel (IS_KDA=True), shape (B, T, H, K).
    g = F.logsigmoid(
        torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
    ).to(dtype)
    # Scalar beta per head (IS_BETA_HEADWISE=False).
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()

    # State buffer: slot 0 is NULL, slots 1..N are per-seq state.
    # Kernel layout is (num_slots, HV, V, K).
    max_slots = N + 1
    state_buf = torch.randn(
        max_slots, H, V, K, dtype=torch.float32, device=DEVICE
    )
    state_buf[0] = 0  # NULL slot

    ssm_state_indices = torch.arange(
        1, N + 1, dtype=torch.long, device=DEVICE
    )

    # --- naive reference per sequence (one decode token each) ---
    ref_outputs = []
    ref_states = []
    for i in range(N):
        slot = i + 1
        q_i = l2norm_fwd(q[:, i : i + 1].contiguous())
        k_i = l2norm_fwd(k[:, i : i + 1].contiguous())
        # Kernel state layout (H, V, K) -> naive (H, K, V).
        init_state_i = state_buf[slot].transpose(-1, -2).unsqueeze(0)
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, i : i + 1],
            g[:, i : i + 1],
            beta[:, i : i + 1],
            initial_state=init_state_i,
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)

    # --- Triton kernel via the KDA production launcher ---
    state_buf_tri = state_buf.clone()
    tri_o, _ = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=state_buf_tri,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_t,
        ssm_state_indices=ssm_state_indices,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)

    for i in range(N):
        slot = i + 1
        tri_state = state_buf_tri[slot].transpose(-1, -2).unsqueeze(0)
        assert_close(f"ht_{i}", ref_states[i], tri_state, NPU_RMSE_RATIO_HT)

    assert torch.all(state_buf_tri[0] == 0), (
        "NULL slot (0) must not be modified"
    )


# ---------------------------------------------------------------------------
# Float32 -- isolate algorithmic error from fp16/bf16 dtype precision.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("H", "K", "V", "N"),
    [
        pytest.param(*test, id="H{}-K{}-V{}-N{}".format(*test))
        for test in [
            (8, 128, 128, 1),
            (8, 128, 128, 4),
            (8, 128, 128, 16),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_kernel_via_kda_decode_inplace_fp32(
    H: int,
    K: int,
    V: int,
    N: int,
):
    """Same call chain as above, fp32 throughout."""
    B = 1
    T = N
    cu_seqlens = list(range(N + 1))

    torch.manual_seed(42)
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)

    q = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE)
    g = F.logsigmoid(
        torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
    )
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()

    max_slots = N + 1
    state_buf = torch.randn(
        max_slots, H, V, K, dtype=torch.float32, device=DEVICE
    )
    state_buf[0] = 0

    ssm_state_indices = torch.arange(
        1, N + 1, dtype=torch.long, device=DEVICE
    )

    ref_outputs = []
    ref_states = []
    for i in range(N):
        slot = i + 1
        q_i = l2norm_fwd(q[:, i : i + 1].contiguous())
        k_i = l2norm_fwd(k[:, i : i + 1].contiguous())
        init_state_i = state_buf[slot].transpose(-1, -2).unsqueeze(0)
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, i : i + 1],
            g[:, i : i + 1],
            beta[:, i : i + 1],
            initial_state=init_state_i,
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)

    state_buf_tri = state_buf.clone()
    tri_o, _ = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=state_buf_tri,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_t,
        ssm_state_indices=ssm_state_indices,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)

    for i in range(N):
        slot = i + 1
        tri_state = state_buf_tri[slot].transpose(-1, -2).unsqueeze(0)
        assert_close(f"ht_{i}", ref_states[i], tri_state, NPU_RMSE_RATIO_HT)

    assert torch.all(state_buf_tri[0] == 0), (
        "NULL slot (0) must not be modified"
    )
