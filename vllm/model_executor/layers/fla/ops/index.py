# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _cu_seqlens_to_lens_host(cu_seqlens: torch.Tensor) -> list[int]:
    # cu_seqlens is tiny (one int per sequence + 1); a single blocking D2H is
    # cheaper than dispatching cdiv/cumsum as AI_CPU ops on NPU.
    cs = cu_seqlens.detach().to("cpu").tolist()
    return [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    for seq_idx, n in enumerate(_cu_seqlens_to_lens_host(cu_seqlens)):
        n_chunks = (n + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            rows.append((seq_idx, c))
    if not rows:
        return torch.empty((0, 2), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    return torch.tensor(rows, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = [0]
    acc = 0
    for n in _cu_seqlens_to_lens_host(cu_seqlens):
        acc += (n + chunk_size - 1) // chunk_size
        out.append(acc)
    return torch.tensor(out, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
