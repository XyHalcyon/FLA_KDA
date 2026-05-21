## 一、chunk_local_cumsum_vector_kernel优化点

**问题：**cumsum使用tl.dot做下三角矩阵乘法，scalar占比96.9%

当前实现(cumsum.py:120-161):

```python
 m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0) # 下三角mask

 b_o = tl.dot(m_s, b_s) # scalar开销巨大
```

**优化策略：**

- 使用tl.cumsum原生算子

**具体代码修改点：**

```python
@@ -115,12 +115,6 @@ def chunk_local_cumsum_vector_kernel(
     else:
         bos, eos = i_b * T, i_b * T + T
 
-    o_i = tl.arange(0, BT)
-    if REVERSE:
-        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
-    else:
-        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
-
     if HEAD_FIRST:
         p_s = tl.make_block_ptr(
             s + (bos * H + i_h * T) * S,
@@ -157,7 +151,14 @@ def chunk_local_cumsum_vector_kernel(
         )
     # [BT, BS]
     b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
-    b_o = tl.dot(m_s, b_s, allow_tf32=False)
+    # Use native tl.cumsum instead of inefficient tl.dot with lower triangular matrix
+    # This reduces scalar operations significantly on NPU (from 96.9% to minimal)
+    b_o = tl.cumsum(b_s, axis=0)
+    if REVERSE:
+        # Reverse cumsum: for each BS column, compute from end to start
+        # equivalent to: cumsum(reverse(s)) then reverse back
+        b_z = tl.sum(b_s, axis=0)  # [BS] - total sum per column
+        b_o = -b_o + b_z[None, :] + b_s
     tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
```

**性能数据：**

| 算子                             | 1K(us) | 8K(us) |        |        |      |        |
| -------------------------------- | ------ | ------ | ------ | ------ | ---- | ------ |
| 优化前                           | 优化后 | 提升   | 优化前 | 优化后 | 提升 |        |
| chunk_local_cumsum_vector_kernel | 5126   | 154    | 97.00% | 41041  | 1408 | 96.57% |

## 二、chunk_gla_fwd_kernel_o性能优化点

**问题（瓶颈分析）：** `chunk_gla_fwd_kernel_o` 是 KDA 前向链路里把 `q·hᵀ`（chunk 间）与 `A·v`（chunk 内）合并写出 `o` 的输出 kernel。NPU profile 显示瓶颈不在算力，而在 program 启动开销：

- cube 计算单元只忙了 **0.02%** 的时间，HBM 带宽、向量单元、cube 单元三路硬件全部空闲；
- 每启动一个 program，Triton 翻译到 NPU 时会带上一笔约 **62 us 的隐式同步"启动税"**；
- 基线网格 `(NV=1, NT=16, B*H=32) = 512` program × 24 个 AIC 核 ≈ **21 轮 wave**，每轮都要付一次启动税；
- 同时 autotune 默认开了 24 组 `BK×BV×nw×ns` 配置，K=V=128 下小 BK/BV 让 `i_k`/`i_v` 退化成无意义的多步循环，循环体里还残留 `if i_k >= 0:` 和 NPU 上无效的 `allow_tf32=False`。

本次共做四件事：

### 方向 1：把 K 维循环拍扁为单步（含死代码清理）

原代码在 K 维度上分块循环：默认 `BK=64`、`K=128`，每个 program 要跑 2 次 `i_k` 循环，每次只覆盖一半 K 数据，附带的 `make_block_ptr`、地址偏移、`boundary_check` 全部翻倍。同时循环体里还有 `if i_k >= 0:` 这种永远为真的判断（`i_k` 从 0 起步），triton-ascend lowering 仍会为它生成一段冗余的控制流。

**改法：** autotune 配置直接固化为 `BK=128, BV=128, num_warps=2, num_stages=4`，让 `i_k`、`i_v` 都塌缩成单步；同时删掉 `if i_k >= 0:`。


### 方向 3：H_PACK —— 一个 program 干 8 个 head 的活（最关键一招）

这是收益最大的一招，直接对准启动税。

**问题数学化：**
- 基线：512 program / 24 AIC ≈ **21 轮 wave**，单 program ~64 us → **21 × 64 ≈ 1344 us**；
- 每轮 wave 都要付一次 ~62 us 的启动税，cube 利用率 0.02%，HBM/Vec/Cube 三路全部空闲，说明硬件不是不够用，而是被 wave 数撬住了；要降总耗时，唯一的杠杆是降 wave 数。

**改法：** 让一个 program 一次性算 8 个相邻 head。
- program 网格从 `(NV, NT, B*H)` 改成 `(NV, NT, B*(H//H_PACK))`，program 数从 512 降到 64（÷8）；
- kernel 内用 `tl.static_range(H_PACK)` 静态展开 8 个 head 的循环，`bos`、`m_s` 等与 head 无关的标量在展开块外只算一次，常量地址偏移在编译期就被打成立即数；
- `H_PACK` 自适应取 `8 / 4 / 2 / 1`，保证不同 H 配置下都能整除。

**收益数学化：**
- program 数：512 → 64（÷8），wave 数：21 → **3**（24 AIC × 3 wave = 72 ≥ 64）；
- 单 program 时间：64 us → **82 us**（多干 8× 工作只涨 27%——多出来的活几乎"白送"，因为 8 倍工作共享同一笔启动税）；
- 总耗时：1344 us → **3 × 82 ≈ 246 us**，和实测 **248 us** 对得上。


### 当前实现（kda.py:1048-1168, 1192-1215）

```python
@triton.autotune(
    configs=[triton.Config({"BK": 128, "BV": 128}, num_warps=2, num_stages=4)],
    key=["BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q, v, g, h, o, A, cu_seqlens, chunk_indices, scale,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    H_PACK: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    HP: tl.constexpr = H // H_PACK
    i_b = i_bp // HP
    i_hp = i_bp % HP
    ...
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    # 一个 program 静态展开 H_PACK 个 head：bos/m_s 只算一次，
    # H 维地址偏移在编译期常量化。
    for i_pack in tl.static_range(H_PACK):
        i_h = i_hp * H_PACK + i_pack
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):  # BK=128, K=128 → 单步
            ...
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))  # 删掉 if i_k>=0
        ...
        b_o += tl.dot(b_A, b_v)            # 去掉 allow_tf32=False
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_o_gk(...):
    H_PACK = (
        8 if H % 8 == 0 else (4 if H % 4 == 0 else (2 if H % 2 == 0 else 1))
    )
    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * (H // H_PACK))
    chunk_gla_fwd_kernel_o[grid](..., H_PACK=H_PACK)
```

### 具体代码修改点

```python
@@ -1006,13 +1047,7 @@ def recompute_w_u_fwd(

 @triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
 @triton.autotune(
-    configs=[
-        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
-        for BK in [32, 64]
-        for BV in [64, 128]
-        for num_warps in [2, 4, 8]
-        for num_stages in [2, 3, 4]
-    ],
+    configs=[triton.Config({"BK": 128, "BV": 128}, num_warps=2, num_stages=4)],
     key=["BT"],
 )
@@ -1033,10 +1068,13 @@ def chunk_gla_fwd_kernel_o(
     BV: tl.constexpr,
+    H_PACK: tl.constexpr,
     IS_VARLEN: tl.constexpr,
 ):
-    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
-    i_b, i_h = i_bh // H, i_bh % H
+    i_v, i_t, i_bp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
+    HP: tl.constexpr = H // H_PACK
+    i_b = i_bp // HP
+    i_hp = i_bp % HP
@@ -1056,71 +1094,78 @@
     m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

-    b_o = tl.zeros([BT, BV], dtype=tl.float32)
-    for i_k in range(tl.cdiv(K, BK)):
-        ...
-        if i_k >= 0:
-            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
-    ...
-    b_o += tl.dot(b_A, b_v, allow_tf32=False)
-    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
+    for i_pack in tl.static_range(H_PACK):
+        i_h = i_hp * H_PACK + i_pack
+
+        b_o = tl.zeros([BT, BV], dtype=tl.float32)
+        for i_k in range(tl.cdiv(K, BK)):
+            ...
+            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
+        ...
+        b_o += tl.dot(b_A, b_v)
+        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@@ -1144,8 +1189,12 @@ def chunk_gla_fwd_o_gk(
     NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

+    H_PACK = (
+        8 if H % 8 == 0 else (4 if H % 4 == 0 else (2 if H % 2 == 0 else 1))
+    )
+
     def grid(meta):
-        return (cdiv(V, meta["BV"]), NT, B * H)
+        return (cdiv(V, meta["BV"]), NT, B * (H // H_PACK))

     chunk_gla_fwd_kernel_o[grid](
         ...
+        H_PACK=H_PACK,
     )
```

### 性能数据

测试 case：`(B=1, T=1024, H=32, K=128, V=128, fp16)`，对应 `H_PACK=8`，program 网格从 `(1, 16, 32)` 收缩到 `(1, 16, 4)`，wave 数从 21 降到 3。

| 算子                   | 1K(us) | 8K(us) |        |        |      |        |
| ---------------------- | ------ | ------ | ------ | ------ | ---- | ------ |
| 优化前                 | 优化后 | 提升   | 优化前 | 优化后 | 提升 |        |
| chunk_gla_fwd_kernel_o | 1368   | 233    | 82.97% | 10661  | 1726 | 83.81% |

> 单 program 时间多干 8× 工作只涨 27%，是这次优化生效的直接证据：多出来的 7 份 head 几乎"白送"，因为它们共享了同一笔 ~62 us 的启动税。

