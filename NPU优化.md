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

| 算子                             | 1K(us) |        |        | 8K(us) |        |        |
| -------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                  | 优化前 | 优化后 | 提升   | 优化前 | 优化后 | 提升   |
| chunk_local_cumsum_vector_kernel | 5126   | 154    | 97.00% | 41041  | 1408   | 96.57% |

## 二、chunk_gla_fwd_kernel_o性能优化点

**问题（瓶颈分析）：** `chunk_gla_fwd_kernel_o` 是 KDA 前向链路里把 `q·hᵀ`（chunk 间）与 `A·v`（chunk 内）合并写出 `o` 的输出 kernel。NPU profile 显示瓶颈不在算力，而在 program 启动开销：

- cube 计算单元只忙了 **0.02%** 的时间，HBM 带宽、向量单元、cube 单元三路硬件全部空闲；
- 每启动一个 program，Triton 翻译到 NPU 时会带上一笔约 **62 us 的隐式同步"启动税"**；
- 基线网格 `(NV=1, NT=16, B*H=32) = 512` program × 24 个 AIC 核 ≈ **21 轮 wave**，每轮都要付一次启动税；
- 同时 autotune 默认开了 24 组 `BK×BV×nw×ns` 配置，K=V=128 下小 BK/BV 让 `i_k`/`i_v` 退化成无意义的多步循环，循环体里还残留 `if i_k >= 0:` 和 NPU 上无效的 `allow_tf32=False`。

**优化策略：**

##### 方向 1：把 K 维循环拍扁为单步（含死代码清理）

原代码在 K 维度上分块循环：默认 `BK=64`、`K=128`，每个 program 要跑 2 次 `i_k` 循环，每次只覆盖一半 K 数据，附带的 `make_block_ptr`、地址偏移、`boundary_check` 全部翻倍。同时循环体里还有 `if i_k >= 0:` 这种永远为真的判断（`i_k` 从 0 起步），triton-ascend lowering 仍会为它生成一段冗余的控制流。

**改法：** autotune 配置直接固化为 `BK=128, BV=128, num_warps=2, num_stages=4`，让 `i_k`、`i_v` 都塌缩成单步；同时删掉 `if i_k >= 0:`。


##### 方向 2：H_PACK —— 一个 program 干 8 个 head 的活（最关键一招）

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


##### 当前实现（kda.py:1048-1168, 1192-1215）

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

##### 具体代码修改点

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

##### 性能数据

测试 case：`(B=1, T=1024, H=32, K=128, V=128, fp16)`，对应 `H_PACK=8`，program 网格从 `(1, 16, 32)` 收缩到 `(1, 16, 4)`，wave 数从 21 降到 3。

| 算子                   | 1K(us) |        |        | 8K(us) |        |        |
| ---------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                        | 优化前 | 优化后 | 提升   | 优化前 | 优化后 | 提升   |
| chunk_gla_fwd_kernel_o | 1368   | 233    | 82.97% | 10661  | 1726   | 83.81% |

> 单 program 时间多干 8× 工作只涨 27%，是这次优化生效的直接证据：多出来的 7 份 head 几乎"白送"，因为它们共享了同一笔 ~62 us 的启动税。



## 三、chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter性能优化点

**问题：** `intra_sub_inter` 算的是 chunk 内 *非对角*（`i_i > i_j`）的 BC×BC 子块，输出 `A` 与 `Aqk`。三笔可量化的浪费：

1. **i_k 内层循环白跑**：base 的 `BK=64`、`K=128`，每个 (i_i, i_j) 对都跑两轮 `for i_k in range(tl.cdiv(K, BK))`，每轮覆盖一半 K，附带 5 个 `make_block_ptr`、若干 `boundary_check` 全部翻倍。
2. **i_i-only 的载入仍困在 i_j 循环里**：`b_q / b_k / b_g / b_gn`、`exp(b_g - b_gn)` 这些只跟 i_i 有关的量，位于 `for i_j` 之内。NC=4 的真实形状下 (i_i=1..3, i_j=0..i_i-1) 共 6 个内层迭代，i_i=2/3 的载入分别被重复 2/3 次，MTE2 上有明显冗余。
3. **`b_A *= b_b[:, None]` 是一笔多余的 vec pass**：在累加完 `b_A += tl.dot(...)` 后再单独乘 beta，相当于一次额外的 `BC×BC` 元素遍历。

**优化策略：**

##### 方向 1：BK = next_power_of_2(K)，i_k 循环塌缩为单步

默认 `triton.Config({"BK": 64}, num_warps=4, num_stages=2)`，优化后 把这层 autotune 干掉，wrapper 显式传 `BK = max(next_power_of_2(K), 16)`，并在 kernel 入口加 `tl.static_assert(BK >= K)`，让 K 维一次性盖完。

`for i_k in range(tl.cdiv(K, BK))` 在 K=128、BK=128 下塌缩成单步，连带 i_k 相关的 `o_k = i_k * BK + tl.arange(0, BK)` 退化成纯 `tl.arange(0, BK)`、`make_block_ptr` 数量减半。

##### 方向 2：把 i_i-only 的载入 hoist 到 i_j 循环外（最关键一招）

代码结构虽然是 `for i_i / for i_j`，但 i_i-only 数据没提到外层。优化后直接挪：

- 提出来的：`p_q / p_k / p_g`（q/k/g 在 i_i 处的 BC×BK 块）、`b_gn`（i_i*BC 行的 g 向量）、`b_g`（i_i 块的 g）；
- 顺手把派生量 `b_eg = exp(b_g - b_gn[None, :])`、`b_kg = b_k * b_eg`、`b_qg = b_q * b_eg * scale` 一起算到外层——其中 `b_eg` 是 `b_k` / `b_q` 各算一次（共两次同样的 exp），优化后 合成一次，再分别乘 k 和 q。

i_j 循环里只剩跟 j 相关的 `b_gk / b_kt`、两次 BC×BK ↔ BK×BC 的 `tl.dot` 以及对应 `A / Aqk` 写出。NC=4 时内层平均长度 `(1+2+3)/3 = 2`，hoist 把这一层 MTE2/exp 重复读减一半左右；BK 塌缩之后 `make_block_ptr` 数量再砍一档。

注意，前提是方向 1 已让 `i_k` 退化为单步——否则 i_i-only 数据里的 `b_g` 是依赖 i_k 的，hoist 不出去。这也是 把 `tl.static_assert(BK >= K)` 写进 kernel 头部的原因，它把这条前置条件用断言固定下来。

##### 方向 3：把 `b_A *= b_b[:, None]` 折进 dot 尾乘

原来是先 `b_A += tl.dot(b_k, b_ktg)` 再单独 `b_A *= b_b[:, None]`，优化后 改为 `b_A = tl.dot(b_kg, b_ktg) * b_b[:, None]`——同时把 `b_A / b_Aqk` 的 `tl.zeros + +=` 累加器改成单步赋值（因为 i_k 已经塌缩，不需要累加）。

收益体感小但代码层面纯净化：少一次 `BC×BC=16×16` 的 vec pass，少一对 `tl.zeros` 初始化。

### 具体代码修改点

```python
@@ chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter @@
-@triton.autotune(
-    configs=[triton.Config({"BK": 64}, num_warps=4, num_stages=2)],
-    key=["BC"],
-)
+# 方向 1：autotune 撤掉，BK 由 wrapper 显式传 next_power_of_2(K)
 @triton.jit(do_not_specialize=["T"])
 def chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(...):
     ...
+    # 方向 1：单 BK 块覆盖 K，前置条件
+    tl.static_assert(BK >= K)
+
     for i_i in range(1, NC):
         if i_t * BT + i_i * BC < T:
+            # —— 方向 2：i_i-only 载入 hoist 到 i_j 循环外 ——
+            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1),
+                                    (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
+            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1),
+                                    (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
+            p_g = tl.make_block_ptr(g, (T, K), (H * K, 1),
+                                    (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
+            o_k  = tl.arange(0, BK)
+            m_k  = o_k < K
+            b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k,
+                           mask=m_k, other=0)
+            b_g  = tl.load(p_g, boundary_check=(0, 1))
+            b_eg = exp(b_g - b_gn[None, :])
+            b_kg = tl.load(p_k, boundary_check=(0, 1)) * b_eg
+            b_qg = tl.load(p_q, boundary_check=(0, 1)) * b_eg * scale
+
             for i_j in range(0, i_i):
-                b_A   = tl.zeros([BC, BC], dtype=tl.float32)
-                b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
-                for i_k in range(tl.cdiv(K, BK)):
-                    p_q = tl.make_block_ptr(q, ..., (i_t*BT+i_i*BC, i_k*BK), ...)
-                    p_k = tl.make_block_ptr(k, ..., (i_t*BT+i_i*BC, i_k*BK), ...)
-                    p_g = tl.make_block_ptr(g, ..., (i_t*BT+i_i*BC, i_k*BK), ...)
-                    b_kt = tl.make_block_ptr(k, ..., (i_k*BK, i_t*BT+i_j*BC), ...)
-                    p_gk = tl.make_block_ptr(g, ..., (i_k*BK, i_t*BT+i_j*BC), ...)
-                    o_k  = i_k * BK + tl.arange(0, BK)
-                    m_k  = o_k < K
-                    b_gn = tl.load(g + (i_t*BT+i_i*BC)*H*K + o_k, mask=m_k, other=0)
-                    b_g  = tl.load(p_g, boundary_check=(0, 1))
-                    b_k  = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
-                    b_gk = tl.load(p_gk, boundary_check=(0, 1))
-                    b_kt_val = tl.load(b_kt, boundary_check=(0, 1))
-                    b_ktg = b_kt_val * exp(b_gn[:, None] - b_gk)
-                    b_A  += tl.dot(b_k,  b_ktg)
-                    b_q   = tl.load(p_q, boundary_check=(0, 1))
-                    b_qg  = b_q * exp(b_g - b_gn[None, :]) * scale
-                    b_Aqk += tl.dot(b_qg, b_ktg)
-                b_A *= b_b[:, None]
+                b_kt  = tl.make_block_ptr(k, (K, T), (1, H * K),
+                                          (0, i_t * BT + i_j * BC), (BK, BC), (0, 1))
+                p_gk  = tl.make_block_ptr(g, (K, T), (1, H * K),
+                                          (0, i_t * BT + i_j * BC), (BK, BC), (0, 1))
+                b_gk  = tl.load(p_gk, boundary_check=(0, 1))
+                b_kt  = tl.load(b_kt, boundary_check=(0, 1))
+                b_ktg = b_kt * exp(b_gn[:, None] - b_gk)
+                # —— 方向 3：beta 折进尾乘，去掉累加器 ——
+                b_A   = tl.dot(b_kg, b_ktg) * b_b[:, None]
+                b_Aqk = tl.dot(b_qg, b_ktg)
                 tl.store(p_A,   b_A.to(A.dtype.element_ty),   boundary_check=(0, 1))
                 tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

@@ def chunk_kda_scaled_dot_kkt_fwd @@
     chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
         ...,
+        BK=BK,                                              # 方向 1：显式传 BK
         NC=NC,
     )
```

### **性能数据**

| 算子                                                | 1K(us) |        |        | 8K(us) |        |        |
| --------------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                                     | 优化前 | 优化后 | 提升   | 优化前 | 优化后 | 提升   |
| chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | 1664   | 820    | 50.72% | 13177  | 6344   | 51.86% |

## 四、merge_16x16_to_64x64_inverse_kernel性能优化点

`merge_16x16_to_64x64_inverse_kernel` 把 4 个 16×16 对角块各自做一次"逆传播 + 单位 I"（求 `(I + L_kk)⁻¹` 的迭代展开），再用 3 次 `tl.dot ∘ tl.dot` 把跨块的 21/32/43 等子块串起来，最终得到 64×64 下三角块逆。基线在 NPU 上 ~716 us，瓶颈在 vec_scalar 路径上的循环开销和无效 `tl.where`。

**问题（base 版本的浪费点）：**

1. **autotune 配置过宽**：base 配 `nw ∈ {2,4,8} × ns ∈ {2,3,4,5}` 共 12 组。NPU 上每开一个 config 都要 JIT profiling 一次首包，且实测最优只有一个点 `(8, 4)`，其余都是噪声。
2. **`tl.where(m_A, b_Ai_xx, 0)` 是空操作**：`m_A = o_i[:, None] > o_i[None, :]` 是严格下三角 mask，但 `b_Ai_xx` 来自上游 `kda.py` `mask_A` 构造时已经是严格下三角（对角与上三角恒为 0）。这一行 `where` 把 0 替成 0，纯属一次 16×16 vec pass 白跑——而且 `tl.where` 在大 tile 上会落到 aiv_scalar 路径。
3. **4 个对角块的行更新写成 4 个串行 for 循环**：每个 block 各自 `for i in range(2, 16)` / `for i in range(16+2, 32)` / 32+2..48 / 48+2..64，循环计数器、`min(...)` 边界判断、迭代 ramp-up 全部翻 4 倍。但其实 4 个块在每一轮里数据流互相独立（各自只读写自己的 `b_Ai_xx`），完全可以打成同一个 `for i` 内层。

本次共做三件事：

### 方向 1：autotune 收敛到单一最优配置

把 12 组 config 直接砍到 `(num_warps=8, num_stages=4)` 一个点。这是 [[kda-bottleneck-overview]] 上 solve_tril 这一支反复 sweep 后的 converged 选择，autotune 网格扩展到 18 个 config 测过是零 EV（参见 [[npu-triton-autotune-grid]]），把它固化下来既消掉首包 12× JIT 损耗，也避免 autotuner 单 run 内的噪声给出错误"赢家"（这个坑 `recompute_w_u_fwd` 上踩过，详见 [[kda-bottleneck-overview]] 里的脚注）。

### 方向 2：删掉无效的 `tl.where(m_A, ...)` 空 pass

直接 `b_Ai_xx = -b_Ai_xx`，省掉 4 次 16×16 `tl.where` 和对应的 `m_A` 掩码生成。

正确性前提：上游 `kda.py` 的 `mask_A` 在写出 A 时已经把对角和上三角清零，下三角矩阵性质由生产者保证；这一段在 NPU Triton 上又特别敏感（`tl.where` 的 2D 大 tile 会被降到 aiv_scalar 路径，参见 [[feedback-npu-triton-large-tile-scalar]]）。删掉是纯净化优化，但因为是 vec 路径上的 4 次 pass，体感不小。

### 方向 3：把 4 个独立的 for 循环 interleave 成单个 for（最关键一招）

base 的写法是 4 个串行 for：

```python
for i in range(2, min(16, T - i_t * BT)):           # block 11
    ...
for i in range(16 + 2, min(32, T - i_t * BT)):      # block 22
    ...
for i in range(32 + 2, min(48, T - i_t * BT)):      # block 33
    ...
for i in range(48 + 2, min(64, T - i_t * BT)):      # block 44
    ...
```

优化后合并成一个：

```python
T_local = T - i_t * BT
for i in range(2, min(16, T_local)):                # 共享同一个 i ∈ [2, 16)
    # block 11
    b_a_11 = -tl.load(A + (i_t*BT + i)      *H*BT + o_i)
    b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
    b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    # block 22 / 33 / 44 由编译期可剪枝的 if 守护，避免越界
    if 16 + i < T_local:
        b_a_22 = -tl.load(A + (i_t*BT + 16 + i) *H*BT + o_i + 16)
        ...
    if 32 + i < T_local:
        ...
    if 48 + i < T_local:
        ...
```

**为什么有效：**

- 4 个 block 在每一轮 i 内**数据流互相独立**——每一行 `b_a_xx` 只读写自己的 `b_Ai_xx`，没有跨块依赖。Triton-Ascend 后端能把 4 组 load / sum / where 在同一迭代里 overlap 进 MTE2/Vec 流水。
- 循环计数器、`min()` 边界、循环 ramp-up 在 4 个 block 之间被摊薄到 1 份，scalar overhead 从 4× 折回 1×。
- 重新参数化下标：`b_a_22` 里的 `(o_i == i - 16)[:, None]` 改成 `(o_i == i)[:, None]`、配合行偏移 `16 + i` 与列偏移 `+16` 匹配，使 4 个 block 共用同一个迭代变量 i ∈ [2, 16)，没有 i ∈ [18, 32) 这种漂移。

### 具体代码修改点

```python
@@ merge_16x16_to_64x64_inverse_kernel @@
 @triton.autotune(
     configs=[
-        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
-        for num_warps in [2, 4, 8]
-        for num_stages in [2, 3, 4, 5]
+        triton.Config({}, num_warps=nw, num_stages=ns)        # 方向 1：固化
+        for nw in [8]
+        for ns in [4]
     ],
     key=["H", "BT", "IS_VARLEN"],
 )
@@ def merge_16x16_to_64x64_inverse_kernel(...):
     o_i = tl.arange(0, 16)
-    m_A = o_i[:, None] > o_i[None, :]                         # 方向 2：删掉
     m_I = o_i[:, None] == o_i[None, :]
     ...
-    # [16, 16]
-    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)                      # 方向 2：4 次空 pass
-    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
-    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
-    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)
+    b_Ai_11 = -b_Ai_11                                        # 直接取负
+    b_Ai_22 = -b_Ai_22
+    b_Ai_33 = -b_Ai_33
+    b_Ai_44 = -b_Ai_44

-    for i in range(2, min(16, T - i_t * BT)):                 # 方向 3：4 个 for
-        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
-        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
-        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
-    for i in range(16 + 2, min(32, T - i_t * BT)):
-        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
-        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
-        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
-    for i in range(32 + 2, min(48, T - i_t * BT)):
-        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
-        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
-        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
-    for i in range(48 + 2, min(64, T - i_t * BT)):
-        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
-        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
-        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
+    # 方向 3：4 个独立循环 interleave 进单个 for，共享 i ∈ [2, 16)
+    T_local = T - i_t * BT
+    for i in range(2, min(16, T_local)):
+        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
+        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
+        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
+        if 16 + i < T_local:
+            b_a_22 = -tl.load(A + (i_t * BT + 16 + i) * H * BT + o_i + 16)
+            b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
+            b_Ai_22 = tl.where((o_i == i)[:, None], b_a_22, b_Ai_22)
+        if 32 + i < T_local:
+            b_a_33 = -tl.load(A + (i_t * BT + 32 + i) * H * BT + o_i + 32)
+            b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
+            b_Ai_33 = tl.where((o_i == i)[:, None], b_a_33, b_Ai_33)
+        if 48 + i < T_local:
+            b_a_44 = -tl.load(A + (i_t * BT + 48 + i) * H * BT + o_i + 48)
+            b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
+            b_Ai_44 = tl.where((o_i == i)[:, None], b_a_44, b_Ai_44)
```

### 性能数据

| 算子                                | 1K(us) |        |        | 8K(us) |        |        |
| ----------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                     | 优化前 | 优化后 | 提升   | 优化前 | 优化后 | 提升   |
| merge_16x16_to_64x64_inverse_kernel | 714    | 536    | 24.93% | 5317   | 4101   | 22.87% |

## 五、chunk_gated_delta_rule_fwd_kernel_h_blockdim64性能优化点

`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 沿 T 方向递推 KV 状态 `h`：每个 (i_t, i_b, i_h) program 顺序处理 NT 个 chunk，对每个 chunk 读 `(k, w, u/g/gk)`、累乘衰减、再 dot 出新 `h`。kernel 体本身是个紧凑递推，性能瓶颈在两侧：单 program 的循环体重不重，以及 grid 一次能不能把核心吃满。

**问题（base 的浪费点）：**

1. **autotune 网格虚胖**：base 配 `BV ∈ {32, 64} × num_warps ∈ {2, 4} × num_stages ∈ {2, 3, 4}` 共 12 组 config，且最大 BV=64。
2. **BV=64 在 V=128 形状下产生冗余 i_v 程序**：grid `(cdiv(V, BV), N*H)`，V=128 / BV=64 → `i_v` 维度 = 2，每个 (i_t, i_b, i_h) 在 V 方向上要跑两个 program，递推体被重复一份。
3. **wave 数被 grid 推高**：N=1, H=32, NV=2 → 64 个 program，24 AIC 核 → 2.67 wave；每轮 wave 都要付一次 program 启动税（参见 [[gla_fwd_o]] 上 ~62 us / wave 的同源开销）。

本次只做一件事：

### 方向 1：autotune 收敛到 `BV=128, num_warps=4, num_stages=3`

把 12 组 config 收成单点。要点是**扩展 BV 取值** —— main 的 `BV ∈ {32, 64}` 都覆盖不了 V=128 的全长，autotune 在这个网格里再怎么挑都会留下 `i_v` 维度 ≥ 2 的冗余 program。新配置直接把 BV 提到 128，让 `cdiv(V, BV) = 1`：

- grid 第 0 维（i_v）从 2 塌缩成 1，program 数 64 → 32（÷2）；
- wave 数 2.67 → 1.33；启动税被分母吃掉一半；
- kernel 体里所有按 `i_v` 翻倍的 `make_block_ptr`、boundary check 全部消失。

`num_warps=4, num_stages=3` 是这一支在 (4, 8) × (2, 3, 4) 网格上反复 sweep 后的 converged 选择，参见 [[kda-bottleneck-overview]]。

**这一招生效的前提：** V=128 是当前实际跑的 head_dim；如果上游传更大的 V，BV=128 会让 i_v 重新拆出多个程序，应当回退到原 autotune 网格让其重新选。



### 具体代码修改点（main → 当前）

```python
@@ chunk_gated_delta_rule_fwd_kernel_h_blockdim64 @@
 @triton.autotune(
-    configs=[
-        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
-        for num_warps in [2, 4]
-        for num_stages in [2, 3, 4]
-        for BV in [32, 64]                                      # 都 < V=128
-    ],
+    configs=[triton.Config({"BV": 128}, num_warps=4, num_stages=3)],
     key=["H", "K", "V", "BT"],
     use_cuda_graph=use_cuda_graph,
 )
```

### 性能数据

| 算子                                           | 1K(us) |        |        | 8K(us) |        |        |
| ---------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
|                                                | 优化前 | 优化后 | 提升   | 优化前 | 优化后 | 提升   |
| chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | 446    | 306    | 31.39% | 3323   | 2333   | 29.79% |
