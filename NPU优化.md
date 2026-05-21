## 一、chunk_local_cumsum_vector_kernel优化点

**问题：**cumsum使用tl.dot做下三角矩阵乘法，scalar占比96.9%

当前实现(cumsum.py:120-161):

 m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0) # 下三角mask

 b_o = tl.dot(m_s, b_s) # scalar开销巨大

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

