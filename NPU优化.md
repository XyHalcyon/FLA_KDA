一、chunk_local_cumsum_vector_kernel优化点
问题：cumsum使用tl.dot做下三角矩阵乘法，scalar占比96.9%
当前实现(cumsum.py:120-161):
m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0) # 下三角mask
b_o = tl.dot(m_s, b_s) # scalar开销巨大
优化策略：
- 使用tl.cumsum原生算子
