---
title: "RoPE 旋转位置编码：从二维旋转到相对位置内积"
description: "以二维向量的手算示例推导 RoPE，解释旋转频率、相对位置、代码实现与 KV Cache 的位置索引。"
category: "transformer"
section: "Transformer 与注意力"
tags: ["RoPE", "Transformer", "位置编码"]
difficulty: "进阶"
order: 5
updated: "2026-11-23"
created: "2026-11-23"
draft: false
featured: true
readingTime: "约 17 分钟"
unlockRequired: true
previewRatio: 0.35
---

## 为什么 Transformer 需要额外的位置机制

标准 Self-Attention 使用 Query 与 Key 的点积计算位置之间的匹配分数，却不会凭空知道输入 Token 的先后顺序。对于语句“甲追乙”和“乙追甲”，词语相同而语义不同；如果模型忽略顺序，表示就不完整。

最直观的设计是把位置向量加到 Token Embedding。RoPE（Rotary Position Embedding）采用另一种做法：先通过投影计算 Query 和 Key，再对它们按位置进行旋转。这样位置因素直接进入 QKᵀ 的相似度，而不仅是输入表示的一个附加量。

## 从二维旋转矩阵开始

二维向量 v=(x,y) 旋转角度 θ，使用矩阵：

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}.
$$

普通写法：R(theta) = [[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]。

因此旋转后的两个分量分别为 x·cosθ−y·sinθ 和 x·sinθ+y·cosθ。旋转保持向量的欧氏长度，但改变方向。

RoPE 给不同位置分配不同的旋转角。若一对通道使用频率 ω，那么处于位置 p 的向量旋转 pω。注意这里的位置 p 是 Token 在整条序列中的逻辑索引，而不是一次 GPU 内核调用的本地行号。

## 为什么内积只与相对位置有关

设两个原始向量为 q 和 k，Query 处于位置 m，Key 处于位置 n。两者应用位置旋转后的内积为：

$$
(R(m\omega)q)^\top(R(n\omega)k)
=q^\top R(m\omega)^\top R(n\omega)k.
$$

普通写法：rotated_q dot rotated_k = q.T @ R(m*omega).T @ R(n*omega) @ k。

旋转矩阵满足 R(θ)ᵀ=R(−θ)，并且 R(a)R(b)=R(a+b)，因此：

$$
q^\top R(-m\omega)R(n\omega)k
=q^\top R((n-m)\omega)k.
$$

普通写法：relative_score = q.T @ R((key_position-query_position)*omega) @ k。

最后的旋转角只与 n−m 有关。这不是一句抽象的“相对位置感知”，而是由旋转矩阵的代数性质直接推导得到的。

需要区分两件事：位置索引 m、n 仍然是绝对索引；只是将两个已旋转向量做点积后，位置因素可以改写为相对位移。这并不意味着 RoPE 保证模型在任意长文本上都能泛化。

## 一个有数字的例子

取 q=(1,0)，k=(1,0)，频率 ω=π/4。没有 RoPE 时，无论两个 Token 离多远，点积都是 1。旋转后：

$$
\mathrm{score}(m,n)=\cos((n-m)\pi/4).
$$

普通写法：score(m,n) = cos((n-m)*pi/4)。

| Query 位置 | Key 位置 | 相对位移 | 点积 |
|---|---|---|---|
| 0 | 0 | 0 | 1 |
| 0 | 1 | 1 | 约 0.707 |
| 0 | 2 | 2 | 0 |
| 2 | 3 | 1 | 约 0.707 |
| 5 | 6 | 1 | 约 0.707 |

虽然位置对 (2,3) 和 (5,6) 的绝对位置不同，两者的相对位移相同，所以本例中的旋转位置因子相同。真实模型的 q、k 还包含内容信息，不能把这里的余弦值直接解释为模型的注意力权重。

## 高维向量怎么旋转

实际 Attention Head 的维度通常是大于 2 的偶数 d。RoPE 将通道分为 d/2 组，每组两个维度按独立频率 ω_i 旋转：

$$
\omega_i=b^{-2i/d},\qquad i=0,\ldots,d/2-1.
$$

普通写法：omega[i] = base ** (-2*i/head_dim)。

经典设置常选基数 b=10000。这样不同二维子空间的旋转速度不相同，同一个相对位置差会在多个频率上留下不同的相位关系。

可以将一个八维 Head 想成四个二维向量依次旋转：

~~~text
[x0,x1 | x2,x3 | x4,x5 | x6,x7]
   R(pω0)  R(pω1)  R(pω2)  R(pω3)
                 ↓
       旋转后的 Query / Key
                 ↓
           注意力内积
~~~

要留意通道配对约定：有的实现使用偶数/奇数通道配对，有的采用前后两半配对。两者可以经通道重排对应，但训练与推理时必须保持相同约定。

## NumPy 代码：只旋转 Q 和 K

下面的实现接收形状为 [sequence_length, head_dimension] 的二维数组，偶奇两维为一对，并允许传入绝对位置偏移：

~~~python
import numpy as np

def rope(x, offset=0, base=10000.0):
    length, dimension = x.shape
    assert dimension % 2 == 0
    p = np.arange(offset, offset + length, dtype=np.float64)
    omega = base ** (-2 * np.arange(dimension // 2) / dimension)
    angle = p[:, None] * omega[None, :]
    cosine, sine = np.cos(angle), np.sin(angle)
    pairs = x.reshape(length, dimension // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    out = np.stack([
        even * cosine - odd * sine,
        even * sine + odd * cosine
    ], axis=-1)
    return out.reshape(x.shape)

x = np.array([[1., 0., 1., 0.],
              [1., 0., 1., 0.]])
print(rope(x))
~~~

本例只用于验证数学结构；生产模型中的输入通常包含 Batch、Head 维度，并通过融合算子减少三角函数和内存访问的重复开销。

在典型 Decoder-only Block 中，先由隐藏状态投影 Q、K、V，再对 Q 和 K 应用 RoPE；Value 一般不执行对应旋转。Causal Mask 的作用仍是限制未来位置能否被读取，与位置旋转是两个独立机制。

## 用一个单元测试验证相对位置

若两组位置 (m,n) 和 (m+c,n+c) 的相对差一样，使用相同 q、k 和频率时，旋转后的内积应该相等：

~~~python
import numpy as np
theta = np.pi / 6

def rotate(v, pos):
    a = pos * theta
    return np.array([
        v[0]*np.cos(a)-v[1]*np.sin(a),
        v[0]*np.sin(a)+v[1]*np.cos(a)
    ])

q = np.array([0.6, -0.2])
k = np.array([0.3, 0.7])
a = rotate(q, 2) @ rotate(k, 5)
b = rotate(q, 8) @ rotate(k, 11)
print(a, b, np.allclose(a, b))
~~~

这条性质可以变成位置编码实现的回归测试。若结果差异过大，需要检查通道配对方式、角度频率和位置索引是否一致。

## KV Cache 与长上下文中的位置问题

Prefill 对输入位置 0 到 L−1 生成历史 K/V。Decode 的新增 Token 必须使用位置 L 的旋转角；下一步使用 L+1。如果每轮 Decode 都误把新 Token 当作位置 0，就会破坏它与历史 Key 的相对位置关系。

在 Prefix Cache、滑窗或截断上下文场景中，“缓存中的物理块位置”和“语义上的序列位置”也不一定相同。位置索引错乱会使模型出现异常重复、前后关联混乱等问题，且这种错误不一定触发运行时异常。

## 上下文长度外推的边界与评估

数学上可给任意 p 计算旋转角，但这不意味着训练范围以外的长文本能力自动成立。模型见过的位置分布有限，外推后可能无法稳定聚合远距离信息。

位置插值、频率缩放和 YaRN 等方案通过调整索引或频率缓解这个问题。对长上下文方案，建议分别测检索型任务、多跳推断、长文聚合和真实问答，而不是仅依靠某个简单 Needle 测试。

## 原始资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention 的矩阵计算](/docs/transformer/attention-four-tokens)
- [KV Cache 的显存与延迟](/docs/transformer/kv-cache)
