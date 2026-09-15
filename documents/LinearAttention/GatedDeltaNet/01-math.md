# GatedDeltaNet：结构、训练公式与计算量

本文总结 Gated Delta Networks（GatedDeltaNet）的数学结构，并估算训练期间的计算量。符号约定为：序列长度 \(L\)、batch size \(B\)、head 数 \(H\)、每个 head 的 query/key 维度 \(d_k\)、value 维度 \(d_v\)，模型隐藏维度 \(D\approx H d_k\)。

## 相关论文

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)，ICLR 2025，提出 Gated Delta Rule。
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)，NeurIPS 2024，提出 DeltaNet 的 WY/UT chunkwise 训练算法。
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060)，提出 SSD 并给出 Mamba-2 的结构化线性训练视角。
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)，GatedDeltaNet 的门控线性注意力前身。
- [Kimi Linear](https://arxiv.org/abs/2510.26692)，将 GatedDeltaNet 的标量衰减推广为逐通道衰减。
- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)，进一步将擦除门和写入门解耦。

## 1. 状态空间形式

每个 head 维护一个矩阵状态：

\[
S_t\in\mathbb R^{d_v\times d_k},\qquad
o_t=S_tq_t.
\]

普通线性注意力为：

\[
S_t=S_{t-1}+v_tk_t^\top.
\]

它会无限累加 key-value 关联，状态饱和后容易发生 memory collision。

Mamba-2 使用全局标量衰减：

\[
S_t=\alpha_tS_{t-1}+v_tk_t^\top,
\qquad 0<\alpha_t<1.
\]

这种方式可以快速遗忘，但会对所有历史关联统一衰减。

DeltaNet 使用 delta rule：

\[
\begin{aligned}
S_t
&=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top\\
&=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
\end{aligned}
\]

其中 \(S_{t-1}k_t\) 是旧状态对当前 key 的预测，\(v_t-S_{t-1}k_t\) 是预测误差；因此 DeltaNet 只沿当前 key 方向进行有针对性的修改。

## 2. Gated Delta Rule

GatedDeltaNet 在 delta 更新之前先对旧状态做衰减：

\[
\boxed{
S_t=S_{t-1}\left[\alpha_t(I-\beta_tk_tk_t^\top)\right]
       +\beta_tv_tk_t^\top
}
\tag{1}
\]

等价形式为：

\[
\boxed{
S_t=\alpha_tS_{t-1}
 +\beta_t\left(v_t-\alpha_tS_{t-1}k_t\right)k_t^\top
}
\tag{2}
\]

- \(\alpha_t\)：状态级遗忘门，控制旧记忆保留比例；
- \(\beta_t\)：当前 key 方向的写入强度；
- \(\alpha_t\to0\)：快速清空状态；
- \(\alpha_t\to1\)：退化为纯 DeltaNet；
- \(\beta_t\) 通常限制在 \((0,1)\)，论文注释指出也可扩展到 \((0,2)\) 以允许更强的状态跟踪。

GatedDeltaNet 与 Mamba-2 共享标量衰减思想，但并不代数等价：Mamba-2 的写入是 \(v_tk_t^\top\)，而式 (2) 的写入包含对旧预测的 delta 修正。

## 3. Online-learning 解释

将状态 \(S\) 视为快速权重矩阵，当前 token 的关联记忆目标为：

\[
\mathcal L_t(S)=\frac12\|Sk_t-v_t\|_2^2.
\]

其梯度为：

\[
\nabla_S\mathcal L_t=(Sk_t-v_t)k_t^\top.
\]

先将旧状态衰减为 \(\widetilde S_{t-1}=\alpha_tS_{t-1}\)，再执行学习率为 \(\beta_t\) 的一步 SGD：

\[
\begin{aligned}
S_t
&=\widetilde S_{t-1}
 -\beta_t(\widetilde S_{t-1}k_t-v_t)k_t^\top\\
&=\alpha_tS_{t-1}
 +\beta_t(v_t-\alpha_tS_{t-1}k_t)k_t^\top.
\end{aligned}
\]

因此，\(\alpha_t\) 类似 adaptive weight decay，\(\beta_t\) 类似 token 级自适应学习率。论文给出的等价 online objective 是：

\[
\|S_t-\alpha_tS_{t-1}\|_F^2
-2\left\langle S_tk_t,\,\beta_t(v_t-\alpha_tS_{t-1}k_t)\right\rangle.
\]

这里的 online-learning 是网络内部状态更新的解释；模型参数本身仍使用标准语言模型损失：

\[
\mathcal L_{\rm LM}(\theta)
=-\frac1N\sum_t\log p_\theta(x_{t+1}\mid x_{\le t}).
\]

## 4. 网络 block

典型 GatedDeltaNet block 遵循 Llama 风格：token mixer 后接 SwiGLU MLP 和残差连接。

\[
x_t\longrightarrow
\{q_t,k_t,v_t,\alpha_t,\beta_t\}
\longrightarrow S_t\longrightarrow o_t
\longrightarrow\text{norm/output gate/output projection}.
\]

原论文的具体路径为：

- \(q,k,v\)：线性投影 -> short convolution -> SiLU；
- \(q,k\)：额外做 L2 normalization；
- \(\alpha,\beta\)：线性投影；
- \(o_t=S_tq_t\)：归一化后与一个 SiLU 输出门逐元素相乘，再做输出投影。

混合版本包括：

- GatedDeltaNet-H1：GatedDeltaNet + sliding-window attention；
- GatedDeltaNet-H2：Mamba-2 + GatedDeltaNet + sliding-window attention。

## 5. Chunkwise/WY 训练

逐 token 递推适合解码，但训练时会阻碍 GPU 并行。将序列划分为长度 \(C\) 的 chunk，在一个 chunk 内部分展开：

\[
S_r=S_0F_r+G_r,
\]

其中 \(F_r\) 是旧状态经过前 \(r\) 个 token 的变换，\(G_r\) 是 chunk 内新写入项的累积。

定义 chunk 内的累积衰减：

\[
\gamma_j=\prod_{i=1}^{j}\alpha_i,
\]

并构造衰减感知的因果矩阵 \(\Gamma\)。论文通过 UT/WY 表示把多个 rank-1/Householder 型变换改写成矩阵运算，核心中间量可写成：

\[
U_g=
\left[I+\operatorname{strictLower}\left(
\operatorname{diag}(\beta)(\Gamma\odot KK^\top)
\right)\right]^{-1}
\operatorname{diag}(\beta)V.
\]

随后使用 chunk 端点的衰减变量完成：

\[
S_{t+1}=\overrightarrow S_t+
\left(U_g-\overleftarrow W_tS_t^\top\right)^\top\overrightarrow K_t,
\]

\[
O_t=\overleftarrow Q_tS_t^\top+
(Q_tK_t^\top\odot M)
\left(U_g-\overleftarrow W_tS_t^\top\right).
\]

箭头表示将衰减传播到 chunk 的开头或结尾。这样 chunk 内主要是 GEMM、三角矩阵运算和固定大小状态传递，兼顾了线性序列复杂度和 Tensor Core 利用率。

## 6. 训练计算量

### 6.1 单 token 的递推量级

用式 (2) 实现单 head 的递推，需要：

1. 计算 \(S_{t-1}q_t\) 得到输出；
2. 计算 \(S_{t-1}k_t\) 得到旧预测；
3. 做一个误差向量与 \(k_t\) 的外积更新状态。

每项约为 \(d_vd_k\) 次乘加，因此单 token、单 head 需要约 \(3d_vd_k\) 次乘加（若按一次乘加计 2 FLOPs，则约 \(6d_vd_k\) FLOPs）：

\[
F_{\rm recurrent}=\Theta(d_vd_k),
\]

所有 head 合计：

\[
\Theta(Hd_vd_k).
\]

这是每个 token 的量级；整段序列则为 \(\Theta(BLHd_vd_k)\) 次乘加。

若 \(d_v=d_k=d\) 且 \(D=Hd\)，则约为 \(\Theta(Dd)\)，远小于全注意力的 \(\Theta(LD)\) 序列混合项。

### 6.2 Chunkwise 前向量级

chunk 内的 \(QK^\top\)、衰减矩阵、WY/UT 变换会产生 \(C\times C\) 的块矩阵。对固定 chunk size \(C\)，GatedDeltaNet token mixer 的主项可写成：

\[
F_{\rm GDN,fwd}
\approx
c_1BLD^2
{}+c_2BHLC(d_k+d_v)
{}+c_3BHLd_vd_k.
\]

其中：

- \(c_1BLD^2\)：q/k/v、输出投影等线性层；
- \(BHLC(d_k+d_v)\)：chunk 内的 \(QK^\top\)、value 聚合及 WY/UT 相关 GEMM；
- \(BHLd_vd_k\)：每个 chunk 的状态与 \(C\) 个低秩方向做交互；虽然 chunk 数是 \(L/C\)，但每个 chunk 的矩阵乘法含有 \(C\) 个方向，合计仍与 \(L\) 成正比；
- \(c_1,c_2,c_3\) 取决于融合 kernel、是否计算反向中间量以及 FLOP 统计口径。

因此，固定 \(C\) 时序列相关部分对 \(L\) 是线性的：

\[
F_{\rm seq}=O\left(BHLC(d_k+d_v)+BHLd_vd_k\right)=O(L).
\]

### 6.3 与全注意力比较

忽略 mask、softmax 和 dropout 的低阶项，标准多头注意力前向约为：

\[
F_{\rm Attn,fwd}
\approx 8BLD^2+4BL^2D.
\]

其中 \(8BLD^2\) 是 q/k/v/output 投影，\(4BL^2D\) 是 \(QK^\top\) 与 \(AV\)。GatedDeltaNet 将后者替换成近似 \(O(BLCD)\) 的 chunk 计算：

\[
4BL^2D
\quad\longrightarrow\quad
O(BLCD).
\]

当 \(C\ll L\) 时，长序列收益明显；当 \(L\) 较短时，线性投影、MLP、kernel 启动和内存读写可能成为主要成本，所以不能仅凭渐进复杂度预测实际吞吐。

### 6.4 训练而非前向的 FLOPs

反向传播需要对 q/k/v、门控和状态递推求梯度。对大多数矩阵乘法，训练 FLOPs 通常约为前向的 2--3 倍，因此可用：

\[
F_{\rm train}\approx(2\text{--}3)F_{\rm fwd}
\]

作为工程估算；这是近似值，不是论文报告的固定常数。

若只比较 token mixer 的序列混合部分，并令 \(d_k=d_v=d\)、\(D=Hd\)，可粗略写为：

\[
F_{\rm GDN,train}^{\rm mix}
\approx(2\text{--}3)\,c\,BLCD,
\]

而全注意力对应：

\[
F_{\rm Attn,train}^{\rm mix}
\approx(2\text{--}3)\,4BL^2D.
\]

二者序列混合项的理想比例约为：

\[
\frac{F_{\rm GDN}}{F_{\rm Attn}}
\sim O\left(\frac{C}{L}\right).
\]

### 6.5 数值例子

取 \(B=1\)、\(L=4096\)、\(D=4096\)、\(H=32\)、\(d_k=d_v=128\)、\(C=64\)：

- 全注意力的序列混合项约为
  \[
  4L^2D\approx2.75\times10^{11}
  \]
  次 FLOPs；
- GatedDeltaNet 的 \(LCD\) 级 chunk 项约为
  \[
  LCD\approx1.07\times10^9
  \]
  次 FLOPs，再乘 kernel 相关常数；状态更新项 \(BLDd_k\) 还会贡献约 \(2.15\times10^9\) 的同量级 FLOPs；
- 理想化的序列混合计算量比例约为
  \[
  \frac{C}{L}=\frac{64}{4096}=\frac1{64}.
  \]

但 q/k/v/output 投影本身约为 \(8LD^2\approx5.50\times10^{11}\) FLOPs，SwiGLU MLP 还会增加大量 \(D^2\) 计算。因此，在真实 LLM 中，GatedDeltaNet 的总训练 FLOPs 不会降低到全注意力的 \(1/64\)；\(1/64\) 只描述序列混合子项。长上下文时，注意力的 \(L^2\) 项快速增长，GatedDeltaNet 的优势才会逐渐显现。

## 7. 结论

GatedDeltaNet 的本质可以概括为：

\[
\text{全局状态擦除 }(\alpha_t)
\quad+\quad
\text{按 key 的误差修正 }(\beta_t).
\]

它把 DeltaNet 的关联记忆能力与 Mamba-2 式的动态遗忘结合起来。训练时并不是简单地逐 token 扫描，而是通过 chunkwise 的衰减展开和 WY/UT 表示，把大部分工作转成 GPU 友好的矩阵乘法；在固定 chunk size 下，序列混合对长度 \(L\) 呈线性复杂度，但完整模型的投影层和 MLP 仍然是重要的 FLOP 来源。
