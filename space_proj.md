
---

## 1. 问题建模：从“四维独立未知数”到“一维快衰落 + 四维慢空间签名”（条件模型）

### 1.1 接收导频解扩

设 4 根天线上，用户 1 和用户 2 在 RE1、RE3 上有导频。天线 $m$ 在这两个 RE 上的接收向量为：

$$
\mathbf{r}_m = h_{1,m}\mathbf{p}_1 + h_{2,m}\mathbf{p}_2 + \mathbf{i}_m + \mathbf{n}_m
$$

其中：

- $\mathbf{p}_1,\mathbf{p}_2$ 分别是用户 1、用户 2 的导频码，长度 2；
- 满足正交性：$\mathbf{p}_1^H\mathbf{p}_2 = 0$，并归一化 $\|\mathbf{p}_1\|^2 = 1$；
- $\mathbf{i}_m$ 是用户 3/4 数据或泄露带来的干扰；
- $\mathbf{n}_m$ 是热噪声。

对用户 1 做码域解扩：

$$
\hat h_{1,m}^{LS} = \mathbf{p}_1^H \mathbf{r}_m
= h_{1,m} + \mathbf{p}_1^H(\mathbf{i}_m+\mathbf{n}_m)
$$

写成 4 天线向量形式：

$$
\hat{\mathbf{h}}_1^{LS}
=
\mathbf{h}_1 + \mathbf{v}
\tag{1}
$$

其中：

$$
\mathbf{h}_1 =
\begin{bmatrix}
h_{1,1}\\
h_{1,2}\\
h_{1,3}\\
h_{1,4}
\end{bmatrix}
,\qquad
\mathbf{v} =
\begin{bmatrix}
v_1\\
v_2\\
v_3\\
v_4
\end{bmatrix}
$$

$v_m = \mathbf{p}_1^H(\mathbf{i}_m+\mathbf{n}_m)$ 是解扩后的等效噪声/干扰。

如果主要是热噪声，则：

$$
\mathbb{E}[\mathbf{v}\mathbf{v}^H] = \sigma_v^2 \mathbf{I}_4
\tag{2}
$$

如果存在用户 3/4 数据泄露等色干扰，则写为：

$$
\mathbb{E}[\mathbf{v}\mathbf{v}^H] = \mathbf{C}_v
\tag{3}
$$

---

### 1.2 用户 1 的空间签名模型

可检验的假设是：用户 1 在 4 根天线上的信道近似由同一个快衰落标量 $\alpha_1$ 乘以空间签名 $\mathbf{c}_1$ 表示。该结构不是一般多径信道的必然性质；要求签名在独立 TTI 间稳定，且所用导频信道频响落在该签名或子空间内。多径、阵列/极化差异、时变与导频残留均可能造成模型失配。

因此可以建模为：

$$
\mathbf{h}_1 = \alpha_1 \mathbf{c}_1
\tag{4}
$$

其中：

$$
\mathbf{c}_1 =
\begin{bmatrix}
1\\
\epsilon_{21}\\
\epsilon_{31}\\
\epsilon_{41}
\end{bmatrix}
$$

解释：

- $\mathbf{c}_1$ 是用户 1 的慢变空间签名，由阵列方向图、波束指向、用户地理位置决定；
- $\alpha_1$ 是快衰落标量，包含瞬时幅相变化；
- 天线 1 是用户 1 的主波束方向，因此归一化 $c_{1,1}=1$；
- $\epsilon_{21},\epsilon_{31},\epsilon_{41}$ 分别是用户 1 泄露到天线 2/3/4 的复数系数。

因此，用户 1 的真实信道不是 4 个自由未知数，而是：

$$
\mathbf{h}_1 =
\alpha_1
\begin{bmatrix}
1\\
\epsilon_{21}\\
\epsilon_{31}\\
\epsilon_{41}
\end{bmatrix}
$$

自由度从 4 降低到了 1。

---

## 2. 传统 LS 估计的病态表现

传统 LS 估计为：

$$
\hat{\mathbf{h}}_1^{LS}
=
\mathbf{h}_1 + \mathbf{v}
=
\alpha_1 \mathbf{c}_1 + \mathbf{v}
\tag{5}
$$

如果只考虑白噪声：

$$
\mathbb{E}[\mathbf{v}\mathbf{v}^H] = \sigma_v^2 \mathbf{I}_4
$$

那么第 $m$ 根天线上 LS 信道估计的误差方差为：

$$
\mathbb{E}\left[
\left|\hat h_{1,m}^{LS} - h_{1,m}\right|^2
\right]
=
\sigma_v^2
\tag{6}
$$

注意，这个误差方差对所有天线都是一样的，与 $|c_{1,m}|$ 无关。

因此第 $m$ 根天线的信道估计 SNR 为：

$$
\gamma_{m}^{LS}
=
\frac{|\alpha_1|^2 |c_{1,m}|^2}{\sigma_v^2}
\tag{7}
$$

当天线 2/3/4 上用户 1 的泄露系数很小时，例如 $|c_{1,3}|=0.1$，则：

$$
\gamma_3^{LS}
=
\frac{|\alpha_1|^2 \cdot 0.01}{\sigma_v^2}
$$

这说明天线 3/4 上的 LS 估计几乎被噪声完全淹没。

---

## 3. 优化方案：空间匹配投影估计

既然真实信道结构为：

$$
\mathbf{h}_1 = \alpha_1 \mathbf{c}_1
$$

而 $\mathbf{c}_1$ 是已知或可长期估计的，那么当前 TTI 内只需要估计标量 $\alpha_1$。

### 3.1 最小二乘投影

我们用一个标量 $\alpha$ 去拟合观测 $\hat{\mathbf{h}}_1^{LS}$，构造代价函数：

$$
J(\alpha)
=
\left\|
\hat{\mathbf{h}}_1^{LS} - \alpha \mathbf{c}_1
\right\|^2
\tag{8}
$$

展开：

$$
J(\alpha)
=
\left(
\hat{\mathbf{h}}_1^{LS} - \alpha \mathbf{c}_1
\right)^H
\left(
\hat{\mathbf{h}}_1^{LS} - \alpha \mathbf{c}_1
\right)
$$

对 $\alpha^*$ 求偏导并令其为 0：

$$
\frac{\partial J}{\partial \alpha^*}
=
-\mathbf{c}_1^H\hat{\mathbf{h}}_1^{LS}
+
\alpha \mathbf{c}_1^H\mathbf{c}_1
=
0
$$

得到：

$$
\boxed{
\hat{\alpha}_1
=
\frac{\mathbf{c}_1^H \hat{\mathbf{h}}_1^{LS}}
{\|\mathbf{c}_1\|^2}
}
\tag{9}
$$

这就是空间匹配滤波，也叫做空间最大比合并。

---

### 3.2 代入观测模型

将 $\hat{\mathbf{h}}_1^{LS} = \alpha_1 \mathbf{c}_1 + \mathbf{v}$ 代入：

$$
\hat{\alpha}_1
=
\frac{\mathbf{c}_1^H(\alpha_1 \mathbf{c}_1 + \mathbf{v})}
{\|\mathbf{c}_1\|^2}
$$

拆开：

$$
\hat{\alpha}_1
=
\alpha_1 \frac{\mathbf{c}_1^H\mathbf{c}_1}{\|\mathbf{c}_1\|^2}
+
\frac{\mathbf{c}_1^H\mathbf{v}}{\|\mathbf{c}_1\|^2}
$$

因为 $\mathbf{c}_1^H\mathbf{c}_1 = \|\mathbf{c}_1\|^2$，所以：

$$
\boxed{
\hat{\alpha}_1
=
\alpha_1
+
\frac{\mathbf{c}_1^H \mathbf{v}}
{\|\mathbf{c}_1\|^2}
}
\tag{10}
$$

这个式子的物理意义非常强：

$$
\mathbf{c}_1^H \hat{\mathbf{h}}_1^{LS}
=
c_{1,1}^* \hat h_{1,1}^{LS}
+
c_{1,2}^* \hat h_{1,2}^{LS}
+
c_{1,3}^* \hat h_{1,3}^{LS}
+
c_{1,4}^* \hat h_{1,4}^{LS}
$$

它把天线 1 上的强信号、天线 2/3/4 上的弱泄露信号按照各自的相位和幅度比例相干相加。

也就是说，天线 2/3/4 上的泄露信号没有被丢弃，而是被合并进了 $\hat{\alpha}_1$ 中。

---

### 3.3 重构高质量信道

得到 $\hat{\alpha}_1$ 后，再乘回空间签名：

$$
\boxed{
\hat{\mathbf{h}}_1^{enh}
=
\hat{\alpha}_1 \mathbf{c}_1
}
\tag{11}
$$

这就是增强后的 $4 \times 1$ 信道估计。

它的误差为：

$$
\hat{\mathbf{h}}_1^{enh} - \mathbf{h}_1
=
\left(
\frac{\mathbf{c}_1^H\mathbf{v}}{\|\mathbf{c}_1\|^2}
\right)
\mathbf{c}_1
\tag{12}
$$

---

## 4. 噪声方差与误差协方差推导

以下推导针对线性估计 $\hat{\mathbf h}=W\hat{\mathbf h}^{LS}$，并要求 $\hat{\mathbf h}^{LS}=\mathbf h+\mathbf v$ 中的零均值误差 $\mathbf v$ 协方差 $C_v$ 正确描述 DMRS 估计后的误差。对任意固定投影 $P$，真实误差包含两项：

$$
\mathbb E\|P\hat{\mathbf h}^{LS}-\mathbf h\|^2
=\operatorname{tr}(P C_v P^H)+\|(I-P)\mathbf h\|^2
$$

第一项是噪声传播，第二项是空间模型失配偏差；在高 SNR 时噪声趋近于零而偏差不会消失，故可能导致 CRC/BLER 退化。WLS 式 (21) 仅在所用 $C_v$ 合适、签名正确且观测模型成立时最小化噪声误差，不能推出一般多径信道的 BLER 收益。

仓库中的 `add_awgn_resource_grid` 返回输入资源网格的每 RX 天线 AWGN 方差；`DftSOfdmDmrsEstimator.forward` 则由 tap-fit 协方差 `cov_user`、频域基函数和该 AWGN 方差计算逐天线/频点 `err_var`。两者不是同一随机变量，不能直接以输入 AWGN 方差判定投影是否安全。

### 4.1 标量 $\hat{\alpha}_1$ 的噪声方差

从式 (10) 可得：

$$
\hat{\alpha}_1 - \alpha_1
=
\frac{\mathbf{c}_1^H \mathbf{v}}{\|\mathbf{c}_1\|^2}
$$

噪声项为：

$$
n_{eff}
=
\frac{\mathbf{c}_1^H \mathbf{v}}{\|\mathbf{c}_1\|^2}
$$

若 $\mathbb{E}[\mathbf{v}\mathbf{v}^H] = \sigma_v^2 \mathbf{I}_4$，则：

$$
\mathbb{E}[|n_{eff}|^2]
=
\frac{
\mathbf{c}_1^H \mathbb{E}[\mathbf{v}\mathbf{v}^H]\mathbf{c}_1
}{
\|\mathbf{c}_1\|^4
}
$$

代入 $\mathbb{E}[\mathbf{v}\mathbf{v}^H] = \sigma_v^2 \mathbf{I}_4$：

$$
\mathbb{E}[|n_{eff}|^2]
=
\frac{
\sigma_v^2 \mathbf{c}_1^H\mathbf{c}_1
}{
\|\mathbf{c}_1\|^4
}
=
\frac{
\sigma_v^2 \|\mathbf{c}_1\|^2
}{
\|\mathbf{c}_1\|^4
}
$$

所以：

$$
\boxed{
\mathbb{E}
\left[
|\hat{\alpha}_1 - \alpha_1|^2
\right]
=
\frac{\sigma_v^2}{\|\mathbf{c}_1\|^2}
}
\tag{13}
$$

因此标量 $\hat{\alpha}_1$ 的估计 SNR 为：

$$
\boxed{
\gamma_{\alpha}
=
\frac{|\alpha_1|^2}
{\frac{\sigma_v^2}{\|\mathbf{c}_1\|^2}}
=
\frac{|\alpha_1|^2 \|\mathbf{c}_1\|^2}
{\sigma_v^2}
}
\tag{14}
$$

---

### 4.2 重构信道的误差协方差

增强后的信道为：

$$
\hat{\mathbf{h}}_1^{enh}
=
\hat{\alpha}_1 \mathbf{c}_1
$$

其误差为：

$$
\mathbf{e}
=
\hat{\mathbf{h}}_1^{enh} - \mathbf{h}_1
=
(\hat{\alpha}_1 - \alpha_1)\mathbf{c}_1
$$

误差协方差矩阵为：

$$
\mathbf{R}_e
=
\mathbb{E}[\mathbf{e}\mathbf{e}^H]
=
\mathbb{E}
\left[
|\hat{\alpha}_1 - \alpha_1|^2
\right]
\mathbf{c}_1\mathbf{c}_1^H
$$

代入 (13)：

$$
\boxed{
\mathbf{R}_e
=
\frac{\sigma_v^2}{\|\mathbf{c}_1\|^2}
\mathbf{c}_1\mathbf{c}_1^H
}
\tag{15}
$$

第 $m$ 根天线上增强信道估计的误差方差为：

$$
\mathbb{E}
\left[
|\hat h_{1,m}^{enh} - h_{1,m}|^2
\right]
=
\frac{\sigma_v^2 |c_{1,m}|^2}{\|\mathbf{c}_1\|^2}
\tag{16}
$$

对比传统 LS 的第 $m$ 根天线误差方差 $\sigma_v^2$，增强后的误差方差变成了：

$$
\sigma_{m,enh}^2
=
\sigma_v^2 \cdot
\frac{|c_{1,m}|^2}{\|\mathbf{c}_1\|^2}
$$

如果 $|c_{1,m}|<1$，这个因子小于 1，说明弱天线上的噪声被大幅压缩。

---

### 4.3 适用边界

式 (13)–(19) 假设 $\mathbf c_1$ 已知、跨独立 TTI 稳定、真实信道精确满足 $\mathbf h_1=\alpha_1\mathbf c_1$，且导频后的误差协方差确为假设的白噪声协方差。若用样本估计的子空间，有限训练样本与子空间漂移还会引入额外误差；投影残差只是失配的可观测代理，并非真实误差的无偏估计。数值示例仅说明理想模型下的估计方差比，不代表实际 CDL、BLER 或当前实现的测量结果。

---

## 5. SNR 增益推导与数值示例

### 5.1 第 $m$ 根天线的 SNR

第 $m$ 根天线上用户 1 的信道功率为：

$$
S_m
=
|\alpha_1|^2 |c_{1,m}|^2
$$

传统 LS 估计该天线上的噪声方差为 $\sigma_v^2$，因此：

$$
\gamma_{m}^{LS}
=
\frac{|\alpha_1|^2 |c_{1,m}|^2}{\sigma_v^2}
\tag{17}
$$

增强估计该天线上的噪声方差为：

$$
\frac{\sigma_v^2 |c_{1,m}|^2}{\|\mathbf{c}_1\|^2}
$$

因此：

$$
\gamma_{m}^{enh}
=
\frac{|\alpha_1|^2 |c_{1,m}|^2}
{\frac{\sigma_v^2 |c_{1,m}|^2}{\|\mathbf{c}_1\|^2}}
=
\frac{|\alpha_1|^2 \|\mathbf{c}_1\|^2}{\sigma_v^2}
\tag{18}
$$

关键结论：

$$
\boxed{
\gamma_{m}^{enh}
=
\frac{\|\mathbf{c}_1\|^2}{|c_{1,m}|^2}
\gamma_{m}^{LS}
}
\tag{19}
$$

也就是说，第 $m$ 根天线的信道估计 SNR 提升倍数为：

$$
\frac{\|\mathbf{c}_1\|^2}{|c_{1,m}|^2}
$$

对于主天线，$|c_{1,1}|=1$，提升倍数为：

$$
\frac{\|\mathbf{c}_1\|^2}{1}
=
\|\mathbf{c}_1\|^2
$$

对于弱天线，提升更加显著。

---

### 5.2 数值示例

假设：

$$
\mathbf{c}_1 =
\begin{bmatrix}
1\\
0.3\\
0.1\\
0.1
\end{bmatrix}
$$

则：

$$
\|\mathbf{c}_1\|^2
=
1^2 + 0.3^2 + 0.1^2 + 0.1^2
=
1 + 0.09 + 0.01 + 0.01
=
1.11
$$

主天线 SNR 增益：

$$
10\log_{10}(1.11)
\approx
0.453\ \text{dB}
$$

即主天线信道估计 SNR 提升约 11%。

弱天线 2，$|c_{1,2}|^2=0.09$，增益为：

$$
\frac{1.11}{0.09}
\approx
12.33
\approx
10.9\ \text{dB}
$$

弱天线 3，$|c_{1,3}|^2=0.01$，增益为：

$$
\frac{1.11}{0.01}
=
111
\approx
20.45\ \text{dB}
$$

这说明：

- 主天线信道估计精度略有提升；
- 弱天线信道估计精度提升了一个数量级甚至两个数量级；
- 原本几乎全被噪声统治的天线 3/4，现在通过空间签名重构，噪声被大幅压缩。

---

## 6. 色噪声/多用户残留干扰下的推广：空间 LMMSE

如果解扩后 $\mathbf{v}$ 不是白噪声，而是包含用户 3/4 数据泄露的色噪声，那么简单投影：

$$
\hat{\alpha}_1
=
\frac{\mathbf{c}_1^H \hat{\mathbf{h}}_1^{LS}}
{\|\mathbf{c}_1\|^2}
$$

不是最优的。

此时应使用加权最小二乘：

$$
J_w(\alpha)
=
\left(
\hat{\mathbf{h}}_1^{LS} - \alpha \mathbf{c}_1
\right)^H
\mathbf{C}_v^{-1}
\left(
\hat{\mathbf{h}}_1^{LS} - \alpha \mathbf{c}_1
\right)
\tag{20}
$$

对 $\alpha^*$ 求导并令为 0：

$$
-
\mathbf{c}_1^H \mathbf{C}_v^{-1}
\hat{\mathbf{h}}_1^{LS}
+
\alpha \mathbf{c}_1^H \mathbf{C}_v^{-1} \mathbf{c}_1
=
0
$$

得到：

$$
\boxed{
\hat{\alpha}_1^{WLS}
=
\frac{
\mathbf{c}_1^H \mathbf{C}_v^{-1}
\hat{\mathbf{h}}_1^{LS}
}{
\mathbf{c}_1^H \mathbf{C}_v^{-1} \mathbf{c}_1
}
}
\tag{21}
$$

这是白化后匹配投影。

其估计误差方差为：

$$
\mathbb{E}
\left[
|\hat{\alpha}_1^{WLS} - \alpha_1|^2
\right]
=
\frac{1}{\mathbf{c}_1^H \mathbf{C}_v^{-1} \mathbf{c}_1}
\tag{22}
$$

---

### 6.1 完整空间 LMMSE 推导

假设 $\alpha_1$ 具有先验统计：

$$
\mathbb{E}[\alpha_1] = 0,\qquad
\mathbb{E}[|\alpha_1|^2] = P
$$

则：

$$
\mathbf{h}_1 = \alpha_1 \mathbf{c}_1
$$

其先验协方差为：

$$
\mathbf{R}_{h_1}
=
\mathbb{E}[\mathbf{h}_1 \mathbf{h}_1^H]
=
P \mathbf{c}_1 \mathbf{c}_1^H
\tag{23}
$$

观测模型为：

$$
\hat{\mathbf{h}}_1^{LS}
=
\mathbf{h}_1 + \mathbf{v}
$$

噪声/干扰协方差为：

$$
\mathbf{C}_v
=
\mathbb{E}[\mathbf{v}\mathbf{v}^H]
$$

LMMSE 估计为：

$$
\boxed{
\hat{\mathbf{h}}_1^{LMMSE}
=
\mathbf{R}_{h_1}
\left(
\mathbf{R}_{h_1} + \mathbf{C}_v
\right)^{-1}
\hat{\mathbf{h}}_1^{LS}
}
\tag{24}
$$

利用 Woodbury 矩阵求逆引理：

$$
\left(
\mathbf{C}_v + P \mathbf{c}_1 \mathbf{c}_1^H
\right)^{-1}
=
\mathbf{C}_v^{-1}
-
\frac{P \mathbf{C}_v^{-1}\mathbf{c}_1\mathbf{c}_1^H \mathbf{C}_v^{-1}}
{1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1}
$$

于是：

$$
P\mathbf{c}_1\mathbf{c}_1^H
\left(
\mathbf{C}_v + P \mathbf{c}_1 \mathbf{c}_1^H
\right)^{-1}
=
\frac{P}{1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1}
\mathbf{c}_1 \mathbf{c}_1^H \mathbf{C}_v^{-1}
$$

因此：

$$
\hat{\mathbf{h}}_1^{LMMSE}
=
\frac{P}{1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1}
\mathbf{c}_1
\left(
\mathbf{c}_1^H \mathbf{C}_v^{-1}
\hat{\mathbf{h}}_1^{LS}
\right)
$$

令：

$$
\hat{\alpha}_1^{WLS}
=
\frac{
\mathbf{c}_1^H \mathbf{C}_v^{-1}
\hat{\mathbf{h}}_1^{LS}
}{
\mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}
$$

则：

$$
\hat{\mathbf{h}}_1^{LMMSE}
=
\frac{
P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}{
1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}
\mathbf{c}_1 \hat{\alpha}_1^{WLS}
$$

定义收缩因子：

$$
\rho
=
\frac{
P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}{
1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}
\tag{25}
$$

则：

$$
\boxed{
\hat{\mathbf{h}}_1^{LMMSE}
=
\rho \hat{\alpha}_1^{WLS} \mathbf{c}_1
}
\tag{26}
$$

物理含义：

- $\hat{\alpha}_1^{WLS}$ 是白化后的空间匹配投影；
- $\rho$ 是贝叶斯收缩因子，根据先验 SNR 自动调节；
- 如果先验很强，$\rho \to 1$；
- 如果噪声很大，$\rho \to 0$，避免过拟合噪声。

在白噪声情况下：

$$
\mathbf{C}_v = \sigma_v^2 \mathbf{I}_4
$$

则：

$$
\mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
=
\frac{\|\mathbf{c}_1\|^2}{\sigma_v^2}
$$

收缩因子变为：

$$
\rho
=
\frac{
P \frac{\|\mathbf{c}_1\|^2}{\sigma_v^2}
}{
1 + P \frac{\|\mathbf{c}_1\|^2}{\sigma_v^2}
}
=
\frac{
P\|\mathbf{c}_1\|^2
}{
\sigma_v^2 + P\|\mathbf{c}_1\|^2
}
\tag{27}
$$

这就是白噪声下的空间 LMMSE 收缩因子。

---

## 7. 空间签名 $\mathbf{c}_1$ 如何先验获取

若跨独立 TTI 的空间统计稳定，可用独立训练时隙估计候选子空间；必须从观测协方差中扣除已估计的 DMRS LS 噪声协方差，并在独立时隙验证稳定性。每个 TTI 的同槽 PCA 不是长期签名估计，也不能证明跨时隙稳定。训练、选型与最终评估数据必须隔离；单帧 MATLAB 参考不能提供无泄漏的多时隙训练集。

用户 1 的长期信道协方差矩阵为：

$$
\mathbf{R}_{h_1}
=
\mathbb{E}
\left[
\mathbf{h}_1 \mathbf{h}_1^H
\right]
$$

若模型 $\mathbf{h}_1 = \alpha_1 \mathbf{c}_1$ 成立，则：

$$
\mathbf{R}_{h_1}
=
P \mathbf{c}_1 \mathbf{c}_1^H
$$

其特征分解：

$$
\mathbf{R}_{h_1}
=
\sum_{k=1}^{4}
\lambda_k \mathbf{u}_k \mathbf{u}_k^H
$$

在理想秩一模型下，只有一个非零特征值，其对应特征向量即为 $\mathbf{c}_1$：

$$
\lambda_1 = P\|\mathbf{c}_1\|^2,\qquad
\mathbf{u}_1 = \frac{\mathbf{c}_1}{\|\mathbf{c}_1\|}
$$

因此工程上可以：

1. 收集多个 TTI 的 $\hat{\mathbf{h}}_1^{LS}$；
2. 估计 $\mathbf{R}_{h_1}$；
3. 取最大特征值对应特征向量作为空间签名方向；
4. 归一化第一根天线系数为 1，得到 $\mathbf{c}_1$。

如果真实信道不是严格秩一，而是存在多径，则可取前 $r$ 个特征向量构成子空间：

$$
\mathbf{U}_1 =
\begin{bmatrix}
\mathbf{u}_1 & \cdots & \mathbf{u}_r
\end{bmatrix}
$$

然后将 LS 估计投影到该子空间：

$$
\hat{\mathbf{h}}_1^{sub}
=
\mathbf{U}_1 \mathbf{U}_1^H
\hat{\mathbf{h}}_1^{LS}
$$

这就是子空间投影信道估计，是单空间签名方法的推广。

---

## 8. 总结

你提出的方法本质上是：

1. 利用波束泄露结构，把用户 1 的 4 维信道建模为一个慢变空间签名 $\mathbf{c}_1$ 和一个快衰落标量 $\alpha_1$ 的乘积；
2. 用空间匹配投影把四根天线上的信号相干合并，估计出标量 $\alpha_1$；
3. 再用 $\mathbf{c}_1$ 重构完整的 4 维信道；
4. 天线 2/3/4 上的弱泄露信号不仅没有被丢弃，反而被相干合并，提升了估计 SNR；
5. 在白噪声下，主天线 SNR 提升倍数为 $\|\mathbf{c}_1\|^2$，弱天线 SNR 提升倍数为 $\|\mathbf{c}_1\|^2/|c_{1,m}|^2$；
6. 在色噪声/多用户泄露下，可进一步升级为白化投影或空间 LMMSE，获得干扰抑制和贝叶斯收缩的好处。

核心公式总结：

$$
\hat{\mathbf{h}}_1^{LS}
=
\alpha_1 \mathbf{c}_1 + \mathbf{v}
$$

$$
\hat{\alpha}_1
=
\frac{\mathbf{c}_1^H \hat{\mathbf{h}}_1^{LS}}{\|\mathbf{c}_1\|^2}
$$

$$
\hat{\mathbf{h}}_1^{enh}
=
\hat{\alpha}_1 \mathbf{c}_1
$$

$$
\gamma_{m}^{enh}
=
\frac{|\alpha_1|^2 \|\mathbf{c}_1\|^2}{\sigma_v^2}
$$

$$
\hat{\mathbf{h}}_1^{LMMSE}
=
\rho \frac{
\mathbf{c}_1^H \mathbf{C}_v^{-1}
\hat{\mathbf{h}}_1^{LS}
}{
\mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}
\mathbf{c}_1
$$

其中：

$$
\rho
=
\frac{
P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}{
1 + P \mathbf{c}_1^H \mathbf{C}_v^{-1}\mathbf{c}_1
}
$$

上述公式给出的是匹配模型下的理论预测，而不是已测得的通用效果。空间投影是否是有效优化，必须由独立数据上的真实 CSI 误差和配对 BLER/BER、以及高 SNR 安全门槛共同判定；如果空间子空间不稳定、模型偏差占主导、保护逻辑频繁回退或 BLER 置信区间未显示收益，则该链路/信道分布下不能宣称有效。任何结论仅适用于实际测量的配置，不外推为所有波束场景有效或无效。

## 9. 独立空间投影验证与结论

### 9.1 实验协议

历史验证 runner 与空间投影实现已因未测得收益而清理；以下协议和数据仅保留为实验记录，当前仓库不再包含对应模块或命令。

实验配置固定为 4 UE、50 RB、MCS 20、4 RX、CDL-A、soft-mmse-pic（1 轮反馈、阻尼 0.25、20 次 BP）；使用 128 个独立 25-dB 帧训练噪声扣除后的 user/group 空间协方差，64 帧 dev 只按两 SNR 合计 TB 错误、BER、低 rank 与小 group 次序选型；训练集、dev、60-dB 安全集和 holdout 不共享 channel realization。真实 CSI 只用于估计误差 NMSE 与 perfect-CSI 参照，不进入签名训练、保护判定或 dev 选型。

每个帧只生成一次 payload、CDL 信道和 noisy grid，所有估计器在同一观测上配对比较；25/30 dB 使用同一帧 bits/channel 与同一标准化噪声样本。配置中的 split seed 连续，因此 runner 为 train/dev/holdout/high-SNR 依次加固定 1,000,000 seed namespace，再按帧递增；noise seed 与帧 seed 分离。脚本将 payload/channel SHA-256 写入 JSON，并拒绝重复 realization 或 split 泄漏。

dev 排序第一的候选必须在独立 60-dB/64 帧上 CRC 与 payload bits 全零错误才可晋级。holdout 对基线、同槽 PCA、晋级候选与 perfect-CSI 参照先各跑 1,000 帧/SNR；若任一 SNR 基线 TB 错误少于 100，两点均接续扩展到 4,000 帧。报告 TB/CRC 错误、BER、DMRS/data CSI NMSE、接受投影比例、回退原因、每帧×UE 配对结果与运行时间。

### 9.2 预注册有效性判据

仅当 25 和 30 dB 两点 BLER 相对下降都至少 10%、配对帧 bootstrap（10,000 次、seed 20260928）的 Bonferroni 97.5% 双侧 BLER 差值置信区间上界都小于 0（并报告相对风险置信区间）、BER 与 data-symbol NMSE 均不恶化、60-dB 安全门槛通过且 holdout UE×group 投影接受率至少 10%，才结论为“该 CDL-A/50RB/MCS20 分布下空间投影有效”。其余结果结论为“该分布下未验证出有效优化”，并区分退化、近乎全回退与统计证据不足；不推广为其他场景有效或普遍无效。

### 9.3 实测 holdout 结果

完整运行于 CPU 完成，耗时 67,464.83 s（18 h 44 min 25 s）。dev 排序第一并通过 60-dB 安全门槛的候选为 `learned_rank1_group0rb`（rank 1、全带组）；holdout 按预注册规则从 1,000 扩展到 4,000 帧，每个 SNR、每种估计器均有 16,000 TB。表中 NMSE 为 data-symbol CSI NMSE。

| 25 dB 估计器 | TB 错误/16,000 | BLER | BER | data NMSE | 接受比例 |
|---|---:|---:|---:|---:|---:|
| baseline | 6,077 | 0.3798125 | 0.03122339 | 0.001416907 | — |
| same-slot PCA | 6,077 | 0.3798125 | 0.03122339 | 0.001416907 | — |
| learned rank-1 全带 | 6,077 | 0.3798125 | 0.03122339 | 0.001416907 | 0% |
| perfect CSI | 2,887 | 0.1804375 | 0.01297526 | 0 | — |

| 30 dB 估计器 | TB 错误/16,000 | BLER | BER | data NMSE | 接受比例 |
|---|---:|---:|---:|---:|---:|
| baseline | 367 | 0.0229375 | 0.00129762 | 0.000448065 | — |
| same-slot PCA | 367 | 0.0229375 | 0.00129762 | 0.000448065 | — |
| learned rank-1 全带 | 367 | 0.0229375 | 0.00129762 | 0.000448065 | 0% |
| perfect CSI | 91 | 0.0056875 | 0.00026493 | 0 | — |

learned 候选在两个 holdout SNR 的 32,000 个 UE×group 上均因 `model_mismatch` 回退；投影增量为零，因此 BLER、BER、NMSE 与 baseline 完全相同。两点 paired bootstrap 的 BLER 差值 97.5% CI 均为 [0, 0]、candidate/baseline 相对风险 CI 均为 [1, 1]，BLER 相对下降均为 0%。候选通过独立 60-dB/64 帧零 TB/CRC/bit 错误门槛，但该组 256 次投影也全部因 `model_mismatch` 回退；安全门槛通过不代表投影被接受。

rank-1 正校正特征值能量占比在四个 UE 为 56.7%–64.2%；训练半集间 principal-angle cosine 为 0.9893–0.9990，train-to-dev 为 0.9955–0.9971，子空间在这些独立 split 上稳定，但保护器仍判定投影模型不匹配。最终结论：**该 CDL-A/50RB/MCS20 分布下未验证出有效优化，原因是近乎全回退**。结果不外推到其他信道分布。