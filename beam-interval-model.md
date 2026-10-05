Q:更详细的，假如天线系统是一个4波束的接收天线，波束1在主瓣上接收到了对应的波束下的ue1的信号，波束2、3、4也通过旁瓣接收到了ue1的信号，同理，ue2、3、4也会存在一些泄露到其他波束的旁瓣上面，请详细分析这个物理模型。进行详细的数学推导

A:
你描述的是一个典型的**四波束上行同频复用系统**：4 个 UE 的信号在天线处叠加，天线通过 4 组模拟权值形成 4 路输出。每一路都包含本波束 UE 的主瓣信号，以及其他 UE 的旁瓣信号。

**从联合接收的角度看，它是一个 \(4\times4\) 的多用户等效 MIMO 系统。** 对某一路独立接收机而言，旁瓣信号是干扰；对能够联合处理四路输出的接收机而言，它们也是可以用于检测的观测。

下面先假设四个 UE **同时、同频、同极化发送独立数据流**，从物理传播一直推导到基带检测。若 UE 使用正交时频资源，下面的空间耦合仍然存在，但理想同步条件下不会形成同资源上的多用户干扰。

**1．物理架构：四个波束是四组空间滤波器**

先以“同一个 \(N\) 阵元阵列，同时形成四个模拟接收波束”为例。

定义：

\[
\mathbf W=
\begin{bmatrix}
\mathbf w_1&\mathbf w_2&\mathbf w_3&\mathbf w_4
\end{bmatrix}
\in\mathbb C^{N\times4}
\]

其中 \(\mathbf w_i\) 是第 \(i\) 个模拟波束的权值。

如果阵元接收信号为 \(\mathbf x(t)\)，四路输出就是：

\[
\boxed{
\mathbf y(t)=\mathbf W^H\mathbf x(t)
}
\]

即：

\[
y_i(t)=\mathbf w_i^H\mathbf x(t)
\]

四波束同时接收，需要能够同时实现四组模拟组合并取得四路输出；只有一组权值、一个输出端口，通过切换方向扫描，并不等价于这里的四路同时观测。模拟多波束通常需要增加每个波束对应的相位调整通道。[Analog Devices](https://www.analog.com/en/resources/analog-dialogue/articles/phased-array-beamforming-ics-simplify-antenna-design.html?utm_source=chatgpt.com)

还要明确：**UE1 并没有先进入波束1，再从波束1“漏”到波束2。**

更准确的过程是：

> UE1 的电磁波到达整个阵列；四组模拟权值分别对同一组阵元信号进行组合，于是四个输出都可能包含 UE1。

这属于空间响应重叠。射频电路之间的串扰是另一种效应，可以额外建模。

---

**2．从传播距离推导阵元接收信号**

设接收阵列的参考点为原点，第 \(n\) 个阵元的位置为：

\[
\mathbf r_n\in\mathbb R^3
\]

第 \(j\) 个 UE 相对于参考点的位置为：

\[
\mathbf p_j=R_j\mathbf u_j
\]

其中 \(R_j\) 是距离，\(\mathbf u_j\) 是从阵列指向 UE 的单位方向向量。

UE \(j\) 到阵元 \(n\) 的距离为：

\[
R_{n,j}=\|\mathbf p_j-\mathbf r_n\|
\]

在远场条件下：

\[
R_{n,j}
\approx
R_j-\mathbf u_j^T\mathbf r_n
\]

所以传播时延为：

\[
\tau_{n,j}
\approx
\tau_j-\frac{\mathbf u_j^T\mathbf r_n}{c}
,\qquad
\tau_j=\frac{R_j}{c}
\]

令 UE \(j\) 发送的复基带信号为 \(\sqrt{P_j}s_j(t)\)，满足：

\[
\mathbb E[|s_j(t)|^2]=1
\]

忽略阵元间路径损耗的微小差别，阵元 \(n\) 收到：

\[
x_{n,j}(t)
=
\sqrt{P_j}\beta_j
s_j(t-\tau_{n,j})
e^{-j2\pi f_c\tau_{n,j}}
\]

对于相对于阵列时延跨度足够窄的信号：

\[
s_j(t-\tau_{n,j})\approx s_j(t-\tau_j)
\]

于是：

\[
x_{n,j}(t)
\approx
\sqrt{P_j}\beta_j
e^{-j2\pi f_c\tau_j}
e^{j\frac{2\pi}{\lambda}\mathbf u_j^T\mathbf r_n}
s_j(t-\tau_j)
\]

定义传播系数：

\[
\alpha_j=\beta_j e^{-j2\pi f_c\tau_j}
\]

以及阵列流形：

\[
\mathbf a(\mathbf u_j)
=
\begin{bmatrix}
e^{j\frac{2\pi}{\lambda}\mathbf u_j^T\mathbf r_1}\\
\vdots\\
e^{j\frac{2\pi}{\lambda}\mathbf u_j^T\mathbf r_N}
\end{bmatrix}
\]

就得到：

\[
\mathbf x_j(t)
=
\sqrt{P_j}\alpha_j
\mathbf a(\mathbf u_j)s_j(t-\tau_j)
\]

四个 UE 的信号根据电磁场的线性叠加原理相加：

\[
\boxed{
\mathbf x(t)=
\sum_{j=1}^{4}
\sqrt{P_j}\alpha_j
\mathbf a(\mathbf u_j)s_j(t-\tau_j)
+\mathbf n(t)
}
\]

这里最重要的是：**阵元处先叠加所有 UE，模拟波束形成再对总信号进行空间滤波。**

---

**3．推导四波束耦合矩阵**

第 \(i\) 个波束的输出：

\[
\begin{aligned}
y_i(t)
&=\mathbf w_i^H\mathbf x(t)\\
&=
\sum_{j=1}^{4}
\sqrt{P_j}\alpha_j
\underbrace{\mathbf w_i^H\mathbf a(\mathbf u_j)}_{b_{ij}}
s_j(t-\tau_j)
+\eta_i(t)
\end{aligned}
\]

定义：

\[
\boxed{
b_{ij}=\mathbf w_i^H\mathbf a(\mathbf u_j)
}
\]

其物理意义是：

> 第 \(i\) 个接收波束，对 UE \(j\) 所在方向的复数响应。

注意下标：**行是接收波束，列是 UE。**

在完成时延处理后的窄带离散模型中：

\[
\mathbf y=\mathbf H\mathbf P^{1/2}\mathbf s+\boldsymbol\eta
\]

其中：

\[
\mathbf P=\operatorname{diag}(P_1,P_2,P_3,P_4)
\]

\[
\boxed{
\mathbf H=
\begin{bmatrix}
\alpha_1b_{11}&\alpha_2b_{12}&\alpha_3b_{13}&\alpha_4b_{14}\\
\alpha_1b_{21}&\alpha_2b_{22}&\alpha_3b_{23}&\alpha_4b_{24}\\
\alpha_1b_{31}&\alpha_2b_{32}&\alpha_3b_{33}&\alpha_4b_{34}\\
\alpha_1b_{41}&\alpha_2b_{42}&\alpha_3b_{43}&\alpha_4b_{44}
\end{bmatrix}
}
\]

按照题设：

- \(b_{11},b_{22},b_{33},b_{44}\)：对应主瓣响应；
- \(b_{ij},i\neq j\)：对应旁瓣响应。

例如第一路：

\[
\boxed{
y_1=
\underbrace{\sqrt{P_1}\alpha_1b_{11}s_1}_{\text{UE1 的有用信号}}
+
\underbrace{
\sqrt{P_2}\alpha_2b_{12}s_2+
\sqrt{P_3}\alpha_3b_{13}s_3+
\sqrt{P_4}\alpha_4b_{14}s_4
}_{\text{其他 UE 的旁瓣干扰}}
+\eta_1
}
\]

混合波束形成中的“阵元信号经过模拟组合，再进行数字处理”也采用这种线性矩阵建模方式。[MATLAB &amp; Simulink](https://www.mathworks.com/help/phased/examples/introduction-to-hybrid-beamforming.html?utm_source=chatgpt.com)

**同一列中的四个分量，携带的是同一个 UE 的同一份数据。**

例如 UE1 对四路输出的贡献为：

\[
\mathbf y^{(1)}
=
\sqrt{P_1}\alpha_1
\begin{bmatrix}
b_{11}\\b_{21}\\b_{31}\\b_{41}
\end{bmatrix}s_1
\]

它们是相干的信号副本，而不是四份独立随机数据。

---

**4．耦合系数如何由阵列几何决定？**

以 \(N\) 阵元均匀线阵为例：

\[
\mathbf a(\theta)=
\begin{bmatrix}
1&e^{jkd\sin\theta}&\cdots&e^{j(N-1)kd\sin\theta}
\end{bmatrix}^{T}
\]

设第 \(i\) 个波束指向 \(\vartheta_i\)，采用均匀加权：

\[
\mathbf w_i=\frac{\mathbf a(\vartheta_i)}{\sqrt N}
\]

则：

\[
b_{ij}
=
\frac{1}{\sqrt N}
\sum_{n=0}^{N-1}
e^{jnkd(\sin\theta_j-\sin\vartheta_i)}
\]

令：

\[
\psi_{ij}=kd(\sin\theta_j-\sin\vartheta_i)
\]

得到：

\[
\boxed{
b_{ij}
=
\frac{1}{\sqrt N}
e^{j(N-1)\psi_{ij}/2}
\frac{\sin(N\psi_{ij}/2)}{\sin(\psi_{ij}/2)}
}
\]

如果每个波束精确指向自己的 UE：

\[
\vartheta_i=\theta_i
\]

则：

\[
b_{ii}=\sqrt N
\]

定义归一化空间耦合：

\[
\rho_{ij}=\frac{b_{ij}}{\sqrt N}
\]

有：

\[
\rho_{ii}=1
\]

\[
\boxed{
\rho_{ij}
=
\frac{1}{N}
e^{j(N-1)\psi_{ij}/2}
\frac{\sin(N\psi_{ij}/2)}{\sin(\psi_{ij}/2)}
}
\]

相对主瓣峰值的功率泄露为：

\[
\boxed{
L_{ij}=|\rho_{ij}|^2
}
\]

因此，“旁瓣泄露为 \(-20\ \mathrm{dB}\)”意味着：

\[
L_{ij}=10^{-2}
,\qquad
|\rho_{ij}|=10^{-1}
\]

**构造基带信道矩阵时，要使用幅度系数 \(0.1e^{j\phi_{ij}}\)，不能使用功率系数 \(0.01\)。**

另外，只有功率方向图不能唯一确定 \(\mathbf H\)：联合检测还需要复数相位。

---

**5．一种重要的特殊情况：耦合矩阵是 Gram 矩阵**

令：

\[
\mathbf A=
\begin{bmatrix}
\mathbf a(\theta_1)&\mathbf a(\theta_2)&
\mathbf a(\theta_3)&\mathbf a(\theta_4)
\end{bmatrix}
\]

当各模拟波束恰好匹配各 UE 的方向时：

\[
\mathbf W=\frac{\mathbf A}{\sqrt N}
\]

所以：

\[
\mathbf B=\mathbf W^H\mathbf A
=
\frac{1}{\sqrt N}\mathbf A^H\mathbf A
\]

定义：

\[
\boxed{
\mathbf G=\frac{1}{N}\mathbf A^H\mathbf A
}
\]

则：

\[
\mathbf B=\sqrt N\,\mathbf G
\]

且：

\[
G_{ii}=1,\qquad G_{ij}=\rho_{ij}
\]

这个结果表明：

> 主瓣和旁瓣之间的耦合，本质上是不同 UE 的空间导向向量之间的内积。

在这个特殊模型中：

\[
\rho_{ji}=\rho_{ij}^{*}
\]

因此：

\[
|\rho_{ji}|^2=|\rho_{ij}|^2
\]

但实际接收功率通常并不对称，因为：

\[
P_{i\leftarrow j}=P_j|\alpha_j|^2|b_{ij}|^2
\]

而：

\[
P_{j\leftarrow i}=P_i|\alpha_i|^2|b_{ji}|^2
\]

两者还受到 UE 发射功率、传播损耗等因素影响。

如果四个波束使用不同加权、没有精确指向 UE，或者来自不同天线，空间耦合本身也不必对称。

---

**6．四路输出的噪声可能相关**

对共用阵元的系统，假设阵元噪声独立：

\[
\mathbb E[\mathbf n\mathbf n^H]=\sigma_a^2\mathbf I
\]

再考虑四路组合器后各自引入的噪声 \(\mathbf v\)，则：

\[
\boldsymbol\eta=\mathbf W^H\mathbf n+\mathbf v
\]

若二者独立：

\[
\boxed{
\mathbf R_\eta
=
\sigma_a^2\mathbf W^H\mathbf W+\mathbf R_v
}
\]

因此：

\[
[\mathbf R_\eta]_{i\ell}
=
\sigma_a^2\mathbf w_i^H\mathbf w_\ell
+
[\mathbf R_v]_{i\ell}
\]

只要权值不正交，来自共用阵元的噪声就可能在不同波束输出之间相关。

对于前面的匹配波束：

\[
\mathbf W^H\mathbf W=\mathbf G
\]

所以：

\[
\boxed{
\mathbf R_\eta=\sigma_a^2\mathbf G+\mathbf R_v
}
\]

这意味着：**空间耦合矩阵和输出噪声协方差，并不总能独立任意指定。**

这也是共用阵列与四副独立接收天线的重要区别：

| 架构 | 等效信号模型 | 接收机噪声特点 |
|---|---|---|
| 共用阵列形成四波束 | \(\mathbf H=\mathbf W^H\mathbf A\operatorname{diag}(\alpha_j)\) | 共用阵元噪声可能相关 |
| 四副独立天线各形成一波束 | 一般的 \(4\times4\) 复信道 | 独立接收机热噪声通常可近似不相关 |

实际模拟网络的分路损耗、增益及噪声需要并入权值与噪声模型。单位范数权值只是分析归一化，并不代表无损地复制四份物理功率。

---

**7．如果每个波束独立解调，会发生什么？**

假设波束 \(i\) 只检测 UE \(i\)，其他 UE 被当作干扰。

定义：

\[
h_{ij}=\alpha_jb_{ij}
\]

则：

\[
\boxed{
\mathrm{SINR}_i
=
\frac{P_i|h_{ii}|^2}
{\displaystyle\sum_{j\neq i}P_j|h_{ij}|^2+
[\mathbf R_\eta]_{ii}}
}
\]

这个功率相加公式使用了不同 UE 数据独立、零均值的假设：

\[
\mathbb E[s_js_\ell^*]=0,\qquad j\neq\ell
\]

瞬时波形依然按复数幅度相加；只有求平均功率时，交叉项才消失。

若四个 UE 在各自主瓣输出上的期望接收功率相同，记为 \(S\)，所有交叉功率泄露均为 \(L\)，每路噪声为 \(N_0\)，那么：

\[
\mathrm{SINR}=\frac{S}{3LS+N_0}
\]

定义没有多用户干扰时的：

\[
\mathrm{SNR}_0=\frac{S}{N_0}
\]

得到：

\[
\boxed{
\mathrm{SINR}
=
\frac{1}{3L+1/\mathrm{SNR}_0}
}
\]

例如：

\[
L=0.01\quad(-20\ \mathrm{dB})
\]

\[
\mathrm{SNR}_0=100\quad(20\ \mathrm{dB})
\]

则：

\[
\mathrm{SINR}=\frac{1}{0.03+0.01}=25
\]

即：

\[
\mathrm{SINR}\approx13.98\ \mathrm{dB}
\]

即使继续提高所有 UE 的功率：

\[
\lim_{\mathrm{SNR}_0\to\infty}\mathrm{SINR}
=
\frac{1}{3L}
\]

此例的上限是：

\[
15.23\ \mathrm{dB}
\]

因此，**所有 UE 一起加功率无法消除由旁瓣耦合形成的干扰上限。**

如果存在近远效应，问题更明显：一个在阵列处强 \(30\ \mathrm{dB}\) 的 UE，即使经过 \(-20\ \mathrm{dB}\) 旁瓣抑制，仍可能比弱 UE 的主瓣信号强 \(10\ \mathrm{dB}\)。

---

**8．四路干扰是相关的，不能独立生成**

令 \(\mathbf h_j\) 表示 \(\mathbf H\) 的第 \(j\) 列：

\[
\mathbf h_j=
\begin{bmatrix}
h_{1j}&h_{2j}&h_{3j}&h_{4j}
\end{bmatrix}^{T}
\]

UE \(j\) 在四路中的信号协方差为：

\[
\boxed{
\mathbf R_j=P_j\mathbf h_j\mathbf h_j^H
}
\]

对于单数据流 UE，这个矩阵的秩最多为 1。

不同输出之间的相关项是：

\[
[\mathbf R_j]_{i\ell}
=
P_jh_{ij}h_{\ell j}^{*}
\]

所以对 UE1 来说，其他三个 UE 形成的四路干扰协方差为：

\[
\boxed{
\mathbf R_{I,1}
=
\sum_{j=2}^{4}P_j\mathbf h_j\mathbf h_j^H
}
\]

它一般不是对角矩阵。

**仿真时必须把同一个 UE 的同一份波形，分别乘以四个信道系数，再送入四路输出。** 如果在四路上分别生成互相独立的“UE2 干扰”，就破坏了真实的空间相关结构，联合检测性能也会失真。

---

**9．联合检测：旁瓣副本可以参与空间分离**

定义包含发送功率的等效矩阵：

\[
\mathbf F=\mathbf H\mathbf P^{1/2}
\]

于是：

\[
\boxed{
\mathbf y=\mathbf F\mathbf s+\boldsymbol\eta
}
\]

令 \(\mathbf f_i\) 为 \(\mathbf F\) 的第 \(i\) 列。用四路输出检测 UE \(i\)：

\[
\hat s_i=\mathbf v_i^H\mathbf y
\]

展开：

\[
\hat s_i
=
\mathbf v_i^H\mathbf f_i s_i
+
\sum_{j\neq i}\mathbf v_i^H\mathbf f_j s_j
+
\mathbf v_i^H\boldsymbol\eta
\]

对应：

\[
\boxed{
\mathrm{SINR}_i(\mathbf v_i)
=
\frac{|\mathbf v_i^H\mathbf f_i|^2}
{
\sum_{j\neq i}|\mathbf v_i^H\mathbf f_j|^2+
\mathbf v_i^H\mathbf R_\eta\mathbf v_i
}
}
\]

独立波束解调只是一个特殊选择：

\[
\mathbf v_i=\mathbf e_i
\]

即只取第 \(i\) 个输出。

联合检测允许选择更合适的 \(\mathbf v_i\)，既利用 UE \(i\) 在其他输出中的副本，也抑制其他 UE。

**零迫检测 ZF**

若 \(\mathbf F\) 满秩：

\[
\hat{\mathbf s}=\mathbf F^{-1}\mathbf y
=
\mathbf s+\mathbf F^{-1}\boldsymbol\eta
\]

理想信道估计下，多用户干扰被消除，但输出噪声为：

\[
\boxed{
\mathbf R_{\mathrm{ZF}}
=
\mathbf F^{-1}\mathbf R_\eta\mathbf F^{-H}
}
\]

若噪声为 \(\sigma^2\mathbf I\)，则：

\[
\mathbf R_{\mathrm{ZF}}
=
\sigma^2(\mathbf F^H\mathbf F)^{-1}
\]

矩阵接近奇异时，求逆会显著放大噪声。

**LMMSE 检测**

在：

\[
\mathbb E[\mathbf s\mathbf s^H]=\mathbf I
\]

条件下，最小化：

\[
\mathbb E[\|\mathbf s-\mathbf K\mathbf y\|^2]
\]

得到：

\[
\boxed{
\mathbf K_{\mathrm{MMSE}}
=
\mathbf F^H
(\mathbf F\mathbf F^H+\mathbf R_\eta)^{-1}
}
\]

等价地，当 \(\mathbf R_\eta\) 正定时：

\[
\boxed{
\mathbf K_{\mathrm{MMSE}}
=
(\mathbf I+\mathbf F^H\mathbf R_\eta^{-1}\mathbf F)^{-1}
\mathbf F^H\mathbf R_\eta^{-1}
}
\]

单个 UE 的最大 SINR 合并方向为：

\[
\boxed{
\mathbf v_i\propto
\left(
\sum_{j\neq i}\mathbf f_j\mathbf f_j^H+\mathbf R_\eta
\right)^{-1}\mathbf f_i
}
\]

这些公式说明，联合处理需要的不只是各路功率，还包括信道相位和噪声相关性。

---

**10．四波束是否一定能分开四个 UE？**

不一定。需要考察：

\[
\operatorname{rank}(\mathbf F)
\]

以及噪声白化后矩阵：

\[
\widetilde{\mathbf F}=\mathbf R_\eta^{-1/2}\mathbf F
\]

的奇异值。

若：

\[
\sigma_{\min}(\widetilde{\mathbf F})\approx0
\]

则至少有一组用户信号在四路观测中很难区分。

例如两个 UE 的入射方向几乎相同：

\[
\mathbf a(\theta_1)\approx\mathbf a(\theta_2)
\]

则：

\[
\mathbf W^H\mathbf a(\theta_1)
\approx
\mathbf W^H\mathbf a(\theta_2)
\]

它们在四路输出中的空间特征近似平行。即使有四个波束，也不能凭空创造空间分辨能力。

对于均匀线阵，第一零陷的方向余弦间隔尺度为：

\[
|\Delta\sin\theta|\approx\frac{\lambda}{Nd}
\]

这给出了区分相邻方向的孔径尺度，但实际可分离性应直接检查矩阵，而不是只按角度阈值判断。

还有一个容易误解的地方：**共用阵列的四路输出不一定带来四份独立接收增益。**

在理想匹配波束模型中：

\[
\mathbf W=\frac{\mathbf A}{\sqrt N},
\qquad
\mathbf R_\eta=\sigma_a^2\mathbf G
\]

假设 \(\mathbf G\) 可逆且忽略后级噪声，令：

\[
\mathbf D=\operatorname{diag}
(\alpha_1\sqrt{P_1},\ldots,\alpha_4\sqrt{P_4})
\]

则：

\[
\mathbf F=\sqrt N\,\mathbf G\mathbf D
\]

因此：

\[
\begin{aligned}
\mathbf F^H\mathbf R_\eta^{-1}\mathbf F
&=
\frac{N}{\sigma_a^2}\mathbf D^H\mathbf G\mathbf D\\
&=
\frac{1}{\sigma_a^2}\mathbf D^H\mathbf A^H\mathbf A\mathbf D
\end{aligned}
\]

右侧正是原始阵元观测对应的信息矩阵。

这说明：在这些理想假设下，四个匹配波束保留了检测所需的信号子空间信息；它们没有额外创造阵列之外的能量或独立观测。

---

**11．卫星场景下的时延、多普勒与 OFDM 扩展**

对同一 UE，在共用孔径模型下，四路主要共享同一个传播时延和传播多普勒：

\[
y_i(t)
=
\sum_{j=1}^{4}
\sqrt{P_j}\beta_j(t)b_{ij}(t)
s_j(t-\tau_j(t))
e^{-j2\pi f_c\tau_j(t)}
+\eta_i(t)
\]

UE \(j\) 的传播多普勒为：

\[
\nu_j(t)=-f_c\dot\tau_j(t)
\]

因此：

- UE1 在波束1、2、3、4中的副本，主要具有相同的传播多普勒；
- UE1 与 UE2 的多普勒可以不同；
- 不同波束的时变权值和射频链路，还可能引入额外的相位变化。

对于宽带移相器阵列：

\[
b_{ij}(f,t)=\mathbf w_i^H(t)\mathbf a(\mathbf u_j(t),f)
\]

所以耦合矩阵还会随子载波变化。

在各 UE 残余频偏足够小、时延满足 OFDM 处理条件、符号内时变可忽略时，每个 RE 可写成：

\[
\boxed{
\mathbf Y[k,m]
=
\mathbf H[k,m]\mathbf P^{1/2}\mathbf S[k,m]
+\boldsymbol\eta[k,m]
}
\]

其中：

\[
H_{ij}[k,m]
=
\alpha_j[k,m]b_{ij}(f_k,t_m)
\]

若各 UE 的残余多普勒不能忽略，则一个 RE 不再只受到同一子载波上四个符号的影响，而是：

\[
Y_i[k,m]
=
\sum_{j=1}^{4}\sum_q
\sqrt{P_j}\,
\mathcal H_{ij}[k,q,m]S_j[q,m]
+\eta_i[k,m]
\]

其中 \(q\neq k\) 的项表示 ICI。

**对 UE1 补偿频偏后，其他 UE 的旁瓣干扰仍可能带有相对于 UE1 的残余频偏。** 此时仅在每个子载波上做静态 \(4\times4\) 检测，未必足以消除干扰。

---

**12．实际仿真应保留哪些量？**

针对你描述的系统，建议将最小完整模型写成：

\[
\boxed{
\begin{aligned}
\mathbf x&=\mathbf A\mathbf D\mathbf s+\mathbf n\\
\mathbf y&=\mathbf W^H\mathbf x+\mathbf v\\
\mathbf F&=\mathbf W^H\mathbf A\mathbf D\\
\mathbf R_\eta&=\mathbf W^H\mathbf R_n\mathbf W+\mathbf R_v
\end{aligned}
}
\]

然后比较独立解调与联合检测。

| 建模量 | 必须保留的内容 |
|---|---|
| UE 信号 | 每个 UE 一份波形，映射到四路输出 |
| 空间响应 | 16 个复数耦合系数，含幅度和相位 |
| 传播信道 | 各 UE 的路径损耗、时延、多普勒 |
| 接收噪声 | 共用阵元噪声及后级噪声的协方差 |
| 宽带效应 | 按需引入 \(b_{ij}(f,t)\) |
| 信道估计 | 能区分各 UE 的导频，估计完整矩阵 |

例如导频观测：

\[
\mathbf Y_p=\mathbf F\mathbf S_p+\mathbf N_p
\]

若：

\[
\mathbf S_p\mathbf S_p^H=T\mathbf I
\]

则 LS 估计为：

\[
\hat{\mathbf F}=\frac{1}{T}\mathbf Y_p\mathbf S_p^H
\]

如果四个 UE 在同一资源上发送完全相同的导频，只能直接观测到它们信道列的叠加，通常无法从这些导频中辨识完整的 \(4\times4\) 矩阵。

你的系统里，真正需要评估的既包括**每路主瓣与旁瓣功率比**，也包括**完整复信道矩阵的可分离性**：前者决定独立波束接收的干扰水平，后者决定四路联合处理能够恢复多少性能。

Q:我想通过sionna-rt进行上面的多波束干扰建模，模拟上行基站接收机同时接收4波束的MUMIMO，并且波束之间存在一定的干扰，如前面对话的干扰模型，我应该怎么做？给我一个详细的执行清单。

A:
可以实现。建议采用 **Sionna RT 生成阵元级传播信道，再自行实现四路模拟波束合并，最后接入 Sionna PHY 的 MU-MIMO 接收机**。

你的核心模型应当是：

\[
\boxed{
\mathbf y[k,m]
=
\underbrace{\mathbf W_{\mathrm{RF}}^H\mathbf H_{\mathrm{ant}}[k,m]\mathbf P^{1/2}}_{\mathbf F[k,m]\in\mathbb C^{4\times4}}
\mathbf s[k,m]+\boldsymbol\eta[k,m]
}
\]

其中：

- \(\mathbf H_{\mathrm{ant}}\in\mathbb C^{N\times4}\)：四个 UE 到 \(N\) 个接收阵元的传播信道，由 RT 生成；
- \(\mathbf W_{\mathrm{RF}}\in\mathbb C^{N\times4}\)：四个模拟波束的权值，由你设计；
- \(\mathbf F\in\mathbb C^{4\times4}\)：包含发送功率的波束域等效信道；
- \(\mathbf F\) 的对角元素对应期望接收，非对角元素对应其他 UE 的空间泄露。

**RT 负责传播，模拟合并负责产生主瓣与旁瓣，PHY 负责发送波形、信道估计和多用户检测。** 不需要让 RT 额外生成一种名为“旁瓣干扰”的随机信号。

下面是一份按依赖关系排列的执行清单。默认采用**同一个阵列、四个模拟输出、四个单天线 UE、每 UE 一流、上行同频复用**。

---

1. **固定软件版本，先跑通官方最小示例**

   - [ ] 创建独立 Python 环境。
   - [ ] 安装完整 Sionna，包含 RT 和 PHY。
   - [ ] 跑通一次 RT 路径计算。
   - [ ] 跑通一次 PHY 的 QPSK/AWGN 链路。
   - [ ] 保存 Python、Sionna、Sionna RT、PyTorch、Mitsuba、Dr.Jit 和 GPU 驱动版本。

   我查到的当前官方文档为 **Sionna 2.2.0**，PHY 使用 PyTorch；安装文档要求 Python 3.11+、PyTorch 2.9+，推荐 Ubuntu 24.04。RT 使用 Mitsuba/Dr.Jit，CPU 后端还需要 LLVM。旧版 TensorFlow 示例不能直接与当前 PyTorch 接口混用。[Sionna 2.2.0](https://nvlabs.github.io/sionna/installation.html?utm_source=chatgpt.com)


   GPU 版 PyTorch 按机器的 CUDA/驱动条件安装。

   **完成标准：** RT 能返回复数信道，PHY 能完成一次正确解调。此时再搭建多波束系统。

2. **确定第一版仿真配置与张量接口**

   第一版采用静态、LoS、理想同步、完美 CSI，集中验证空间耦合。

   | 项目 | 第一版建议 |
   |---|---|
   | UE 数量 | 4 |
   | 每 UE 天线/流数 | 1 / 1 |
   | 接收阵列 | \(8\times8\)，64 个单极化阵元 |
   | 模拟输出数量 | 4 |
   | 阵元间距 | \(0.5\lambda\) |
   | 阵元方向图 | 初期 `"iso"`，随后换实际方向图 |
   | 载频 | 例如 3.5 GHz，先验证模型 |
   | OFDM | FFT 512、SCS 30 kHz、14 符号 |
   | 调制 | QPSK，随后 16QAM |
   | 数据资源 | 四 UE 使用相同数据 RE |
   | 同步 | 理想定时、零残余频偏 |
   | 信道 | 静态 LoS |
   | 接收机 | 独立波束、ZF、联合 LMMSE |

   建议在自己的代码中统一为：

   | 变量 | 形状 |
   |---|---|
   | 阵元域信道 `H_ant` | `[M, K, N, 4]` |
   | 模拟权值 `W` | `[N, 4]` |
   | 波束域信道 `H_beam` | `[M, K, 4, 4]` |
   | 单位功率发送符号 `S` | `[B, M, K, 4]` |
   | 包含功率的信道 `F` | `[M, K, 4, 4]` |
   | 四路接收信号 `Y` | `[B, M, K, 4]` |
   | 输出噪声协方差 `R_eta` | `[4, 4]`，或随 RE 变化 |

   这里 \(B\) 是蒙特卡洛批次，\(M\) 是符号数，\(K\) 是子载波数。

   **不要把四个波束定义成四个独立数据接收节点。** 联合检测时，它们是同一个接收机的四个观测端口。

3. **创建“4 个发射机 + 1 个接收阵列”的 RT 场景**

   - [ ] 添加四个 `Transmitter`，对应 UE1～UE4。
   - [ ] 添加一个 `Receiver`，对应基站或卫星接收阵列。
   - [ ] `scene.tx_array` 配成单天线。
   - [ ] `scene.rx_array` 配成 \(8\times8\) 单极化阵列。
   - [ ] 第一版使用空场景，仅计算直达路径。
   - [ ] 固定 UE 名称与信道列顺序，记录到元数据。

   下面是按当前接口编写的场景骨架，**尚未在你的机器上运行验证**：

   ```python
   import numpy as np
   from sionna.rt import (
       load_scene, PlanarArray, Transmitter, Receiver, PathSolver
   )

   scene = load_scene()
   scene.frequency = 3.5e9

   scene.tx_array = PlanarArray(
       num_rows=1, num_cols=1,
       pattern="iso", polarization="V"
   )

   scene.rx_array = PlanarArray(
       num_rows=8, num_cols=8,
       vertical_spacing=0.5,
       horizontal_spacing=0.5,
       pattern="iso", polarization="V"
   )

   bs_pos = np.array([0.0, 0.0, 25.0])
   bs = Receiver(name="bs", position=bs_pos.tolist())
   scene.add(bs)

   # 阵列默认朝向保持不变；UE 分布在其正前方扇区
   azimuths = np.deg2rad([-40.0, -15.0, 15.0, 40.0])
   ue_positions = []

   for j, az in enumerate(azimuths):
       pos = np.array([
           200.0*np.cos(az),
           200.0*np.sin(az),
           1.5
       ])
       ue_positions.append(pos)
       scene.add(Transmitter(
           name=f"ue{j+1}", position=pos.tolist()
       ))

   solver = PathSolver(deterministic=True)
   paths = solver(
       scene=scene,
       max_depth=0,
       los=True,
       specular_reflection=False,
       diffuse_reflection=False,
       refraction=False,
       synthetic_array=True,
       seed=123
   )
   ```

   当前 RT 的 `synthetic_array=True` 使用阵列中心求路径，再按平面波假设补充阵元空间相位；`False` 则显式求各天线对之间的路径。第一版远场窄带模型适合从前者开始。[Sionna 2.2.0](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html?utm_source=chatgpt.com)

   **完成标准：** 四个 UE 均有有效 LoS 路径，接收天线端口数为 64，而非 4。

4. **导出阵元级 CFR，保留路径损耗与复数相位**

   - [ ] 使用 `Paths.cfr()`，传入**基带子载波频率偏移**。
   - [ ] 设置 `normalize=False`。
   - [ ] 明确时延是否已经通过理想上行定时提前量补偿。
   - [ ] 检查原始输出维度，再转成统一接口。
   - [ ] 缓存信道，后续 BER 仿真重复使用，避免每个码块都重新追踪。

   示例：

   ```python
   fft_size = 512
   scs = 30e3
   freqs = (np.arange(fft_size) - fft_size//2) * scs

   h_rt = paths.cfr(
       frequencies=freqs,
       num_time_steps=1,
       normalize=False,
       normalize_delays=True,
       out_type="numpy"
   )

   # 当前接口：
   # [num_rx, num_rx_ant, num_tx, num_tx_ant, time, frequency]
   assert h_rt.shape[:4] == (1, 64, 4, 1)

   h0 = h_rt[0, :, :, 0, :, :]       # [N, UE, time, frequency]
   H_ant = np.transpose(h0, (2, 3, 0, 1))
   # [time, frequency, N, UE]
   ```

   `normalize=False` 保留原始幅度尺度；`normalize_delays=True` 将各链路首径对齐，不能把它当成已经模拟了实际定时控制过程。上述设置用于第一版理想同步基线。若研究 UE 间残余时差，应保留原始时延并显式施加各 UE 的定时提前量。`Paths.cfr()` 的频率响应、时延与归一化选项见官方文档。[Sionna 2.2.0](https://nvlabs.github.io/sionna/rt/api/paths.html?utm_source=chatgpt.com)

   **不要在得到四波束信道后，将 16 条“UE→波束”链路分别归一化到单位功率。** 这会抹掉主瓣与旁瓣的幅度差异。

5. **设计四个模拟波束，并验证方向、相位符号和阵元顺序**

   - [ ] 获取接收阵元的实际位置。
   - [ ] 将阵元位置和 UE 方向放到同一坐标系。
   - [ ] 用几何方向构造四组相位权值。
   - [ ] 扫描方向图，检查四个主瓣是否指向预期 UE。
   - [ ] 验证其余 UE 是否位于旁瓣区；主瓣重叠也会产生干扰，但不应称为旁瓣泄露。

   对阵元位置矩阵：

   \[
   \mathbf R=
   \begin{bmatrix}
   \mathbf r_1^T\\ \vdots\\ \mathbf r_N^T
   \end{bmatrix}
   \]

   以及指向 UE \(i\) 的单位方向：

   \[
   \mathbf u_i=
   \frac{\mathbf p_i-\mathbf p_{\mathrm{BS}}}
   {\|\mathbf p_i-\mathbf p_{\mathrm{BS}}\|}
   \]

   导向向量为：

   \[
   a_n(\mathbf u_i)=
   e^{j\frac{2\pi}{\lambda}\mathbf r_n^T\mathbf u_i}
   \]

   相位型均匀权值：

   \[
   \boxed{
   \mathbf w_i=\frac{\mathbf a(\mathbf u_i)}{\sqrt N}
   }
   \]

   因为采用的是：

   \[
   y_i=\mathbf w_i^H\mathbf x
   \]

   所以构造 \(\mathbf w_i\) 后，合并时还要取共轭。

   ```python
   # R_world: [N, 3]，相对于接收参考点的阵元位置，单位米
   # U: [3, 4]，四个波束的指向单位向量
   wavelength = 299792458.0 / scene.frequency

   A = np.exp(1j * 2*np.pi/wavelength * (R_world @ U))
   W = A / np.sqrt(A.shape[0])
   ```

   Sionna `PlanarArray` 默认位于局部 \(y-z\) 平面，阵元采用列优先编号，间距以波长为单位。直接假定它位于 \(x-y\) 平面或使用错误的展平顺序，会导致权值与信道错位。应通过 `positions()` 或 `rotate()` 获取对应位置。[Sionna 2.2.0](https://nvlabs.github.io/sionna/rt/api/antenna_array.html?utm_source=chatgpt.com)

   一个很实用的 LoS 校验是：

   \[
   \mathbf w_j^{\mathrm{check}}
   =
   \frac{\mathbf h_j}{\|\mathbf h_j\|}
   \]

   几何波束与这个理想匹配方向应得到接近的主瓣接收功率。这只用于校验；多径场景中的完美信道匹配权值，不应直接替代实际固定模拟波束。

6. **计算波束域信道，量化真实泄露**

   - [ ] 用同一组权值作用于所有 UE 的阵元信道。
   - [ ] 保存完整 \(4\times4\) 复矩阵。
   - [ ] 分别记录空间泄露和实际干扰功率。
   - [ ] 检查矩阵秩、奇异值及用户空间相关性。

   核心代码：

   ```python
   # W: [N, beam]
   # H_ant: [M, K, N, UE]
   H_beam = np.einsum(
       "ni,mknj->mkij",
       W.conj(), H_ant
   )
   # [M, K, beam, UE]
   ```

   定义 UE \(j\) 从自己的波束泄露到波束 \(i\) 的功率比：

   \[
   \boxed{
   L_{i\leftarrow j}
   =
   \frac{\mathbb E_{k,m}|H_{ij}[k,m]|^2}
   {\mathbb E_{k,m}|H_{jj}[k,m]|^2}
   }
   \]

   它比较的是**同一个 UE 在不同输出中的功率**。

   而在波束 \(i\) 上，UE \(j\) 相对于期望 UE \(i\) 的干扰比为：

   \[
   \boxed{
   \mathrm{ISR}_{i\leftarrow j}
   =
   \frac{P_j\mathbb E|H_{ij}|^2}
   {P_i\mathbb E|H_{ii}|^2}
   }
   \]

   这两个指标不要混用。

   如果希望出现例如 \(-10/-20/-30\ \mathrm{dB}\) 不同强度的耦合：

   - 调整 UE 角度、波束间隔、孔径和指向偏差；
   - 有幅度控制的硬件可增加 taper；
   - 纯移相器硬件应保持恒模约束；
   - 调整 UE 功率可改变 ISR，但不会改变天线自身的相对泄露。

   **真实几何下，12 个非对角泄露系数通常不能自由独立指定。** 可以额外建立人工耦合矩阵作为算法对照，但要将其与 RT 物理模型分开报告。

7. **建立正确的输出噪声模型**

   - [ ] 明确噪声是在模拟合并前还是合并后引入。
   - [ ] 共用阵列使用合并后的噪声协方差。
   - [ ] 不重复加噪声。
   - [ ] 用纯噪声数据验证样本协方差。

   如果阵元噪声满足：

   \[
   \mathbf n_{\mathrm{ant}}\sim
   \mathcal{CN}(\mathbf0,\sigma_a^2\mathbf I)
   \]

   后级四路独立噪声满足：

   \[
   \mathbf v\sim
   \mathcal{CN}(\mathbf0,\sigma_v^2\mathbf I_4)
   \]

   那么：

   \[
   \boxed{
   \mathbf R_\eta
   =
   \sigma_a^2\mathbf W^H\mathbf W
   +
   \sigma_v^2\mathbf I_4
   }
   \]

   仿真有两种等价实现：

   - 生成阵元噪声，再用 \(\mathbf W^H\) 合并；
   - 对 \(\mathbf R_\eta\) 做 Cholesky 分解，直接生成四路相关噪声。

   ```python
   R_eta = (
       sigma_a2 * (W.conj().T @ W)
       + sigma_v2 * np.eye(4)
   )

   L = np.linalg.cholesky(R_eta)

   z = (
       rng.standard_normal((batch, M, K, 4))
       + 1j*rng.standard_normal((batch, M, K, 4))
   ) / np.sqrt(2)

   eta = np.einsum("ij,bmkj->bmki", L, z)
   ```

   第一版若扫描 SNR，可固定 UE 功率、改变噪声尺度。例如明确将 UE1 主波束的无干扰 SNR 定义为：

   \[
   \mathrm{SNR}_{1,0}
   =
   \frac{P_1\mathbb E|H_{11}|^2}
   {[\mathbf R_\eta]_{11}}
   \]

   其他 UE 的 SNR 按实际功率和信道计算。不要同时独立固定所有 UE 的 SNR，又声称保留了近远效应。

8. **同时生成四个 UE 的发送信号**

   - [ ] 四个 UE 使用独立比特、独立调制符号。
   - [ ] 四 UE 的数据占用相同 RE。
   - [ ] 每个 UE 只生成一份波形，通过其信道列进入四路输出。
   - [ ] 发射功率只乘一次。
   - [ ] 先做无编码 QPSK，确认模型后再加入 LDPC。

   若 \(\mathbb E|s_j|^2=1\)，将功率并入信道：

   \[
   \mathbf F=\mathbf H_{\mathrm{beam}}\mathbf P^{1/2}
   \]

   ```python
   F = H_beam * np.sqrt(p_ue)[None, None, None, :]

   Y_clean = np.einsum("mkij,bmkj->bmki", F, S)
   Y = Y_clean + eta
   ```

   对于静态信道，可把一个时间切片广播到 14 个符号；这种广播表示信道静态，不是 14 份独立信道实现。

   **完成标准：** 单独开启 UE1 时，四路都能看到同一份 UE1 数据，只是幅相不同。关闭 UE1 后，它在四路中的贡献全部消失。

9. **实现独立波束、ZF、联合 LMMSE 三条接收链**

   - [ ] 独立波束：每个 UE 只使用对应输出。
   - [ ] ZF：四路联合，记录噪声增强。
   - [ ] LMMSE：四路联合，传入完整噪声协方差。
   - [ ] 使用均衡器输出的有效噪声方差计算 LLR。

   独立波束的理论 SINR：

   \[
   \mathrm{SINR}_i=
   \frac{|F_{ii}|^2}
   {\sum_{j\neq i}|F_{ij}|^2+[\mathbf R_\eta]_{ii}}
   \]

   独立均衡：

   \[
   \hat s_i=\frac{y_i}{F_{ii}}
   \]

   对应残余方差：

   \[
   n_{\mathrm{eff},i}
   =
   \frac{\sum_{j\neq i}|F_{ij}|^2+[\mathbf R_\eta]_{ii}}
   {|F_{ii}|^2}
   \]

   用这个方差做普通高斯解调，是将其他 UE 的离散星座干扰近似为高斯的接收基线。

   对联合 LMMSE，建议先使用支持完整协方差的低层接口：

   ```python
   import torch
   from sionna.phy.mimo import lmmse_equalizer

   # y_t: [..., 4]
   # f_t: [..., 4, 4]
   # r_t: [..., 4, 4]
   x_hat, no_eff = lmmse_equalizer(y_t, f_t, r_t)
   ```

   官方接口中的第三个参数是噪声协方差矩阵，并返回软符号和每流有效噪声方差。它还包含供解调器使用的增益归一化。[Sionna 2.2.0](https://nvlabs.github.io/sionna/phy/api/mimo/sionna.phy.mimo.lmmse_equalizer.html?utm_source=chatgpt.com)

   **联合检测全部四个 UE 时，其他三个 UE 已在 \(\mathbf F\) 中建模，不能再次将它们的功率加进噪声协方差。** 只有未联合检测的外部用户，才作为额外干扰进入协方差。

   四路还必须具有可用的相对相位参考。第一版假定相干、已校准；后续加入 RF 链路幅相误差。

10. **接入 OFDM 导频、信道估计和编码**

   - [ ] 使用 `ResourceGrid` 配置 4 个发射机、每发射机 1 流。
   - [ ] 使用一个接收机、4 个接收观测端口的关联关系。
   - [ ] 导频能够区分四个 UE。
   - [ ] 估计完整 \(4\times4\) 波束域信道。
   - [ ] 比较完美 CSI 与估计 CSI。
   - [ ] 加入 LDPC 后统计每 UE 的 BER、BLER 和吞吐量。

   `StreamManagement` 的关联应对应：

   ```python
   rx_tx_association = np.ones((1, 4), dtype=np.int32)
   ```

   而不是用 \(4\times4\) 单位矩阵把四路输出视作四个互不协作的接收机。

   联合接收机需要估计：

   \[
   \hat{\mathbf F}[k,m]\in\mathbb C^{4\times4}
   \]

   可以先使用四 UE 正交导频，再进入 NR PUSCH 的 DMRS 配置。相同数据资源并不要求发送完全相同的导频。

   官方 MU-MIMO 上行教程包含四 UE、`ResourceGrid`、`StreamManagement`、LS 估计、LMMSE 和 LDPC。可以沿用其 PHY 链路，将原来的随机传播信道替换为你的波束域 RT 信道。[Sionna 2.2.0](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Realistic_Multiuser_MIMO_Simulations.html?utm_source=chatgpt.com)

   如果高层 OFDM 均衡封装没有提供你需要的完整相关噪声输入，第一版就保留自定义 RE 级调用。也可以先白化：

   \[
   \mathbf R_\eta=\mathbf L\mathbf L^H
   \]

   \[
   \tilde{\mathbf y}=\mathbf L^{-1}\mathbf y,\qquad
   \tilde{\mathbf F}=\mathbf L^{-1}\mathbf F
   \]

   白化后噪声协方差为单位阵，但信道估计及其误差统计也要与该变换保持一致。

11. **按场景逐级加入复杂度**

   | 阶段 | 增加的内容 | 要解决的问题 |
   |---|---|---|
   | A | 静态 LoS、完美 CSI | 验证空间耦合与检测 |
   | B | UE 功率不均衡 | 验证近远效应 |
   | C | 导频估计、编码 | 验证实际链路性能 |
   | D | 地面/建筑反射 | 验证多径加权及频率选择性 |
   | E | 指向误差、权值量化、链路校准误差 | 验证模拟前端限制 |
   | F | 卫星运动、不同 UE 多普勒 | 验证动态上行接收 |
   | G | 宽带阵列时延、ICI、相位噪声 | 验证宽带高速场景 |

   多径情况下：

   \[
   H_{ij}(f,t)
   =
   \sum_\ell
   \alpha_{j\ell}(t)
   \underbrace{\mathbf w_i^H\mathbf a(\mathbf u_{j\ell},f)}_{\text{每条路径分别被加权}}
   e^{-j2\pi f\tau_{j\ell}}
   \]

   不能总是将 RT 生成的一个标量多径信道，统一乘一个固定旁瓣系数。不同反射路径具有不同到达角，可能分别落在主瓣、旁瓣或零陷。

   **宽带时尤其要检查 synthetic array 的限制。** 当前实现的 synthetic array 空间相位按场景载频构造，后续 CFR 变换使用路径时延；不能仅因计算了多个子载波，就假定已包含完整的阵元间宽带时延和 beam squint。这个判断来自官方源码。[Sionna 2.2.0](https://nvlabs.github.io/sionna/_modules/sionna/rt/path_solvers/paths.html?utm_source=chatgpt.com)

   需要精确研究宽带阵列时，可选择：

   - `synthetic_array=False`，保留阵元间差分传播时延；
   - 或根据 RT 的逐路径角度和中心路径参数，自行构造随 \(f_c+f_k\) 变化的阵列响应。

   对卫星多普勒，符号级 CFR 适合残余 ICI 很小的情况；如果要评估不同 UE 残余频偏引起的 ICI，需要时域波形信道。

   从地面基线转为真实 NTN 几何时，还要检查：接收阵列朝向、地球曲率、轨道位置速度、链路预算和坐标精度。尤其不要通过随意缩短卫星距离来替代真实场景，因为这会改变用户角间隔和空间可分离性。大气、雨衰及实际射频损耗按需求作为额外模型接入。

12. **执行验证与参数扫描，形成可解释的结果**

   首先完成以下检查：

   | 检查 | 预期结果 |
   |---|---|
   | 单 UE、无噪声 | 四路输出均等于对应信道列乘同一 UE 符号 |
   | 几何波束检查 | 主瓣峰值指向预期方向 |
   | 理想 LoS 阵列检查 | RT 合并结果与解析阵列响应一致 |
   | 合并一致性 | \(\mathbf W^H(\mathbf H_{\mathrm{ant}}\mathbf x)=(\mathbf W^H\mathbf H_{\mathrm{ant}})\mathbf x\) |
   | 纯噪声检查 | 样本协方差接近 \(\mathbf R_\eta\) |
   | 无噪声、满秩、完美 CSI 的 ZF | 能恢复四个 UE 符号 |
   | UE 方向逐渐接近 | 空间相关性增大，联合检测可能恶化 |
   | 同比例提高四 UE 功率 | 独立波束接收出现干扰限制 |
   | 完美 CSI 与估计 CSI | 能解释性能差距来源 |

   然后按单因素逐步扫描：

   | 参数 | 建议扫描内容 |
   |---|---|
   | SNR | 从低到高，覆盖性能转折区 |
   | 用户角间隔 | 从易分离到难分离 |
   | 功率不均衡 | 0、10、20、30 dB |
   | 指向偏差 | 按主瓣宽度的一定比例设置 |
   | 相位量化 | 理想、有限 bit |
   | 场景 | LoS、地面反射、建筑多径 |
   | 信道信息 | 完美 CSI、估计 CSI |
   | 接收算法 | 独立、ZF、LMMSE，随后 SIC |
   | 宽带/运动 | 窄带基线、宽带、残余频偏 |

   每个实验保存：

   - 四波束方向图及 UE 方向标记；
   - \(4\times4\) 泄露功率矩阵与复信道；
   - 噪声协方差；
   - 白化信道的奇异值和条件数；
   - 每 UE 检测前后 SINR、EVM、BER、BLER；
   - 吞吐量定义、发射功率和导频开销；
   - 场景、版本、随机种子和权值。

   BLER 仿真设置错误块数和最大块数两个停止条件；零错误结果要同时记录仿真块数，不能解释成真实 BLER 为零。

建议你的第一个完成里程碑是：**空场景 LoS → 64 阵元信道 → 四组模拟权值 → 完整四波束耦合矩阵 → 正确相关噪声 → 无编码 QPSK → 独立/ZF/LMMSE 对比**。这一阶段跑通后，再接你的 PUSCH、DMRS 和 NTN 动态模型，后续性能变化就有清楚的基线可查。