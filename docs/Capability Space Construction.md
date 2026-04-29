

 **Capability Space Construction**

**P1 脉络**

**(1) 动机与基线缺陷 (Motivation & Baseline Limitations)**

要对异构的训练数据进行精准的价值评估，首要任务是构建一个能够反映数据认知分布的“能力空间”。最直观的做法是在数据特征空间中直接应用 K-Means 等全局扁平聚类算法，提取每个簇的中心作为能力维度。然而，真实世界大模型训练数据的能力分布呈现出**极度的粒度不均衡**（例如，通识数据可能是宏观的粗粒度簇，而垂直领域的代码或数学数据则需要极细粒度的子簇划分）。传统的静态聚类无法自适应这种多粒度特征，容易导致小众核心能力的特征被淹没在庞大的粗粒度簇中。

**(2) 目标驱动的增量层次聚类 (Algorithm 1 Overview)**

为此，我们提出了一种增量式的能力树构建算法（ALG1）。与传统的固定二叉树不同，当我们的算法决定对某个过载的父节点 $P$ 进行细化时，它会在局部子空间内通过优化轮廓系数（Silhouette Coefficient），自适应地将其分裂为最能代表局部拓扑结构的 $k$ 个纯净子簇。整个路由与吸收过程极为流畅，但该算法面临着一个最核心的决策难题：**何时触发父节点的分裂？（Splitting Criterion）** 在面对海量数据时，我们无法依赖 LLM 庞大的上下文开销来进行启发式判别，因此，我们需要一个严格的数学准则来指导空间的动态演化。

**(3) 表达能力与鲁棒性的权衡 (The Expressivity-Robustness Trade-off)**

确立分裂准则的核心，在于平衡能力空间的“拟合精度”与簇的“抗噪质量”：

* **一方面（拟合上限）：** 提取出的能力维度被用于线性组合并逼近最终的“理想优化目标”。显然，父节点被拆分的子簇数量越多、粒度越细，整个能力空间张成的基向量（Basis）就越丰富，拟合理想目标能力的残差就越小。  
* **另一方面（抗噪底线）：** 子簇的紧凑程度（由轮廓系数/密度支撑）决定了该维度特征的纯净度。过度的分裂会导致簇内数据点过于稀疏，引入局部噪声，甚至将单一的语义逻辑强行割裂，导致梯度方向的剧烈震荡。

**(4) 理论指导与算法闭环 (Theoretical Guidance via Lemma 1\)**

如何在“降低模型拟合误差”与“维持数据特征高密度”之间取得最优解？我们在接下来的 **Lemma 1** 中对这一权衡进行了严格的理论建模。通过界定数据采样的遗憾上界（Regret Bound），Lemma 1 从理论上推导出了一个临界条件。我们将这一由理论推导出的临界不等式直接嵌入到 ALG1 中，作为触发节点分裂的硬性判别条件，从而完成了从理论边界到系统工程的完整闭环。

**P2 Lemma1**

以下是详尽的前置定义以及 Lemma 1 的严密证明。

### **1\. 严密前置定义 (Rigorous Preliminaries)**

**假设 1：特征空间与梯度映射 (Feature Space & Gradient Mapping)**

设所有数据点 $x$ 存在于维度为 $d$ 的特征空间 $\\mathcal{X} \\subset \\mathbb{R}^d$ 中。存在一个参数优化映射 $\\mathcal{G}: \\mathcal{X} \\to \\mathbb{R}^p$，其中 $\\mathcal{G}(x) \= \\nabla \\mathcal{L}(\\theta; x)$ 表示样本 $x$ 对模型参数的真实梯度。

**假设 2：利普希茨连续性 (Lipschitz Continuity)**

假设映射 $\\mathcal{G}$ 在特征空间中满足利普希茨连续条件，即存在常数 $L \> 0$，使得对于任意 $x, y \\in \\mathcal{X}$，有：

$$||\\mathcal{G}(x) \- \\mathcal{G}(y)||\_2 \\le L ||x \- y||\_2$$

---

**定义 1：向量量化分裂与经验基底 (Vector Quantization Split & Empirical Basis)**

考虑一个包含 $N\_P$ 个样本的父节点簇 $P$。使用 K-Means 将其划分为 $m$ 个互不相交的子簇 $\\{C\_1, \\dots, C\_m\\}$，各子簇的样本数为 $N\_i$，特征中心为 $\\mu\_i$。

对于任意子簇 $C\_i$，其理论上的**真实基底梯度 (True Basis Gradient)** 定义为期望值 $g\_i \= \\mathbb{E}\_{x \\in C\_i}\[\\mathcal{G}(x)\]$。

而在实际应用中，我们只能依赖有限样本的**经验基底梯度 (Empirical Basis Gradient)**：

$$\\hat{g}\_i \= \\frac{1}{N\_i}\\sum\_{x \\in C\_i} \\mathcal{G}(x)$$  
在动态数据配方中，我们赋予各子簇采样权重 $\\alpha \\in \\Delta^{m-1}$（即单纯形 $\\sum \\alpha\_i \= 1, \\alpha\_i \\ge 0$），从而得到**经验合成梯度 (Empirical Synthesized Gradient)**：

$$\\hat{g}\_{\\alpha} \= \\sum\_{i=1}^m \\alpha\_i \\hat{g}\_i$$  
**定义 2：分裂表达能力与逼近偏差界 (Expressivity & Approximation Bias Bound)**

令未知且理想的优化目标梯度为 $g^\*$。**逼近偏差 (Approximation Bias)** 定义为能力空间所能合成的最优理论梯度与 $g^\*$ 之间的距离：

$$Bias(m) \= \\min\_{\\alpha \\in \\Delta^{m-1}} \\left|\\left| \\sum\_{i=1}^m \\alpha\_i g\_i \- g^\* \\right|\\right|\_2$$  
根据高分辨率量化理论，当有限支撑集被划分为 $m$ 个簇时，簇中心张成的子空间对原空间的量化误差随 $m$ 严格衰减。存在仅与内在维度 $d$ 相关的常数 $K\_d \> 0$，使得平方逼近偏差满足界限：

$$Bias^2(m) \\le K\_d \\cdot m^{-\\frac{2}{d}}$$  
**定义 3：几何紧凑度与采样方差 (Geometric Compactness & Sampling Variance)**

由于实际训练通过经验基底 $\\hat{g}\_i$ 进行，有限采样会引入偏差波动。定义子簇 $C\_i$ 的**几何紧凑度 (或簇内惯量 Inertia)** 为 $\\mathcal{I}(C\_i) \= \\frac{1}{N\_i} \\sum\_{x \\in C\_i} ||x \- \\mu\_i||\_2^2$。

基于假设 2，经验基底估计的**采样方差 (Sampling Variance)** 存在由几何紧凑度和样本量共同决定的上界：

$$Var(\\hat{g}\_i) \= \\mathbb{E}\\left\[ ||\\hat{g}\_i \- g\_i||\_2^2 \\right\] \\le \\frac{L^2 \\mathcal{I}(C\_i)}{N\_i}$$  
我们定义 $\\mathcal{R}(C\_i) \= \\frac{\\mathcal{I}(C\_i)}{N\_i}$ 为该簇的**结构风险密度 (Structural Risk Density)**，它衡量了过拟合和局部噪声的极限。

### ---

**2\. Lemma 1 及其详细证明 (Lemma 1 & Proof)**

**Lemma 1 (分裂的误差界与临界判定准则 \- Error Bound and Splitting Criterion)**

给定一个全局已存在 $m$ 个簇的能力空间。假设对局部父节点簇 $P$（规模为 $N\_P$，结构风险密度为 $\\mathcal{R}(P)$）执行分裂操作，产生 $k$ 个子簇 $\\{C\_1, \\dots, C\_k\\}$。

该分裂操作能严格降低全局预期梯度合成误差上界（即充分保证 Expressivity 收益大于 Robustness 损耗）的充要条件是，分裂后子簇的**最大结构风险密度**增量被全局表达能力的边缘收益严格包络，即满足：

$$L^2 \\left( \\max\_{1 \\le j \\le k} \\frac{\\mathcal{I}(C\_j)}{N\_j} \- \\frac{\\mathcal{I}(P)}{N\_P} \\right) \< K\_d \\left( m^{-\\frac{2}{d}} \- (m+k-1)^{-\\frac{2}{d}} \\right)$$  
**证明 (Proof):**

**Step 1: 建立全局期望总误差的偏差-方差分解**

我们需要最小化实际采样的经验合成梯度 $\\hat{g}\_{\\alpha}$ 与理想目标 $g^\*$ 之间的期望均方误差 (Expected Mean Squared Error)。设 $\\alpha^\*$ 为达到最小逼近偏差的理论最优分配比例。

$$E\_{total} \= \\mathbb{E} \\left\[ ||\\hat{g}\_{\\alpha^\*} \- g^\*||\_2^2 \\right\]$$  
通过引入理论合成梯度 $\\bar{g}\_{\\alpha^\*} \= \\sum \\alpha\_i^\* g\_i$ 作为桥梁，对其进行分解：

$$E\_{total} \= \\mathbb{E} \\left\[ ||(\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*}) \+ (\\bar{g}\_{\\alpha^\*} \- g^\*)||\_2^2 \\right\]$$  
由于 $\\hat{g}\_{\\alpha^\*}$ 的期望即为 $\\bar{g}\_{\\alpha^\*}$，交叉项期望为零，上式严格解耦为：

$$E\_{total} \= \\underbrace{||\\bar{g}\_{\\alpha^\*} \- g^\*||\_2^2}\_{Bias^2(m)} \+ \\underbrace{\\mathbb{E} \\left\[ ||\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*}||\_2^2 \\right\]}\_{Var(\\alpha^\*)}$$  
**Step 2: 确定方差的鲁棒性上界**

对于给定的分配 $\\alpha^\*$，由于各个子簇的经验采样相互独立，方差项可以展开为：

$$Var(\\alpha^\*) \= \\sum\_{i=1}^m (\\alpha\_i^\*)^2 \\mathbb{E} \\left\[ ||\\hat{g}\_i \- g\_i||\_2^2 \\right\]$$  
将前置定义 3 代入：

$$Var(\\alpha^\*) \\le \\sum\_{i=1}^m (\\alpha\_i^\*)^2 \\frac{L^2 \\mathcal{I}(C\_i)}{N\_i}$$  
在动态数据配方（Recipe Evolution）中，为了最大化梯度对齐，机制可能导致 $\\alpha^\*$ 在某几个最优子簇上产生极端激活（例如 $\\alpha\_j \\to 1$ 且其余为 $0$）。为了在任何可能的动态权重分配下都绝对保证框架的抗噪鲁棒性，我们必须放缩至最坏情况下的全局风险上界：

$$Var\_{worst}(m) \= L^2 \\max\_{1 \\le i \\le m} \\left( \\frac{\\mathcal{I}(C\_i)}{N\_i} \\right)$$  
因此，分裂前的总误差上界为：

$$E\_{bound}(m) \= K\_d \\cdot m^{-\\frac{2}{d}} \+ L^2 \\max\_{1 \\le i \\le m} \\mathcal{R}(C\_i)$$  
**Step 3: 计算分裂的边缘收益与代价极限**

当我们把节点 $P$ 细分为 $k$ 个子簇 $\\{C\_1, \\dots, C\_k\\}$ 时，全局簇数量从 $m$ 变为 $m' \= m \+ k \- 1$。

此时，**逼近偏差的衰减（收益）** 为：

$$\\Delta Bias^2\_{gain} \= K\_d \\left( m^{-\\frac{2}{d}} \- (m')^{-\\frac{2}{d}} \\right) \> 0$$  
而在局部，节点 $P$ 的结构风险密度从 $\\mathcal{R}(P) \= \\frac{\\mathcal{I}(P)}{N\_P}$ 突变为子集中的最坏风险 $\\max\_{1 \\le j \\le k} \\frac{\\mathcal{I}(C\_j)}{N\_j}$。虽然 K-Means 保证绝对惯量总和 $\\sum \\mathcal{I}(C\_j) \\le \\mathcal{I}(P)$，但由于样本量 $N\_j$ 可能呈现断崖式萎缩（碎片化），导致比值激增。

因此，分裂带来的**最大潜在方差恶化（代价）** 为：

$$\\Delta Var\_{penalty} \= L^2 \\left( \\max\_{1 \\le j \\le k} \\frac{\\mathcal{I}(C\_j)}{N\_j} \- \\frac{\\mathcal{I}(P)}{N\_P} \\right)$$  
**Step 4: 得出临界判定条件**

要求此次操作能优化全局空间（即严格降低总误差上界），必须满足收益大于代价：

$$\\Delta Var\_{penalty} \< \\Delta Bias^2\_{gain}$$  
即：

$$L^2 \\left( \\max\_{1 \\le j \\le k} \\frac{\\mathcal{I}(C\_j)}{N\_j} \- \\frac{\\mathcal{I}(P)}{N\_P} \\right) \< K\_d \\left( m^{-\\frac{2}{d}} \- (m+k-1)^{-\\frac{2}{d}} \\right)$$  
$\\blacksquare$

 

**P3 疑难解决：**

### **1\. $\\hat{g}\_{\\alpha^\*}$ 和 $g^\*$ 的本质区别是什么？**

要理解这个，我们需要建立一个“上帝视角”和“现实视角”的对比，中间还夹着一个“理论上限”。这三个变量代表了模型优化的三个层次：

* **$g^\*$ (终极理想目标)**：这是“上帝视角”的绝对完美梯度。你可以把它理解为如果模型吃透了真正的目标任务，它本应该走的绝对正确的方向。这是一个固定的、未知的常数向量。  
* **$\\bar{g}\_{\\alpha^\*}$ (理论期望合成梯度)**：我们在特征空间中切出了 $m$ 个簇，用这些簇的**真实质心（期望）** $g\_i$ 按照最优比例 $\\alpha^\*$ 合成的梯度。这是我们用 $m$ 个基底去拟合 $g^\*$ 所能达到的**理论上限**。它与 $g^\*$ 的差距，就是**偏差（Bias）**。  
* **$\\hat{g}\_{\\alpha^\*}$ (实际经验合成梯度)**：这是“现实视角”。因为我们没法拿到真正的簇质心 $g\_i$，我们只能从每个簇里**随机抽样（Sample）** 一部分数据来算出当前的梯度 $\\hat{g}\_i$。这就是说，$\\hat{g}\_{\\alpha^\*}$ 是一个**带有随机采样噪声的随机变量**。

**总结**：$g^\*$ 是我们要追逐的死目标（常量）；而 $\\hat{g}\_{\\alpha^\*}$ 是我们在现实中带着抽样误差拼凑出来的带噪方向（随机变量）。我们要算的总误差 $\\mathbb{E} \\left\[ ||\\hat{g}\_{\\alpha^\*} \- g^\*||\_2^2 \\right\]$，就是算“带噪现实”偏离“终极理想”的期望距离。

### ---

**2\. 交叉项期望为 0 那一步是怎么来的？**

这实际上是经典的“偏差-方差分解”操作。利用的是向量的 $L\_2$ 范数展开公式：$||A \+ B||\_2^2 \= ||A||\_2^2 \+ ||B||\_2^2 \+ 2\\langle A, B \\rangle$。

我们令：

* $A \= \\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*}$ （这是纯粹的抽样噪声，是一个**随机变量**）  
* $B \= \\bar{g}\_{\\alpha^\*} \- g^\*$ （这是理论合成梯度偏离终极目标的常数，是一个**固定向量**，不含随机性）

展开总误差的平方项：

$$||\\underbrace{(\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*})}\_{A} \+ \\underbrace{(\\bar{g}\_{\\alpha^\*} \- g^\*)}\_{B}||\_2^2 \= ||A||\_2^2 \+ ||B||\_2^2 \+ 2\\langle A, B \\rangle$$  
现在对整体套上期望 $\\mathbb{E}\[\\cdot\]$。因为期望是对**随机抽样**求的，所以常量可以提出来：

$$\\mathbb{E}\[2\\langle A, B \\rangle\] \= 2 \\langle \\mathbb{E}\[A\], B \\rangle$$  
我们来看 $\\mathbb{E}\[A\]$ 是多少：

$$\\mathbb{E}\[A\] \= \\mathbb{E}\[\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*}\] \= \\mathbb{E}\[\\hat{g}\_{\\alpha^\*}\] \- \\bar{g}\_{\\alpha^\*}$$  
根据定义，经验采样梯度 $\\hat{g}\_{\\alpha^\*}$ 的数学期望，正好就是理论梯度 $\\bar{g}\_{\\alpha^\*}$。所以：

$$\\mathbb{E}\[\\hat{g}\_{\\alpha^\*}\] \- \\bar{g}\_{\\alpha^\*} \= 0$$  
既然 $\\mathbb{E}\[A\] \= 0$，那么整个交叉项 $2 \\langle \\mathbb{E}\[A\], B \\rangle$ 就等于 0。式子由此干净地解耦为两部分。

### ---

**3\. Step 2 中是怎么展开把 $\\alpha^\*$ 提出来的？**

这一步利用的是**独立随机变量的方差性质**。

我们要计算的是 $Var(\\alpha^\*) \= \\mathbb{E} \\left\[ ||\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*}||\_2^2 \\right\]$。

先把里面代入各自的定义（都是累加和）：

$$\\hat{g}\_{\\alpha^\*} \- \\bar{g}\_{\\alpha^\*} \= \\sum\_{i=1}^m \\alpha\_i^\* \\hat{g}\_i \- \\sum\_{i=1}^m \\alpha\_i^\* g\_i \= \\sum\_{i=1}^m \\alpha\_i^\* (\\hat{g}\_i \- g\_i)$$  
这是一个求和的 $L\_2$ 范数平方 $\\mathbb{E}\[||\\sum X\_i||\_2^2\]$。展开后会包含平方项和交叉相乘项：

$$\\mathbb{E} \\left\[ \\sum\_{i=1}^m (\\alpha\_i^\*)^2 ||\\hat{g}\_i \- g\_i||\_2^2 \+ \\sum\_{i \\neq j} 2 \\alpha\_i^\* \\alpha\_j^\* \\langle \\hat{g}\_i \- g\_i, \\hat{g}\_j \- g\_j \\rangle \\right\]$$  
**关键点来了**：在数据采样的过程中，我们从簇 $C\_i$ 中抽样和从簇 $C\_j$ 中抽样是**相互独立**的事件。

因为相互独立，所以它们噪声的协方差期望为 0，即对于任意 $i \\neq j$：

$$\\mathbb{E} \\left\[ \\langle \\hat{g}\_i \- g\_i, \\hat{g}\_j \- g\_j \\rangle \\right\] \= \\langle \\mathbb{E}\[\\hat{g}\_i \- g\_i\], \\mathbb{E}\[\\hat{g}\_j \- g\_j\] \\rangle \= \\langle 0, 0 \\rangle \= 0$$  
所有的交叉项全部消失，只剩下自己平方的累加和，常数 $(\\alpha\_i^\*)^2$ 可以直接提到期望外面。这就类似于概率论中最基本的定理：相互独立随机变量和的方差，等于各自方差的和（$Var(aX \+ bY) \= a^2Var(X) \+ b^2Var(Y)$）。

由此得到：

$$Var(\\alpha^\*) \= \\sum\_{i=1}^m (\\alpha\_i^\*)^2 \\mathbb{E} \\left\[ ||\\hat{g}\_i \- g\_i||\_2^2 \\right\]$$

### ---

**4\. $E\_{bound}(m)$ 前面的 $K\_d \\cdot m^{-\\frac{2}{d}}$ 是怎么突然出现的？**

它并没有突然出现，它是从 **Step 1** 直接“继承”过来的。

在 Step 1 结束时，我们把总误差分成了两部分：

$$E\_{total} \= \\underbrace{||\\bar{g}\_{\\alpha^\*} \- g^\*||\_2^2}\_{Bias^2(m)} \+ Var(\\alpha^\*)$$  
这个式子的意思是：总误差 \= 逼近偏差 \+ 采样方差。

* **第一部分（逼近偏差）：** 仔细看我们前面写的 **“定义 2”**，里面明确给出了高分辨率量化理论的数学界限：  
  $$Bias^2(m) \= \\min\_{\\alpha} \\left|\\left| \\sum \\alpha\_i g\_i \- g^\* \\right|\\right|\_2^2 \\le K\_d \\cdot m^{-\\frac{2}{d}}$$  
  所以，第一项的最大上限就是 $K\_d \\cdot m^{-\\frac{2}{d}}$。  
* **第二部分（采样方差）：** 我们在 **Step 2** 刚刚算出来，无论权重怎么分配，它在最坏情况下的上界是：  
  $$Var\_{worst}(m) \= L^2 \\max\_{1 \\le i \\le m} \\mathcal{R}(C\_i)$$

因为我们要找的是总误差的最坏边界（Bound），自然就是把第一部分的上限和第二部分的上限加起来。所以把这两个项拼接在一起，就得到了最终的边界公式：

$$E\_{bound}(m) \= K\_d \\cdot m^{-\\frac{2}{d}} \+ L^2 \\max\_{1 \\le i \\le m} \\mathcal{R}(C\_i)$$