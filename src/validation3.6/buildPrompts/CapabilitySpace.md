# 角色与任务
你是一位资深的 AI 算法工程师与 Python 架构师。你的任务是为一个全新的大模型微调数据配方（Data Recipe）框架实现“能力空间构建 (Capability Space Construction, ALG1)”模块。

# 理论背景
该模块的核心目标是构建一个具备“高覆盖率 (High Coverage)”和“高明确度 (High Distinctness)”的动态粒度能力空间。
- **明确度 (Distinctness)**：由目标密度 $\rho_k = A_k / r_k \ge 1$ 衡量。其中 $A_k$ 是平均对齐度（簇内数据梯度与目标任务梯度的平均点积），$r_k$ 是簇内最大偏离（簇内数据与中心梯度的最大差异）。
- **分离度 (Separation)**：由 $\delta_{ij} = |A_i - A_j|$ 衡量。

# 架构与核心需求
请设计一套高模块化、面向对象的 Python 代码。代码需要针对 Linux/Ubuntu 环境进行优化，并在合适的地方（如批量数据处理）使用多进程。

请实现以下核心组件：

## 1. 数据结构设计 (Data Structures)
创建以下基础类：
- `DataPoint`: 存储原始文本、提取的标签（Tags）以及梯度向量的占位符。
- `CapabilityCluster`: 代表能力簇 $D_k$。它需要维护一个 `DataPoint` 的列表，并包含动态计算其中心向量、对齐度 ($A_k$)、偏离度 ($r_k$) 以及密度 ($\rho_k$) 的方法。

## 2. 阶段一：原子化画像生成 (Phase I: Atomic Profiling)
创建 `AtomicProfiler` 类：
- **输入**：原始数据集 $\mathcal{D}$ 和目标模型描述 $T$。
- **功能**：实现一个 `extract_tags(data_point, target_desc)` 方法。该方法目前可以 mock（模拟）零样本 LLM 的调用，用于提取细粒度的技能标签（例如："Physics", "Calculus"）。
- **输出**：原子技能空间 $S$。

## 3. 阶段二：动态密度聚类 (Phase II: Dynamic Density Clustering)
创建 `DynamicCapabilitySpace` 类，实现 ALG1 的核心循环。
*注意：由于直接计算神经网络梯度的开销极大，请将梯度计算和对齐度计算抽象为一个独立接口或 mock 函数（例如 `mock_compute_gradient_alignment`），以便我们后期可以无缝接入真实的 PyTorch Tensor。*

- **步骤 1：初始化 (Initialization)**：使用 `scikit-learn` 或 `scipy` 对标签特征进行层次聚类，生成初始簇集合 $\mathcal{C}^{(0)}$。
- **步骤 2：密度检测与分裂 (Density Check & Split)**：遍历所有簇。如果某个簇的密度 $\rho_k < \tau$（默认阈值 $\tau=1.0$），触发分裂机制（例如对该簇内部的向量使用 k=2 的 K-Means）将其拆分为两个子簇。
- **步骤 3：分离度检测与合并 (Separation Check & Merge)**：计算两两簇之间的分离度。如果 $\delta_{ij} < \epsilon$，说明区分这两个簇的收益极低，将它们合并以减少冗余。
- **步骤 4：覆盖率对齐 (Coverage Alignment)**：剔除那些平均对齐度 $A_k \approx 0$（与目标任务正交/无关）的簇。
- **循环机制**：不断重复步骤 2 和步骤 3，直到聚类结构达到 `Stable`（即在一轮遍历中没有发生任何分裂或合并操作）。

## 4. 输出规范 (Output Formatting)
- 强制使用严格的 Python 类型提示 (`typing` 模块)。
- 提供符合标准的 Docstring（Google 或 NumPy 风格）解释参数和数学逻辑。
- 提供一个 `main()` 执行块，生成一些假数据（Dummy Data），完整演示从 Phase I 到 Phase II 的端到端运行过程。
- 输出配套的 `requirements.txt`。

# 约束条件
- 绝对不要写成单个臃肿的脚本文件。必须将逻辑拆分到清晰的类和方法中。
- 数学逻辑必须严格遵循公式：密度 $\rho_k = A_k / r_k$。