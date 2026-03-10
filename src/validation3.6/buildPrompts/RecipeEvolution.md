# 角色与任务
你是一位资深的 AI 算法工程师与 PyTorch 架构师。你的任务是为一个全新的大模型微调数据配方框架实现“基于梯度对齐的动态配方演化 (Dynamic Recipe Evolution via Gradient Alignment, 4.3节)”模块。

# 理论背景
该模块通过细粒度的梯度匹配（Fine-grained Gradient Matching）在线提取样本级梯度反馈，交替更新**评估指标权重向量 $\alpha$** 和**能力维度权重向量 $\beta$**，并在离线阶段利用权重演化轨迹进行策略诊断。
- $\alpha^{(t)} \in \mathbb{R}^n$：第 $t$ 步的评估指标（Metrics）权重。
- $\beta^{(t)} \in \mathbb{R}^m$：第 $t$ 步的能力维度（Capability Clusters）权重。

# 架构与核心需求
请设计一套高模块化、基于 PyTorch 的面向对象代码。代码需要针对张量运算进行优化，避免低效的 for 循环。

请实现以下核心组件：

## 1. 状态追踪器 (State & Trajectory Tracker)
创建 `RecipeStateTracker` 类，用于初始化和存储 $\alpha$、$\beta$ 权重，并记录每个 step 的权重演化轨迹 $\mathcal{T} = \{ \alpha^{(t)}, \beta^{(t)} \}_{t=0}^T$。

## 2. 在线配方演化核心 (OnlineRecipeOptimizer)
创建 `OnlineRecipeOptimizer` 类，实现基于梯度的更新逻辑：
- **步骤 1：$\beta$ 权重更新 (`update_beta`)**：
  接收能力锚点验证集的平均梯度集合 $\{g_{C_j}^{(t)}\}$。
  更新公式：$\beta_j^{(t+1)} = \beta_j^{(t)} + \eta_\beta \cdot \frac{||g_{C_j}^{(t)}||_2}{\sum_{i=1}^m ||g_{C_i}^{(t)}||_2}$。
  更新后必须对 $\beta^{(t+1)}$ 进行 $L_1$ 归一化。
- **步骤 2：样本级奖励计算 (`calculate_reward`)**：
  接收单条数据梯度 $g_d$ 和它映射的 Top-K 候选能力簇的梯度。
  计算奖励：$R(d) = \frac{1}{|K_d|} \sum_{j \in K_d} \frac{g_d^\top g_{C_j}^{(t)}}{||g_d||_2 ||g_{C_j}^{(t)}||_2}$（即平均余弦相似度）。要求使用 PyTorch 向量化实现批量数据的奖励计算。
- **步骤 3：$\alpha$ 权重指数加权更新 (`update_alpha`)**：
  接收 Batch 内所有数据的奖励 $R(d)$ 及其对应的各指标得分 $M_i^{(d)}$。
  使用乘法权重更新（MWU）：$\tilde{\alpha}_i^{(t+1)} = \alpha_i^{(t)} \cdot \exp\left( \gamma \cdot \mathbb{E}_{d \in B_t} \left[ R(d) \cdot M_{i}^{(d)} \right] \right)$。
  随后进行熵正则化（Entropy Regularization）：$\alpha_i^{(t+1)} = (1 - \epsilon) \frac{\tilde{\alpha}_i^{(t+1)}}{\sum_{l=1}^n \tilde{\alpha}_l^{(t+1)}} + \frac{\epsilon}{n}$。

## 3. 离线诊断模块 (OfflineDiagnoser)
创建 `OfflineDiagnoser` 类，接收 `RecipeStateTracker` 记录的轨迹 $\mathcal{T}$ 进行诊断：
- **评估算子失效诊断**：计算全周期平均权重 $\bar{\alpha}_i = \frac{1}{T}\sum_{t=1}^T \alpha_i^{(t)}$。若 $\bar{\alpha}_i < \tau_\alpha$，则输出警告：该评估指标与实际优化目标不一致。
- **能力分布缺陷诊断**：若某能力维度 $j$ 的权重 $\beta_j^{(t)}$ 在训练后期（例如后 20% steps）呈现单调上升趋势且未衰减，同时传入的验证损失列表 $\mathcal{L}(V_{C_j})$ 趋于平缓（停滞），则输出警告：该维度可能存在信息熵不足或数据污染。

## 4. 输出规范与 Mock 验证 (Output Formatting & Mock Test)
- 提供符合标准的 Docstring，清楚注明输入 Tensor 的形状（Shape）预期，例如 `(batch_size, gradient_dim)`。
- 提供一个 `main()` 执行块，写一个 mock 的训练循环（Mock Training Loop），生成随机的假梯度 Tensor，实例化上述类，跑通 `update_beta -> calculate_reward -> update_alpha` 的流程，并最后调用 `OfflineDiagnoser` 输出诊断结果。

# 约束条件
- 所有的梯度相似度、范数计算必须完全使用 `torch` 原生算子，不要在计算图外使用 numpy 中转。
- 确保公式中的超参数（如 $\eta_\beta$, $\gamma$, $\epsilon$）在类初始化时可以自由配置。