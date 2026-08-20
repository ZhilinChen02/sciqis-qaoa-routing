# 0. 怎么使用这份文档

LEAD: 这份报告的目标不是只给出结果，而是让汇报者能够把“图上的路径问题如何一步步变成量子态上的概率优化”完整讲清楚。正文按照实际代码执行顺序组织；第 16 节可以直接作为 15 分钟口播稿；第 17 节是老师可能追问的问题与建议回答。

建议明天汇报时只抓住一条主线：**边变量编码约束与成本；cost layer 写入相位；mixer 通过干涉搬运概率；经典优化器寻找合适角度；最后按路径可行率和最优路径概率评价。**

> 一句话版：本项目在固定的 7 节点、14 边有向图上，把路由写成流守恒 QUBO，再比较不同 QAOA 搜索空间和 mixer 如何把成本相位转化为概率集中；最终课程算法在 20 条已枚举可行路径上，用 incumbent 阈值和 Grover mixer 将最优路径概率从 5% 的均匀基线提高到 p=3 三个种子的中位 99.8712%。

需要主动说明的边界：这是理想 statevector 教学模拟；最短路本身可以由 Dijkstra 等经典算法多项式时间求解；可行路径空间由经典方法完全枚举；因此本项目**不声称量子优势、可扩展性或硬件效率**。

# 1. 汇报先讲清楚的五个结论

1. **问题建模：**每一条有向边对应一个二进制变量 xₑ。14 条边给出 2¹⁴ = 16,384 个边选择状态，但只有 20 个状态是真正的 0→6 简单路径。
2. **约束进入目标函数：**节点流守恒误差平方后乘惩罚系数 A=6，与边权成本相加形成 QUBO：Q₆(x)=C(x)+6P_flow(x)。
3. **QAOA 的物理机制：**cost unitary 只改变振幅相位，不会立刻改变测量概率；mixer 将不同相位的振幅重新叠加，产生相长/相消干涉，才会改变概率分布。
4. **算法差异来自“在哪里搜”和“怎么移动”：**Penalty-X 与 Global-Grover 都在 16,384 维全空间中搜索；Feasible-Grover 与最终 GM-Th-QAOA 在 20 维可行路径空间中搜索。搜索空间和 mixer 几何都会改变初始基线及概率流。
5. **最终算法：**先用不含最优标签的贪心 incumbent 成本 11 构造严格阈值 C(P)&lt;11，再用阈值 phase + 可行域 Grover mixer 优化 Better Solution Probability。这个实例只有一条路径低于 11，事后确认它就是成本 10 的最优路径，因此 BSP 数值上等于 p_opt。

![图 1　固定路由图。红色高亮为唯一成本 10 的最优路径 0→1→2→4→5→6。](figures/qaoa_dynamics_deep_dive/v1/01_fixed_weighted_routing_graph.png)

<!-- pagebreak -->

# 2. “算法层”在整个项目中的位置

项目可以分成四层。汇报时所谓“算法层”主要覆盖中间两层，但必须交代它们的输入和输出。

| 层次 | 输入 | 核心工作 | 输出 | 主要代码 |
|---|---|---|---|---|
| 问题与数据层 | `graph.json` | 固定节点、边、边权、source、target 和 qubit 顺序 | 有向加权图 G | `src/graph.py` |
| 数学建模层 | 图 G、惩罚 A | 边变量、流守恒、QUBO、Ising、可行路径基 | 成本对角线或逻辑成本向量 | `src/qubo.py`, `src/feasible_qaoa.py` |
| 量子—经典算法层 | 初态、phase、mixer、深度 p | 交替量子演化；COBYLA 优化 γ、β | 最终 statevector 与概率分布 | `src/qaoa.py`, `src/feasible_experiments.py` |
| 评价与实验层 | 概率、解码器、参考最短路 | 计算 p_feas、p_opt、BSP、期望成本并比较方法 | 表格、图、保存结果 | `src/metrics.py`, `src/experiments/` |

整个数据流可以写成：

```text
weighted directed graph
  -> one bit x_e per edge
  -> flow residual f_v(x)
  -> Penalty-QUBO Q_6(x)
  -> Ising / diagonal cost Hamiltonian H_C
  -> initial state |psi_0>
  -> [cost phase U_C(gamma_l) -> mixer U_M(beta_l)] repeated p times
  -> statevector probabilities |a_x|^2
  -> route decoding
  -> p_feas, p_opt, BSP, expected cost
```

![图 2　代码采用的 Graph→QUBO→Ising→QAOA 主流水线。](figures/qaoa_dynamics_deep_dive/v1/02_qubo_ising_qaoa_pipeline.png)

## 2.1 两条实现路线

仓库里同时保留两条相关但不同的算法路线：

- **基础课程主线 Penalty-X：**14 个边变量，16,384 维计算基；用惩罚 QUBO 表达约束；用标准 X mixer 在海明超立方体上局部移动。
- **最终课程算法 GM-Th-QAOA：**先经典枚举 20 条可行路径，每条路径作为一个逻辑基态；使用 incumbent 阈值 phase 与秩一 Grover feasible mixer；深度 p=3。

前一条路线回答“约束如何进入二进制量子优化”；后一条路线回答“若把搜索限制在可行域并使用全局 mixer，概率如何被集中到比 incumbent 更好的路径”。两者应该作为递进关系讲，而不是互相替代。

# 3. 固定路由实例

## 3.1 图的基本信息

图 G=(V,E) 是有向无环图（DAG），节点 V={0,1,2,3,4,5,6}，source s=0，target t=6，共 14 条正权边。边按 q0→q13 固定排序，这个顺序同时决定二进制向量、QUBO 系数和量子比特编号。

| qubit | 有向边 | 权重 | qubit | 有向边 | 权重 |
|---:|---|---:|---:|---|---:|
| q0 | 0→1 | 2 | q7 | 2→4 | 3 |
| q1 | 0→2 | 4 | q8 | 2→5 | 7 |
| q2 | 0→3 | 7 | q9 | 3→4 | 2 |
| q3 | 1→2 | 1 | q10 | 3→5 | 4 |
| q4 | 1→3 | 4 | q11 | 4→5 | 2 |
| q5 | 1→4 | 7 | q12 | 4→6 | 5 |
| q6 | 2→3 | 2 | q13 | 5→6 | 2 |

经典参考由两种独立方式交叉验证：NetworkX 加权最短路与全部简单路径枚举都得到唯一最优解：

EQ: P* = 0 → 1 → 2 → 4 → 5 → 6，　C(P*) = 2+1+3+2+2 = 10。

对应被选中的边为 q0、q3、q7、q11、q13，canonical bitstring 是：

```text
q0 q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13
 1  0  0  1  0  0  0  1  0  0   0   1   0   1
canonical: 10010001000101
```

> 易错点：代码中的整数 basis index 把 q0 当作最低有效位；文档显示的 canonical bitstring 则明确按 q0→q13 排列。Qiskit 常见显示顺序会反过来，汇报时不要把“显示顺序”误当成“变量定义改变”。

## 3.2 为什么共有 16,384 个状态，但只有 20 条路径

每条边独立取 0 或 1，因此边变量全空间大小为：

EQ: |Ω_all| = 2^|E| = 2¹⁴ = 16,384。

然而任意边子集通常会断开、分叉、汇合错误或不能从 0 连到 6。独立解码器要求：从 source 出发，每一步恰好有一条已选出边；不重复节点；最终到达 target；并且不能遗留任何未使用的已选边。在该固定 DAG 上恰好有 20 个 bitstring 通过解码。

因此均匀全空间初态的结构基线为：

EQ: p_feas = 20/16,384 = 0.001220703125；　p_opt = 1/16,384 = 0.00006103515625。

如果直接在 20 条可行路径上均匀初始化，则：

EQ: p_feas = 1；　p_opt = 1/20 = 0.05。

这两个完全不同的初始基线，是理解后续结果的关键。

# 4. 从路径约束到 QUBO

## 4.1 边变量与路径成本

为每条边 e 定义二进制变量：

EQ: xₑ = 1 表示选择边 e；xₑ = 0 表示不选择边 e。

原始路径成本是边权的线性和：

EQ: C(x) = Σₑ wₑxₑ。

仅最小化 C(x) 会错误地选择空集，因为空集成本为 0。因此必须把“从 source 向 target 输送一个单位流”的约束加入模型。

## 4.2 节点流守恒

定义节点—边关联矩阵 B。若边 e 从节点 v 流出，则 Bᵥₑ=+1；若流入 v，则 Bᵥₑ=−1；否则为 0。供应向量 b 为：source 取 +1，target 取 −1，其余节点取 0。

节点 v 的流残差是：

EQ: fᵥ(x) = Σₑ Bᵥₑxₑ − bᵥ = outgoing(v) − incoming(v) − bᵥ。

所有残差为零时满足：

- source：流出比流入多 1；
- target：流入比流出多 1；
- 中间节点：流入等于流出。

矩阵形式更紧凑：

EQ: f(x)=Bx−b，　约束为 Bx=b。

对最优路径 q0、q3、q7、q11、q13，source 0 选一条出边；节点 1、2、4、5 都是一进一出；target 6 有一条入边，所以七个残差全为 0。

## 4.3 平方惩罚

将每个等式残差平方并求和：

EQ: P_flow(x) = Σᵥ fᵥ(x)² = ‖Bx−b‖²₂。

最终惩罚 QUBO 为：

EQ: Q_A(x) = C(x) + A·P_flow(x) = wᵀx + A‖Bx−b‖²₂。

仓库固定 A=6：

EQ: Q₆(x)=Σₑwₑxₑ + 6Σᵥfᵥ(x)²。

平方项展开为二次多项式：

EQ: ‖Bx−b‖² = xᵀBᵀBx − 2bᵀBx + bᵀb。

由于 xᵢ²=xᵢ，展开后只有常数项、线性项和二变量乘积 xᵢxⱼ，正好是 QUBO 形式。当前实例展开结果包含 14 个线性系数和 46 个非零二次耦合，常数项为 12。

## 4.4 惩罚系数 A=6 的意义

A 太小，会让违反流守恒但边权很低的状态拥有比真实路径更低的总能量；A 太大，则会拉大能量范围，使相位参数尺度更敏感。这里不是声称 A=6 对所有图都最优，而是对固定实例做了 16,384 状态穷举验证：

| 项目 | 数值 |
|---|---:|
| 唯一可行最优能量 | 10 |
| 次优可行路径成本 | 11 |
| 最低不可行 QUBO 能量 | 12 |
| 全部 QUBO 能量范围 | 10 到 207 |

因此 A=6 下没有不可行状态达到或低于成本 10 的基态，QUBO 的唯一基态与经典最短路一致。

> 严谨表述：**A=6 已通过该固定实例的穷举验证**。不要扩大成“A=6 是一般路由问题的理论最优惩罚”。

## 4.5 流守恒是否总等于一条合法路径

不总是。一般有向图里，流守恒还可能允许与 source-target 路径无关的有向环，或者更复杂的循环流。本项目的固定图是 DAG，不存在有向环；再结合二进制边变量和独立路径解码器，零流残差状态与 20 条有效路径在该实例上完全一致。

这是一个很适合答辩时主动说出的限定：说明我们理解数学约束的适用条件，而不是把实例性质误当成一般定理。

# 5. QUBO 到 Ising 哈密顿量

量子电路通常使用 Pauli-Z 算符表达对角成本。对计算基 |xᵢ⟩，Z 的本征值 zᵢ∈{+1,−1}，二进制变量与自旋变量的映射是：

EQ: xᵢ = (1−zᵢ)/2，对应算符 x̂ᵢ=(I−Zᵢ)/2。

将它代入 QUBO：

EQ: Q(x)=c + Σᵢlᵢxᵢ + Σᵢ&lt;ⱼqᵢⱼxᵢxⱼ，

可得到 Ising 成本哈密顿量：

EQ: H_C = c₀I + ΣᵢhᵢZᵢ + Σᵢ&lt;ⱼJᵢⱼZᵢZⱼ。

若 QUBO 二次系数为 qᵢⱼ，则 Jᵢⱼ=qᵢⱼ/4；每个二次项还会贡献到常数项和两个单体 h 系数。代码保留 identity 常数，使每个 basis state 的 Ising 能量与原 QUBO 能量完全相同。

当前 A=6 模型得到 Ising 常数 c₀=86、14 个 hᵢ、46 个 Jᵢⱼ。对全部 16,384 个计算基态逐一比较，最大 QUBO/Ising 能量误差为 0。

## 5.1 为什么要归一化能量

算法实际演化使用仿射归一化后的对角线，以稳定 γ 的数值尺度。全空间能量范围为 10 到 207：

EQ: H̃_C = (H_C−10I)/197。

可行路径成本范围为 10 到 14：

EQ: C̃(P) = (C(P)−10)/4。

仿射变换不改变能量排序。减去常数只引入全局相位；除以正数等价于重新定义 γ 的尺度。报告表里的 raw ⟨H_C⟩ 和期望路径成本仍然使用原始单位，不是归一化损失。

# 6. QAOA 的核心机制

## 6.1 p 层变分态

深度为 p 的 QAOA 状态写成：

EQ: |ψ(γ,β)⟩ = ∏ₗ₌₁ᵖ U_M(βₗ)U_C(γₗ)|ψ₀⟩。

代码每层严格执行“cost/phase 在前，mixer 在后”。参数存储顺序是：

```text
[gamma_1, gamma_2, ..., gamma_p, beta_1, beta_2, ..., beta_p]
```

因此 p=1 有 2 个连续参数，p=2 有 4 个，最终 p=3 有 6 个。

## 6.2 Cost layer：把成本写进相位

因为 H_C 对计算基是对角的：

EQ: U_C(γ)=exp(−iγH_C)，　aₓ → aₓexp(−iγEₓ)。

相位因子的模长为 1，所以：

EQ: |aₓexp(−iγEₓ)|² = |aₓ|²。

这意味着 cost layer 单独作用时：

- 每个 basis state 的测量概率不变；
- p_feas、p_opt 等概率指标不变；
- 因为 [U_C,H_C]=0，成本自身的期望值 ⟨H_C⟩ 也不变；
- 不同能量 Eₓ 获得不同旋转角，信息被编码到相对相位中。

> 汇报中最值得强调的一句：**成本哈密顿量把目标函数信息编码成相对相位；mixer 再把这些相位差转化成干涉和概率重分配。**因此不能说“cost layer 直接把能量降下来”。

## 6.3 Mixer：把相位差变成概率变化

mixer 一般不与 H_C 对易。它将不同 basis state 的复振幅线性组合；此前由成本产生的相位差会在叠加时造成相长或相消干涉，因此概率和期望能量可能改变。

Mixer 不保证每一层都让目标变好。保存的 p=2 Global-Grover 轨迹中，第一个 mixer 把 raw energy 从 86 提高到 117.6428，第二个 mixer 再降到 61.9427；Feasible-Grover 也先从 12.20 升到 12.9217，再降到 11.3066。QAOA 优化的是最终层结果，不是要求每层单调下降。

![图 3　cost checkpoint 保持概率，mixer checkpoint 通过干涉重分配概率。](figures/qaoa_dynamics_deep_dive/v1/10_phase_and_interference_evolution.png)

![图 4　逐层期望能量。水平段对应 cost layer 保持自身期望能量。](figures/qaoa_dynamics_deep_dive/v1/07_layer_by_layer_expected_hc.png)

# 7. 三种搜索几何与 mixer

![图 5　三种搜索空间和 mixer 几何：全空间局部 X、全空间全局 Grover、可行域全局 Grover。](figures/qaoa_dynamics_deep_dive/v1/03_three_search_spaces_and_mixers.png)

## 7.1 Penalty-X QAOA

初态是 14 qubit 的均匀叠加：

EQ: |+⟩^⊗14 = 1/√16,384 Σₓ|x⟩。

Cost diagonal 是归一化 Q₆。标准 X mixer 哈密顿量为：

EQ: H_X=ΣⱼXⱼ。

每个 Xⱼ 翻转一位，因此只连接海明距离为 1 的 bitstring；几何上是 14 维超立方体的连续时间量子游走。代码保留课程约定，在每层使用 β/14：

EQ: U_X(β)=∏ⱼexp[−i(β/14)Xⱼ]。

优点是表示直接、可写成常见单量子比特 RX 门；缺点是绝大多数全空间状态不可行，均匀初态有 99.8779% 的概率质量落在无效状态上。

## 7.2 Global-Grover QAOA

它与 Penalty-X 使用完全相同的 16,384 维表示、均匀初态和 byte-identical Penalty-QUBO diagonal；唯一关键变化是 mixer：

EQ: |s_all⟩=1/√16,384 Σₓ|x⟩，　H_G,all=|s_all⟩⟨s_all|。

EQ: U_G(β)=I+(exp(−iβ)−1)|s_all⟩⟨s_all|。

对任意状态 |ψ⟩，只需要计算 overlap ⟨s_all|ψ⟩ 再做秩一修正：

EQ: |ψ'⟩=|ψ⟩+(exp(−iβ)−1)|s_all⟩⟨s_all|ψ⟩。

实现时间和存储均为 O(2^q)，绝不构造 16,384×16,384 的稠密矩阵。由于只改变 mixer，Penalty-X 与 Global-Grover 的对比能较干净地说明 mixer 几何的影响。

## 7.3 Feasible-Grover QAOA

先用 `nx.all_simple_paths` 经典枚举 20 条有效路径，按“成本、再按节点序列字典序”排序。每条路径 Pᵢ 成为一个逻辑基态 |i⟩：

EQ: |s_F⟩=1/√20 Σᵢ₌₀¹⁹|i⟩。

成本 Hamiltonian 是这 20 条路径的归一化成本对角线，mixer 是：

EQ: H_G,F=|s_F⟩⟨s_F|，　U_G,F(β)=I+(exp(−iβ)−1)|s_F⟩⟨s_F|。

整个演化不可能离开这 20 个逻辑坐标，因此 p_feas=1 是结构不变量。要注意，这个 20 维逻辑模拟并不是自动等价于 5 个物理 qubit 上的廉价电路；状态制备、基态编码和 projector 分解都可能有额外成本。

## 7.4 Path-exchange mixer（历史可行域方案）

仓库还实现了逻辑 path-exchange mixer：若两条路径只在某个分叉—重汇区间选择了不同子路径，就在对应逻辑基态之间连边。本实例形成 106 条 route-exchange 边，route graph 连通。其邻接矩阵作为 mixer Hamiltonian，并通过特征分解计算 exp(−iβH_M)。

它体现“局部、结构感知”的可行域移动；Grover feasible mixer 则对均匀可行态做全局秩一更新。最终算法选择后者配合阈值 phase。

## 7.5 公平比较应该控制什么

| 对比 | 保持不变 | 改变 | 可以回答的问题 |
|---|---|---|---|
| Penalty-X vs Global-Grover | 16,384 维表示、初态、QUBO diagonal | mixer | 相同成本与表示下，mixer 几何如何影响概率流？ |
| Global-Grover vs Feasible-Grover | Grover projector 形式 | 支撑集合、初态基线、成本向量维度 | 搜索空间限制如何改变可行性与最优态基线？ |
| Route-cost Grover vs Threshold Grover | 20 维可行基、均匀初态、Grover mixer | phase 与经典 loss | 连续成本相位与 Boolean 标记相位有何差异？ |

不能只看最终 p_opt 就断言某个 mixer 普遍更好，因为改变搜索空间会同时把初始 p_opt 从 1/16,384 提高到 1/20。

# 8. 最终算法：incumbent-threshold GM-Th-QAOA

## 8.1 为什么引入 incumbent 阈值

普通期望值优化最小化整个分布的平均成本，可能把很多概率移到“还不错但不是最好”的路径。最终算法改用 Better Solution Probability（BSP）：只关心输出是否严格优于一个已有可行解。

贪心规则生成 incumbent：

EQ: P_inc = 0→1→2→3→4→5→6，　C(P_inc)=11。

阈值集合只根据 raw cost 和 incumbent cost 构造：

EQ: B={Pᵢ : C(Pᵢ)&lt;C(P_inc)}={Pᵢ:C(Pᵢ)&lt;11}。

定义 Boolean threshold function：

EQ: h_T(Pᵢ)=1（若 C(Pᵢ)&lt;11），否则 h_T(Pᵢ)=0。

当前实例恰好只有 route id 0 被标记。注意构造 h_T 时没有传入 `optimal_route_id`；只有优化完成后，评价模块才把标记集合与经典最优参考比较。

## 8.2 初态、phase 和 mixer

初态是均匀可行态：

EQ: |ψ₀⟩=|s_F⟩=1/√20 Σᵢ|i⟩。

Threshold phase 只给被标记路径相位：

EQ: U_T(γ)=exp[−iγ·diag(h_T)]。

Grover feasible mixer 为：

EQ: U_G(β)=I+(exp(−iβ)−1)|s_F⟩⟨s_F|。

p=3 变分态为：

EQ: |ψ(θ)⟩=U_G(β₃)U_T(γ₃)U_G(β₂)U_T(γ₂)U_G(β₁)U_T(γ₁)|s_F⟩。

## 8.3 优化目标

给定最终概率 pᵢ=|aᵢ|²：

EQ: BSP(θ)=Σᵢ:h_T(Pᵢ)=1 pᵢ(θ)。

SciPy COBYLA 是最小化器，因此代码返回：

EQ: L_BSP(θ)=−BSP(θ)。

在当前实例中 B 只有一个元素，且事后确认它就是唯一最优路径，所以 BSP=p_opt。这是实例事实，不是 BSP 定义要求预先知道最优解。

## 8.4 与振幅放大的关系

当一共有 N=20 个可行态、其中 M=1 个被标记态时，均匀初态标记概率为 M/N=0.05。标准 Grover 振幅放大可限制在“标记态均匀方向”和“未标记态均匀方向”张成的二维子空间中。令 sin²θ=M/N，则理想固定 Grover 迭代 k 次后的标记概率是：

EQ: P_marked(k)=sin²[(2k+1)θ]，　sin²θ=1/20。

由此得到 k=0: 0.05，k=1: 0.392，k=2: 0.81608，k=3: 0.9999392。实验 p=1 和 p=2 的中位结果正好是 0.392 与 0.81608；p=3 的变分优化中位 0.998712 接近第三次理想旋转。这说明机制与振幅放大高度一致。

但 GM-Th-QAOA 的 γ、β 是每层独立优化的连续参数，不应简单说成“就是固定角度的标准 Grover 算法”。p=4 仍可通过变分相位维持很高概率，而标准固定 Grover 迭代会发生 overshoot。

![图 6　项目结果中的 Grover 型概率放大机制。](figures/course/05_grover_amplification.png)

## 8.5 最终算法逐步伪代码

```text
Input: directed graph G, source s, target t, depth p=3,
       seeds {2601,2602,2603}, evaluation budget 150

1. F <- enumerate_all_simple_paths(G, s, t)       # classical preprocessing
2. sort F by (route_cost, node_sequence)
3. P_inc <- deterministic_greedy_route(G)
4. T <- route_cost(P_inc) = 11
5. h[i] <- 1 if route_cost(F[i]) < T else 0
6. prepare |s_F> = uniform superposition over the 20 logical routes

7. for each seed:
       initialize theta = [gamma_1..gamma_p, beta_1..beta_p]
       repeat until COBYLA stops or 150 requests are used:
           psi <- |s_F>
           for layer l = 1..p:
               psi[i] <- psi[i] * exp(-i * gamma_l * h[i])
               overlap <- <s_F | psi>
               psi <- psi + (exp(-i*beta_l)-1) * |s_F> * overlap
           probabilities <- |psi|^2
           loss <- -sum(probabilities[i] for h[i] == 1)
       retain the best finite in-bounds theta evaluated

8. decode the highest-probability route and report BSP, p_opt,
   p_feas, expected route cost and seed variability
```

# 9. 经典优化器如何包住量子演化

QAOA 是 hybrid algorithm：量子（这里由 statevector 模拟器替代）部分给定 θ 产生概率分布，经典部分根据 loss 更新 θ。

## 9.1 固定优化配置

| 配置项 | 最终 GM-Th-QAOA |
|---|---|
| 深度 | p=3 |
| 参数数 | 6 |
| 参数顺序 | 所有 γ 在前，所有 β 在后 |
| γ 范围 | [0,2π] |
| β 范围 | [0,π] |
| 优化器 | COBYLA（无梯度、局部） |
| 初始点 | `numpy.random.default_rng(seed)` 确定性采样 |
| seeds | 2601、2602、2603 |
| 每个 seed 请求上限 | 150 |
| `rhobeg` | 0.5 |
| `tol`, `catol` | 1e−8 |
| 保留策略 | 优化过程中见到的最佳 finite、in-bounds 参数 |

COBYLA 不需要解析梯度，适合小参数、可能有噪声或不可微评价的场景；但它是局部优化器，结果依赖初始点和预算。三个 p=3 运行都达到 150 次上限，所以 `success=False` 表示“达到预设预算”，不是状态向量错误，也不是结果无效；保留的是预算内见到的最佳参数，而不是收敛证明。

## 9.2 一次 objective evaluation 做什么

1. 从固定初态重新开始，不能沿用上一次 θ 的状态。
2. 依次施加 p 个 phase 和 p 个 mixer。
3. 取 statevector 振幅模平方并归一化。
4. 计算 BSP 或期望归一化成本。
5. 返回一个标量 loss 给 COBYLA。

真实采样型量子硬件不会直接给出完整 statevector，而会通过有限 shots 估计概率和期望值。本项目用精确 statevector，因此没有 shot noise；这里的重点是机制和数值验证。

# 10. 输出解码与评价指标

## 10.1 全空间方法的路径解码

对 14-bit 状态，解码器从 source 开始沿唯一已选出边前进，拒绝以下情况：没有出边、多条出边、回到已访问节点、未到 target，或到达 target 后仍有遗留选中边。解码成功才算可行路径。

## 10.2 核心指标

| 指标 | 定义 | 解释 |
|---|---|---|
| p_feas | Σₓ∈F p(x) | 采样得到有效路径的概率 |
| p_opt | Σₓ:C(x)=C* p(x) | 采样得到最优路径的概率；本实例最优唯一 |
| invalid mass | 1−p_feas | 无效边选择的总概率 |
| raw ⟨H_C⟩ | Σₓp(x)Q₆(x) | 全空间惩罚成本期望 |
| E[C\|feasible] | Σₓ∈F p(x)C(x)/p_feas | 条件于有效路径的平均原始成本 |
| BSP | Σᵢ:C(Pᵢ)&lt;C(P_inc) pᵢ | 比 incumbent 更好的概率 |
| amplification | final p_opt / initial p_opt | 相对初始基线的概率放大倍数 |

对于可行路径逻辑空间，所有坐标都代表有效路径，所以 p_feas=1 是构造保证。它证明演化没有离开定义好的逻辑空间，但不证明算法具有量子优势。

## 10.3 为什么 raw energy 不能随意跨表示比较

全空间 raw ⟨H_C⟩ 包含边权贡献和 6 倍流惩罚：

EQ: ⟨H_C⟩=⟨H_route⟩+6⟨P_flow⟩。

可行空间的流惩罚恒为 0，raw energy 就是期望路径成本。因此 61.94（全空间）和 11.31（可行空间）来自不同支撑分布与不同工作量，不能仅凭数字大小宣称一个算法“快了多少倍”或“能量好多少倍”。

# 11. 实验结果怎么解释

## 11.1 p=1/p=2 三几何动力学结果

所有六个单元都使用 COBYLA、seed 2601、最多 100 次 objective request 和相同的 grouped parameter convention。

| 方法 | p | 维度 | p_feas | p_opt | invalid mass | raw ⟨H_C⟩ | E[C\|feasible] | evals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Penalty-X | 1 | 16,384 | 0.00122070 | 0.00006104 | 0.99877930 | 86.0000 | 12.2000 | 50 |
| Penalty-X | 2 | 16,384 | 0.00122070 | 0.00006104 | 0.99877930 | 86.0000 | 12.2000 | 73 |
| Global-Grover | 1 | 16,384 | 0.00122070 | 0.00006104 | 0.99877930 | 86.0000 | 12.2000 | 32 |
| Global-Grover | 2 | 16,384 | 0.00417370 | 0.00021337 | 0.99582630 | 61.9427 | 12.1891 | 100 |
| Feasible-Grover | 1 | 20 | 1.00000000 | 0.05000000 | 0 | 12.2000 | 12.2000 | 23 |
| Feasible-Grover | 2 | 20 | 1.00000000 | 0.17092702 | 0 | 11.3066 | 11.3066 | 100 |

正确解释是：

- 固定单起点下，三个 p=1 运行都保留均匀基线；这是观察结果，不是 p=1 理论上必然无效。
- p=2 Global-Grover 在与 Penalty-X 完全相同的表示和成本 diagonal 下出现非平凡概率重分配，说明改变 mixer 足以改变浅层动力学。
- Feasible-Grover 的 p_feas=1 来自表示限制；p=2 将 p_opt 从 5% 提高到 17.09%。
- 两个 p=2 Grover 优化都达到 100 次预算上限，报告的是有限预算内最佳值，不是全局最优或收敛保证。

![图 7　p=1/p=2 最终概率质量分解。](figures/qaoa_dynamics_deep_dive/v1/06_probability_mass_decomposition.png)

## 11.2 最终 p=3 GM-Th-QAOA

最终课程配置使用 p=3、三个 seeds、每个 150 次请求：

| seed | p_feas | p_opt = BSP | 期望路径成本 | 放大倍数 | evals | 停止原因 |
|---:|---:|---:|---:|---:|---:|---|
| 2601 | 1 | 0.9998041831 | 10.0004534707 | 19.9961× | 150 | budget exhausted |
| 2602 | 1 | 0.8160340271 | 10.4260264635 | 16.3207× | 150 | budget exhausted |
| 2603 | 1 | 0.9987120038 | 10.0029827281 | 19.9742× | 150 | budget exhausted |
| **中位数** | **1** | **0.9987120038** | **10.0029827281** | **19.9742×** | **150** | — |

初始 p_opt=0.05，因此中位概率放大约 19.97 倍。seed 2602 只有 81.60%，说明局部优化仍有 seed 依赖，不能只展示最好的一次。三个运行全保留并报告，是比 cherry-pick 更可靠的做法。

![图 8　不同方法与深度的 p_opt 中位数；最终选择 GM-Th-QAOA p=3。](results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/figures/01_median_popt_versus_depth.png)

![图 9　最强最终分布与旧 Q2-F 分布比较。](results/q2f_final_improvement/q2fi-473475526184d55e7870caee93679b092d4443f5ec87dbd399a017412f430bb1/figures/02_strongest_distribution_vs_old_q2f.png)

## 11.3 p=1 到 p=4 的最终方法对比

| 方法 | p=1 median p_opt | p=2 | p=3 | p=4 |
|---|---:|---:|---:|---:|
| 旧 Q2-F expectation | 0.07119 | 0.04456 | 0.26490 | N/A |
| BSP path-exchange | 0.15223 | 0.13608 | 0.44466 | 0.37434 |
| GM-QAOA expectation | 0.05000 | 0.17093 | 0.18241 | 0.17562 |
| GM-Th-QAOA | **0.39200** | **0.81608** | **0.998712** | **0.998429** |

这个表支持“在该实例、该有限配置下，incumbent-threshold phase 与 Grover feasible mixer 的组合表现最强”。它不支持“GM-Th-QAOA 对所有路由实例、所有深度、所有优化器都更强”。

# 12. 计算复杂度与可扩展性

## 12.1 全空间 statevector 方法

设 q=|E|，N=2^q：

- statevector 存储 O(N) 个复数；当前 q=14 时 complex128 约 256 KiB。
- 对角 cost phase 每层 O(N)。
- X mixer 对 q 个 qubit 分别作用，每层 O(qN)。
- Global Grover rank-one mixer 每层 O(N)，但初态和整体模拟仍需 O(N) 状态存储。
- 经典优化总成本还要乘 objective evaluation 次数。

当边数增大时，2^q 是根本瓶颈。本项目能穷举 16,384 状态是因为实例很小。

## 12.2 可行路径逻辑方法

设可行简单路径数为 |F|：

- route cost phase 每层 O(|F|)。
- Grover feasible rank-one update 每层 O(|F|)。
- path-exchange mixer 当前通过稠密特征分解，预处理约 O(|F|³)，每次演化约 O(|F|²)。
- 关键瓶颈在经典枚举全部简单路径；一般图中的简单路径数可能指数增长。

当前 |F|=20，因此模拟极快。但一旦先完整枚举并计算每条路径成本，经典 `argmin` 已经能直接看到最优路径。这个逻辑空间实验的价值是解释量子概率动力学，不是提供比最短路算法更高效的求解器。

## 12.3 硬件实现还缺什么

从“20 维 Python 向量”到真实量子电路，至少需要回答：

- 怎样制备只覆盖所有可行路径的均匀态 |s_F⟩；
- 怎样把逻辑 route id 编码到物理 qubit；
- 怎样分解 exp(−iβ|s_F⟩⟨s_F|)；
- 怎样实现 threshold oracle h_T 而不先把全部路径成本写入经典表；
- 噪声、有限 shots 和 transpilation 后的电路深度是否可接受。

因此“逻辑维度只有 20”不能直接等价为“只用 5 个 qubit 就能高效实现完整算法”。

# 13. 代码调用链：老师问“公式在哪里落到代码”时怎么答

| 数学/算法对象 | 代码位置 | 关键函数 |
|---|---|---|
| 图加载与边顺序 | `src/graph.py` | `load_graph`, `get_edge_order` |
| 路径→14-bit 编码 | `src/graph.py` | `path_to_edge_bitstring` |
| 流残差与平方惩罚 | `src/qubo.py` | `node_flow_residuals`, `flow_penalty` |
| QUBO 展开 | `src/qubo.py` | `build_qubo` |
| QUBO→Ising | `src/qubo.py` | `qubo_to_ising` |
| 16,384 状态枚举 | `src/qubo.py` | `enumerate_state_space` |
| Penalty-X statevector | `src/qaoa.py` | `qaoa_state`, `apply_x_mixer` |
| Global Grover rank-one mixer | `src/qaoa.py` | `GlobalGroverMixer.evolve` |
| 20 条可行路径逻辑基 | `src/feasible_qaoa.py` | `build_feasible_route_basis` |
| Path-exchange mixer | `src/feasible_qaoa.py` | `build_logical_path_exchange_mixer` |
| Feasible Grover mixer | `src/feasible_experiments.py` | `build_grover_feasible_mixer` |
| incumbent threshold | `src/feasible_experiments.py` | `build_incumbent_threshold`, `threshold_phase_values` |
| 最终 phase→mixer 演化 | `src/feasible_experiments.py` | `simulate_final_improvement` |
| COBYLA 预算与 best-retention | `src/feasible_experiments.py` | `optimize_final_variant` |
| 最终课程入口 | `src/experiments/course_final.py` | `run_course_final` |
| 基础 Penalty-X 入口 | `src/main.py` | `run` |

最短的阅读顺序是：`data/graph.json` → `src/main.py` → `src/graph.py` → `src/qubo.py` → `src/qaoa.py` → `src/feasible_qaoa.py` → `src/feasible_experiments.py`。

## 13.1 两个关键代码片段对应的数学含义

Penalty-X 的 cost layer：

```python
state *= np.exp(-1j * gamma * cost_diagonal)
```

逐元素乘相位，模长不变，对应 aₓ→aₓe^(−iγEₓ)。

Grover mixer 的秩一更新：

```python
overlap = np.vdot(uniform, state)
state = state + (np.exp(-1j * beta) - 1) * uniform * overlap
```

对应 |ψ'⟩=|ψ⟩+(e^(−iβ)−1)|s⟩⟨s|ψ⟩，无需构造稠密 projector exponential。

# 14. 科学与软件验证

项目采用“模型验证、动力学验证、结果完整性验证”三层检查。

## 14.1 模型验证

- 图固定为 7 节点、14 条正整数权边，source=0、target=6，qubit index 完整覆盖 0..13。
- NetworkX 最短路与确定性简单路径穷举一致，得到唯一成本 10 路径。
- 对全部 16,384 状态验证 QUBO 与 Ising basis energy 完全相等。
- A=6 时 QUBO ground state 是唯一最优有效路径；最低无效状态能量为 12。
- 零流残差状态数量与独立 decoder-valid 状态数量均为 20。

## 14.2 量子演化验证

- 初态、每个 cost checkpoint、每个 mixer checkpoint 和最终态的范数均为 1（浮点容差内）。
- cost step 最大概率变化低于 3×10⁻²⁰，raw ⟨H_C⟩ 变化低于 1.5×10⁻¹⁴。
- 小维度 Global Grover 实现与显式稠密矩阵指数在 10⁻¹² 内一致。
- 代码禁止构造 16,384×16,384 的 Grover 稠密矩阵，实际使用线性秩一更新。
- 新的 trace 路径与保留的历史模拟器振幅误差至多约 5.1×10⁻¹⁸。

## 14.3 实验完整性

- seeds、深度、参数范围、COBYLA 预算和停止规则在配置中显式固定。
- 达到预算上限的运行仍然保留，不能因结果不理想而重跑替换。
- 结果目录含 config、raw run、aggregate summary、validation 和 hash manifests。
- 最终三个 p=3 seeds 全部报告，包含表现较低的 seed 2602。

老师如果问“为什么相信结果不是画图脚本造出来的”，可以回答：图是从保存的 raw probability 和 summary 表再生成；模型恒等式、归一化、路径参考和结果哈希都有独立检查。

# 15. 可以说什么、不能说什么

## 15.1 有证据支持的结论

- 在同一 16,384 维表示、同一初态和同一 Penalty-QUBO 下，改变 mixer 会改变 p=2 的概率流和最终指标。
- cost phase 自身不改变概率；mixer 是把相位信息变成概率变化的步骤。
- 把逻辑支撑限制为 20 条可行路径，会结构性保证 p_feas=1，并把均匀初始 p_opt 提高到 5%。
- 在当前固定实例和有限优化协议下，incumbent-threshold GM-Th-QAOA p=3 得到最高 median p_opt=0.998712。
- 最终方法的概率轨迹与 Grover 型振幅放大机制相符。

## 15.2 不应扩大的结论

- 不说“量子算法超过 Dijkstra”；本项目没有做这种比较，且最短路经典可解。
- 不说“p 越大一定越好”；QAOA 非凸、局部优化且可能 overshoot。
- 不说“Grover mixer 一般都优于 X mixer”；只有一个图、浅深度和固定优化协议。
- 不说“p_feas=1 证明量子优势”；它来自经典枚举后的逻辑基定义。
- 不说“只需要 5 个物理 qubit 就能高效实现”；逻辑算符到硬件门分解尚未完成。
- 不说“99.87% 是统计显著结果”；只有三个确定性 ideal-simulation seeds。

> 最稳妥的结论句：**这是一个 teaching-scale mechanism demonstration。它清楚展示了表示、phase 设计和 mixer 几何如何共同决定浅层 QAOA 的概率流，但不外推到规模优势或硬件优势。**

<!-- pagebreak -->

# 16. 15 分钟汇报口播稿

## 16.1 0:00–1:30　问题与目标

“我们研究的是一个固定的有向加权路由实例：7 个节点、14 条边，从 0 到 6。经典最短路是 0→1→2→4→5→6，成本 10，而且是唯一最优。我们的目的不是击败 Dijkstra，而是用这个可完全验证的小实例理解 QAOA 的算法机制。”

指图 1：“每条边会对应一个 bit 或 qubit，因此全空间有 2¹⁴=16,384 个边选择，但只有 20 个选择是真正的有效路径。”

## 16.2 1:30–4:00　QUBO 建模

“定义 xₑ 表示边是否被选，路径成本是 Σwₑxₑ。仅最小化它会选空集，所以我们对每个节点写流残差：outgoing−incoming−supply。source 的 supply 是 +1，target 是 −1，中间节点是 0。”

在白板写：

EQ: fᵥ=out−in−bᵥ，　Q₆(x)=Σₑwₑxₑ+6Σᵥfᵥ²。

“A=6 不是一般理论最优值，而是对这个固定实例穷举验证过：唯一 ground state 是成本 10 路径，最低无效能量是 12。再用 x=(1−Z)/2 把 QUBO 映射成 Ising Hamiltonian，全部 16,384 个状态的能量逐一相等。”

## 16.3 4:00–6:30　一层 QAOA 到底做什么

“一层先 cost 后 mixer。Cost 对每个振幅乘 e^(−iγE)，所以它只转相位，不改模长，概率和成本期望在这个 checkpoint 都不变。”

强调：

> “Cost Hamiltonian 把目标函数信息写入相对相位；mixer 把相位差转化成干涉和概率重分配。”

“因此真正看到 p_opt、p_feas 或能量变化是在 mixer 之后；而且不要求每层单调下降，第一层可以先变差，第二层再通过干涉变好。”

## 16.4 6:30–9:30　三种搜索几何

“Penalty-X 在全部 16,384 个 bitstring 上用单 bit 翻转做局部超立方体游走。Global-Grover 保持同一表示、同一 cost diagonal 和同一初态，只把 mixer 换成均匀全空间 projector，所以这组对比隔离了 mixer 的影响。”

“Feasible-Grover 先经典枚举 20 条路径，每条路径作为一个逻辑基态，再在可行均匀态上做 Grover projector mixing。因此 p_feas 永远等于 1，初始 p_opt 也从 1/16,384 变成 1/20。这个提升的一部分来自表示，而不能全归功于 mixer。”

## 16.5 9:30–12:00　最终 GM-Th-QAOA

“最终算法不用最优标签构造目标，而从确定性贪心路径成本 11 出发，标记所有成本严格低于 11 的路径。这个实例只有一个被标记态。初态是 20 条路径均匀叠加，phase 只给标记态相位，Grover mixer 做全局可行域干涉，COBYLA 最大化标记集合总概率，也就是 BSP。”

“事后才确认被标记态就是唯一最优路径，所以本实例 BSP 等于 p_opt。p=1、2、3 的中位结果为 39.2%、81.608%、99.8712%，与 Grover 振幅放大非常一致。”

## 16.6 12:00–14:00　结果与可信度

展示 p=1/p=2 表：“全空间 p=2 Global-Grover 比相同 QUBO 的 Penalty-X 出现更明显的概率流；可行域 Grover p=2 达到 17.09% p_opt。最终 threshold 版本 p=3 三个 seeds 分别是 99.98%、81.60%、99.87%，中位 99.8712%，p_feas 全部为 1。”

“三个最终运行都碰到 150 次预算上限，所以它们是有限预算最好值，不是全局收敛证明。我们保留了较差 seed 2602，没有重跑替换。”

## 16.7 14:00–15:00　结论与边界

“核心认识有三点：第一，cost phase 是信息编码，不直接集中概率；第二，mixer 几何决定相位信息怎样变成概率；第三，限制到可行空间既保证可行性，也改变初始基线。”

“最后，这只是一个 20 状态的理想模拟。路径已被经典完全枚举，最优值其实可由 argmin 直接看到；我们不做量子优势、可扩展性或硬件效率主张。”

# 17. 老师可能追问的 18 个问题

## 17.1 为什么不用 Dijkstra？

答：如果目标只是求这个最短路，当然应该用 Dijkstra。本项目选择 routing 作为小型、直观、可穷举验证的载体，研究 QUBO 编码和 QAOA 概率动力学，不声称实际求解效率超过经典算法。

## 17.2 为什么空集不是答案？

答：仅边权成本下空集成本为 0，所以必须加入从 source 送一个单位流到 target 的守恒约束。惩罚项使空集在 source 和 target 各有残差，得到正惩罚。

## 17.3 A=6 怎么来的？

答：它是固定课程配置，并对全部 16,384 状态做实例级验证。A=6 下唯一 ground state 是成本 10 的有效路径，最低无效能量为 12。我们不声称它对任意图最优。

## 17.4 流守恒会不会允许环？

答：一般会，所以流守恒未必等价于单一路径。本图是 DAG，不允许有向环；代码还使用独立 route decoder 做交叉验证，二者在本实例上都恰好给出 20 个有效状态。

## 17.5 QUBO 为什么能变成 Ising？

答：计算基上的 Z 本征值是 ±1，用 x=(1−Z)/2 就能把二进制一次、二次项变成 I、Z 和 ZZ 项。代码逐状态验证映射前后能量误差为 0。

## 17.6 Cost layer 为什么不降低成本？

答：它只给振幅乘单位模长相位 e^(−iγE)，所以概率不变；同时它与 H_C 对易，所以自己的期望能量也不变。能量变化发生在不对易的 mixer 之后。

## 17.7 Mixer 为什么能利用成本信息？

答：cost layer 让不同能量状态具有不同相位。mixer 线性组合这些复振幅，相位相近时可相长，相反时可相消，从而把相位差变成概率差。

## 17.8 为什么 mixer 有时让能量升高？

答：单层 mixer 不受单调下降约束，QAOA 优化的是 p 层后的最终 loss。多层过程可以先进入看似更差的分布，再在下一次相位编码和干涉后得到更好的最终分布。

## 17.9 Penalty-X 和 Global-Grover 的比较公平吗？

答：在表示、初态和 cost diagonal 方面是受控的，二者只改变 mixer，因此适合分析 mixer 几何。但只测试了一个图、一个 seed 和 p=1/2，不能做一般排名。

## 17.10 为什么可行域方法 p_feas=1？

答：因为向量的每个坐标从定义上就是一条有效路径，不存在无效坐标。它是结构不变量，不是算法从无效状态中“学会”了可行性。

## 17.11 20 个逻辑态是不是只要 5 个 qubit？

答：维度上 5 个 qubit 可容纳 32 个基态，但还要设计 route encoding、均匀可行态制备、threshold oracle 和 projector mixer 电路。当前实现是 20 维逻辑 operator 模拟，不能直接等同于高效 5-qubit hardware circuit。

## 17.12 阈值算法是不是偷偷用了最优解？

答：构造 marked set 的函数只接收 raw route costs 和 incumbent cost 11，条件是 cost&lt;11，没有最优 route id 输入。最优标签只在优化完成后用于计算 p_opt。不过完整路径与成本确实已被经典枚举，因此整个构造仍是教学型、非可扩展的。

## 17.13 为什么 BSP 恰好等于 p_opt？

答：因为这个实例在成本 11 以下恰好只有一条路径，事后参考表明它就是唯一成本 10 最优路径。若阈值下有多条次优和最优路径，BSP 会是它们的概率总和，不再等于 p_opt。

## 17.14 为什么选择 p=3 而不是 p=4？

答：预设规则先选最高 median p_opt；若候选差距在 0.01 内则优先较低深度。p=3 为 0.998712，p=4 为 0.998429，性能在 0.01 内，所以选更浅的 p=3。

## 17.15 为什么 seed 2602 明显较低？

答：COBYLA 是非凸局部优化，初始参数由 seed 决定，且只有 150 次请求。seed 2602 落在较差吸引域或预算内未走到更好区域。这正是需要报告多 seed 和中位数的原因。

## 17.16 `success=False` 是不是算法失败？

答：不是数值或物理失败。这里停止原因是达到预设 `MAXFUN=150`；所有状态归一、参数合法、loss 有限。按照事先固定的策略，保留预算内见到的最佳 in-bounds 点，但不能声称优化器已收敛。

## 17.17 99.87% 能说明量子优势吗？

答：不能。它是 20 维理想 statevector 中的采样概率；构造前已经经典枚举路径，且没有与包括预处理在内的经典求解成本做优势比较。

## 17.18 下一步最有价值的工作是什么？

答：一是避免完整可行路径枚举，研究可扩展的约束保持 mixer 或 oracle；二是把逻辑算符编译为明确门模型并测资源；三是在更多图、更多阈值密度、噪声和有限 shots 下测试；四是增加 multi-start 或参数迁移，同时预注册预算，避免事后调参。

<!-- pagebreak -->

# 附录 A　全部 20 条可行路径

路径按 `(routing_cost, lexicographic node sequence)` 排序，route id 0 是唯一最优，route id 1 是确定性贪心 incumbent。

| route id | 节点序列 | 成本 | canonical 14-bit |
|---:|---|---:|---|
| 0 | 0→1→2→4→5→6 | 10 | 10010001000101 |
| 1 | 0→1→2→3→4→5→6 | 11 | 10010010010101 |
| 2 | 0→1→2→3→5→6 | 11 | 10010010001001 |
| 3 | 0→1→2→4→6 | 11 | 10010001000010 |
| 4 | 0→2→4→5→6 | 11 | 01000001000101 |
| 5 | 0→1→2→3→4→6 | 12 | 10010010010010 |
| 6 | 0→1→2→5→6 | 12 | 10010000100001 |
| 7 | 0→1→3→4→5→6 | 12 | 10001000010101 |
| 8 | 0→1→3→5→6 | 12 | 10001000001001 |
| 9 | 0→2→3→4→5→6 | 12 | 01000010010101 |
| 10 | 0→2→3→5→6 | 12 | 01000010001001 |
| 11 | 0→2→4→6 | 12 | 01000001000010 |
| 12 | 0→1→3→4→6 | 13 | 10001000010010 |
| 13 | 0→1→4→5→6 | 13 | 10000100000101 |
| 14 | 0→2→3→4→6 | 13 | 01000010010010 |
| 15 | 0→2→5→6 | 13 | 01000000100001 |
| 16 | 0→3→4→5→6 | 13 | 00100000010101 |
| 17 | 0→3→5→6 | 13 | 00100000001001 |
| 18 | 0→1→4→6 | 14 | 10000100000010 |
| 19 | 0→3→4→6 | 14 | 00100000010010 |

# 附录 B　QUBO 系数展开公式

设 B 的第 i 列为 b_i（不要与 supply vector b 混淆），供应向量为 s。则：

EQ: Q_A(x)=wᵀx+A(xᵀBᵀBx−2sᵀBx+sᵀs)。

利用 xᵢ²=xᵢ，可以读出：

EQ: constant = A·sᵀs。

EQ: linear_i = wᵢ + A[(BᵀB)ᵢᵢ − 2(Bᵀs)ᵢ]。

EQ: pair_ij = 2A(BᵀB)ᵢⱼ，　i&lt;j。

当前 source 和 target 的 supply 分别为 +1、−1，所以 sᵀs=2，A=6 时常数项为 12。代码使用 `fractions.Fraction` 保存小型教学模型的精确系数，避免在 QUBO→Ising 恒等验证中引入不必要浮点误差。

# 附录 C　复杂度汇总

| 模块 | 时间复杂度（单层/主要步骤） | 空间复杂度 | 当前规模 |
|---|---|---|---|
| 全状态枚举 | O(2^q·poly(q)) | O(2^q) records | q=14, N=16,384 |
| Full cost phase | O(2^q) | O(2^q) | 16,384 complex amplitudes |
| X mixer | O(q2^q) | O(2^q) | q=14 |
| Global Grover mixer | O(2^q) | O(2^q) | 秩一更新 |
| 简单路径枚举 | 输出敏感；最坏指数级 | O(|F|·path length) | |F|=20 |
| Feasible cost phase | O(|F|) | O(|F|) | 20 amplitudes |
| Feasible Grover mixer | O(|F|) | O(|F|) | 秩一更新 |
| Path-exchange eigendecomposition | O(|F|³) 预处理 | O(|F|²) | 20×20 |
| COBYLA outer loop | evaluations × simulation cost | trace dependent | 100 或 150 requests |

# 附录 D　术语表

| 术语 | 本项目中的含义 |
|---|---|
| QUBO | 二进制变量的无约束二次目标；约束通过惩罚并入目标 |
| Ising Hamiltonian | 由 I、Z、ZZ 项构成、与 QUBO basis energy 等价的量子成本算符 |
| statevector | 所有 basis states 的复振幅向量；本项目进行精确理想模拟 |
| cost / phase separator | 根据成本或 Boolean threshold 给不同状态施加相位 |
| mixer | 混合振幅、把相位差转成概率重分配的 unitary |
| depth p | cost→mixer 交替层数 |
| feasible basis | 每个逻辑坐标本身就是一条有效路径的 20 维表示 |
| incumbent | 已知可行但不一定最优的参考解；本项目贪心成本为 11 |
| BSP | 输出严格优于 incumbent 的总概率 |
| p_opt | 输出最优路径的概率；只用于事后评价 |
| rank-one update | 通过一个 overlap 和向量加法实现 projector unitary，而非构造稠密矩阵 |

# 附录 E　参考与仓库证据

- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028。
- Bärtschi and Eidenbenz, *Grover Mixers for QAOA: Shifting Complexity from Mixer Design to State Preparation*, arXiv:2006.00354。
- Golden, Bärtschi, O’Malley, Eidenbenz, *Threshold-Based Quantum Optimization*, arXiv:2106.13860。
- Feeney, Tate, Eidenbenz, *The Better Solution Probability Metric: Optimizing QAOA to Outperform its Warm-Start Solution*, arXiv:2409.09012。
- 仓库算法总览：`README.md`、`docs/QAOA_DYNAMICS_DEEP_DIVE.md`。
- 最终方法说明：`docs/methods/Q2F_FINAL_IMPROVEMENT.md`。
- 固定最终配置：`configs/course_final.json`。
- p=1/p=2 动力学结果：`results/qaoa_dynamics_deep_dive/v1/final_summary.json`。
- p=3 最终结果：`results/q2f_final_improvement/.../summary/decision.json`。

SMALL: 本报告中的数值均来自当前仓库的固定图、源代码、已保存结果或本地只读复核；图表使用仓库已有输出。文档生成器不会重新优化或覆盖实验结果。
