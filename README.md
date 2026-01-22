# 基于个性化多模态语义ID的生成式推荐系统研究

> **研究生大论文实验项目**
> **更新日期**: 2026-01-22
> **方案版本**: 方案A（保守修改版）

---

## 📋 目录

- [研究背景](#研究背景)
- [创新点概述](#创新点概述)
- [创新点1: PMAT框架](#创新点1-pmat框架)
- [创新点2: MCRL方法](#创新点2-mcrl方法)
- [理论分析](#理论分析)
- [实验设计](#实验设计)
- [项目结构](#项目结构)
- [快速开始](#快速开始)

---

## 🎯 研究背景

### 问题定义

多模态生成式推荐系统通过语义ID（Semantic ID）将物品表示为离散token序列，利用大语言模型进行推荐。然而，现有方法存在以下核心问题：

1. **模态融合同质化**：采用全局统一的模态权重，忽略用户个性化偏好差异
2. **静态编码局限**：无法适应用户兴趣的动态变化
3. **ID表征质量不足**：缺乏针对性的表征空间优化

### 研究目标

构建**个性化、动态、高质量**的多模态语义ID生成与优化框架，提升生成式推荐系统的性能。

---

## 💡 创新点概述

本研究提出两个相互协同的创新点：

| 创新点 | 名称 | 核心贡献 | 技术路线 |
|--------|------|---------|---------|
| **创新点1** | PMAT | 个性化语义ID生成 | 个性化模态融合 + 动态更新 |
| **创新点2** | MCRL | ID表征空间优化 | 多任务对比学习 |

**技术链路**：
```
用户历史 + 物品多模态特征
         ↓
    PMAT生成个性化ID
         ↓
    MCRL优化ID表征空间
         ↓
    高效检索 + 精准推荐
```

---

## 🔬 创新点1: PMAT框架

### 1.1 研究动机

**现有方法的痛点**：

- **FusID/MMQ**：全局共享模态权重，无法区分"视觉敏感型"和"文本敏感型"用户
- **静态编码**：物品ID固定，无法适应用户兴趣从"科幻电影"到"喜剧电影"的漂移

**我们的洞察**：
> 同一物品对不同用户应该有不同的语义ID表示，且ID应随用户兴趣动态演化

### 1.2 核心思想

**PMAT (Personalized Multimodal Adaptive Tokenizer)** 框架包含两个关键模块：

#### 模块1：个性化模态注意力权重分配

```python
# 伪代码示例
user_modal_weights = UserModalAttention(user_history)
# 输出: {α_visual, α_text, α_audio} 个性化权重

fused_features = α_visual * f_visual + α_text * f_text + α_audio * f_audio
```

**创新点**：
- 在**语义ID生成阶段**就融入个性化（vs. PRISM/AMMRM在特征融合阶段）
- 用户-物品级个性化（vs. 用户级个性化）

#### 模块2：兴趣感知的动态语义ID更新机制

```python
# 检测兴趣漂移
drift_score = KL_divergence(short_term_interest, long_term_interest)

if drift_score > threshold:
    # 增量更新ID
    new_id = (1 - λ) * old_id + λ * new_features
```

**创新点**：
- 基于兴趣漂移检测的触发机制
- 语义一致性约束，避免ID剧烈变化

### 1.3 与最新工作的差异

| 维度 | PRISM (2025) | AMMRM (2025) | **PMAT (Ours)** |
|------|--------------|--------------|-----------------|
| 作用阶段 | 特征融合 | 特征融合 | **语义ID生成** |
| 输出 | 连续特征向量 | 连续特征向量 | **离散ID序列** |
| 个性化粒度 | 用户级 | 用户级 | **用户-物品级** |
| 动态更新 | ✗ | ✗ | **✓** |
| 理论保证 | ✗ | ✗ | **✓ (定理1-3)** |

### 1.4 理论贡献

**定理1（语义漂移界）**：
```
D(ID_i^u, ID_i^v) ≤ C · ||α_u - α_v||_2 · sim(u, v)
```

**推论**：相似用户的同一物品ID应该接近，不同偏好用户的ID应该有差异。

详见 [theory_analysis.md](theory_analysis.md)

---

## 🆕 创新点2: MCRL方法

### 2.1 研究动机

**问题**：PMAT生成的个性化ID如何进一步优化表征空间，提升检索效率和精度？

**现有对比学习方法的局限**：
- **LETTER (2024)**：仅用于对齐CF信号，单一对比损失
- **CoFiRec (2025)**：关注层级token化，未优化表征空间

### 2.2 核心思想

**MCRL (Multi-task Contrastive Representation Learning)** 通过**三层对比学习**优化ID表征空间：

```
┌─────────────────────────────────┐
│  Layer 1: 用户偏好对比学习       │
│  - 拉近相似用户偏好的物品ID      │
│  - 推远不同偏好用户的物品ID      │
├─────────────────────────────────┤
│  Layer 2: 模态内对比学习         │
│  - 增强单模态判别性              │
│  - 同一模态相似物品ID接近        │
├─────────────────────────────────┤
│  Layer 3: 模态间对比学习         │
│  - 对齐不同模态的互补信息        │
│  - 最大化模态间互信息            │
└─────────────────────────────────┘
```

### 2.3 创新价值

| 维度 | LETTER (2024) | CoFiRec (2025) | **MCRL (Ours)** |
|------|---------------|----------------|-----------------|
| 核心思想 | CF信号对齐 | 层级token化 | **多任务对比学习** |
| 对比层数 | 1层 | 1层 | **3层协同** |
| 优化目标 | 特征对齐 | 生成效率 | **表征空间结构** |
| 个性化 | ✗ | ✗ | **✓** |

### 2.4 理论贡献

**定理5（多任务协同）**：三层对比学习提供互补的监督信号，优于单一对比学习。

---

## 📐 理论分析

### 核心定理

| 定理 | 内容 | 意义 |
|------|------|------|
| **定理1** | 语义漂移界 | 个性化权重减少ID漂移 |
| **定理2** | 更新稳定性 | 动态更新的收敛保证 |
| **定理3** | 一致性保证 | 平衡稳定性与适应性 |
| **定理4** | 判别性界 | 对比学习提升ID判别性 |
| **定理5** | 多任务协同 | 三层对比优于单层 |
| **定理6** | 检索复杂度 | 语义ID检索效率提升 |

详细证明见 [theory_analysis.md](theory_analysis.md)

---

## 🧪 实验设计

### 实验架构

```
实验体系
├── 创新点1实验（PMAT）
│   ├── 对比实验：vs. FusID, MMQ, PRISM, AMMRM
│   ├── 消融实验：个性化权重 / 动态更新
│   └── ID质量评估：唯一性 / 一致性 / 区分度
│
├── 创新点2实验（MCRL）
│   ├── 对比实验：vs. LETTER, CoFiRec
│   ├── 消融实验：三层对比学习的贡献
│   └── 效率分析：检索延迟 / 吞吐量
│
└── 整体系统实验（PMAT + MCRL）
    ├── 端到端性能：Recall, NDCG, MRR
    ├── 全基线对比：8个基线模型
    └── 案例分析：可视化ID空间
```

### 数据集

| 数据集 | 物品数 | 用户数 | 模态 | 用途 |
|--------|--------|--------|------|------|
| Amazon-Beauty | 12,101 | 22,363 | 文本+视觉 | 主实验 |
| MM-Rec | 8,000 | 15,000 | 文本+视觉+音频 | 多模态验证 |
| MovieLens-25M | 62,138 | 162,541 | 文本+视觉 | 大规模验证 |

### 基线模型（2025年最新）

**经典基线**：
- Pctx (上下文感知)
- MMQ (多模态量化)
- FusID (融合ID)
- RPG (循环个性化生成)

**2025年最新基线（GitHub源码适配）**：
- **PRISM** (WWW 2026) - 个性化多模态融合 ✅ **已实现并测试**
  - 来源: https://github.com/YutongLi2024/PRISM
  - 参数量: 4.04M
  - 核心: 交互专家层 + 自适应融合
  - 状态: 已完成快速测试（2 epochs）

- **DGMRec** (SIGIR 2025) - 解耦和生成模态 ✅ **已实现并测试**
  - 来源: https://github.com/ptkjw1997/DGMRec
  - 参数量: 6.66M
  - 核心: 模态解耦 + 缺失模态生成
  - 状态: 已完成快速测试（2 epochs）

- **REARM** (MM 2025) - 关系增强自适应表示 ✅ **已实现并测试**
  - 来源: https://github.com/MrShouxingMa/REARM
  - 参数量: 6.27M
  - 核心: 元网络学习 + 同态关系 + 正交约束
  - 状态: 已完成快速测试（2 epochs）

**待实现基线（可选）**：
- AMMRM (自适应多模态推荐) - 优先级：中
- CoFiRec (粗细粒度token化) - 优先级：低
- LETTER (可学习token化) - 优先级：低

**说明**：已实现3个2025年最新基线，足够进行对比实验。其他基线可根据需要选择性实现。

### 评估指标

**推荐性能**：
- Recall@K, NDCG@K, MRR, Hit Rate

**ID质量**（新增）：
- ID唯一性 (ID Uniqueness)
- 语义一致性 (Semantic Consistency)
- 个性化区分度 (Personalization Discrimination)

**效率指标**：
- ID生成时间
- 检索延迟
- 内存占用

### 预期结果

| 指标 | FusID | PRISM | AMMRM | PMAT | **PMAT+MCRL** |
|------|-------|-------|-------|------|---------------|
| Recall@10 | 0.245 | 0.268 | 0.271 | 0.285 | **0.302** |
| NDCG@10 | 0.182 | 0.195 | 0.198 | 0.208 | **0.221** |
| ID Uniqueness | 0.78 | 0.81 | 0.82 | 0.89 | **0.91** |
| Retrieval Latency (ms) | 45 | 48 | 47 | 43 | **38** |

---

## 📁 项目结构

```
experiment3/
├── data/                    # 数据集目录
├── checkpoints/             # 模型检查点
├── results/                 # 实验结果
├── logs/                    # 训练日志
│
├── our_models/              # 🌟 我们的创新模型
│   ├── __init__.py
│   ├── pmat.py             # 创新点1：PMAT框架
│   └── mcrl.py             # 创新点2：MCRL方法
│
├── baseline_models/         # 📊 基线模型（对比工作）
│   ├── __init__.py
│   ├── pctx.py             # 上下文感知推荐
│   ├── mmq.py              # 多模态量化
│   ├── fusid.py            # 融合语义ID
│   ├── rpg.py              # 循环个性化生成
│   ├── prism.py            # 待实现：2025基线
│   ├── ammrm.py            # 待实现：2025基线
│   └── letter.py           # 待实现：2024基线
│
├── config.py                # 配置文件
├── data_utils.py            # 数据处理
├── metrics.py               # 评估指标
├── main.py                  # 主实验入口
├── util.py                  # 工具函数
├── models.py                # 原有模型（兼容）
├── theory_analysis.md       # 理论分析文档
├── requirements.txt         # 依赖包
└── README.md                # 本文件
```

### 📂 目录说明

| 目录/文件 | 说明 |
|----------|------|
| **our_models/** | 🌟 **我们的创新贡献**，包含PMAT和MCRL |
| **baseline_models/** | 📊 **基线模型**，用于对比实验 |
| **data/** | 数据集存储（Amazon-Beauty, MM-Rec等） |
| **checkpoints/** | 训练过程中的模型检查点 |
| **results/** | 实验结果（CSV、JSON、图表） |
| **theory_analysis.md** | 📐 理论分析与数学证明 |

---

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n pmat_mcrl python=3.9
conda activate pmat_mcrl

# 安装依赖
pip install -r requirements.txt
```

### 运行实验

#### 1. 本地快速测试（推荐）

```bash
# 快速测试模式 - 2个epoch，适合本地调试
python main.py --mode quick --dataset mock --epochs 2 --batch_size 64

# 指定模型测试
python main.py --mode quick --model pmat --epochs 2
```

#### 2. 服务器完整实验

```bash
# 完整实验模式 - 运行所有实验（基线+消融+超参）
python main.py --mode full --dataset amazon --epochs 10 --batch_size 32 --device cuda

# 仅运行基线对比实验
python main.py --mode baseline --dataset amazon --epochs 10 --device cuda

# 仅运行消融实验
python main.py --mode ablation --dataset amazon --epochs 10 --device cuda

# 仅运行超参数搜索
python main.py --mode hyper --dataset amazon --device cuda

# 🆕 运行MCRL完整实验（三层对比学习）
python main.py --mode mcrl --dataset amazon --epochs 10 --batch_size 32 --device cuda
```

#### 3. 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `quick` | 实验模式: `quick`(快速测试), `full`(完整实验), `baseline`(基线对比), `ablation`(消融实验), `hyper`(超参搜索), **`mcrl`(MCRL完整实验)** 🆕 |
| `--dataset` | str | `mock` | 数据集: `mock`(模拟数据), `amazon`(Amazon Books), `movielens`(MovieLens-25M) |
| `--model` | str | `pmat` | 模型: `pmat`, `mcrl`, `pmat_mcrl`, `pctx`, `mmq`, `fusid`, `rpg`, `prism`, `dgmrec`, `rearm` |
| `--epochs` | int | 自动 | 训练轮数 (quick默认2, full默认10) |
| `--batch_size` | int | 自动 | 批大小 (quick默认64, full默认32) |
| `--lr` | float | `1e-4` | 学习率 |
| `--device` | str | 自动 | 设备: `cpu`, `cuda` |
| `--seed` | int | `42` | 随机种子 |
| `--save_dir` | str | `./results` | 结果保存目录 |
| `--log_dir` | str | `./logs` | 日志保存目录 |

#### 4. 使用示例

```bash
# 示例1: 本地CPU快速测试PMAT
python main.py --mode quick --model pmat --epochs 2 --device cpu

# 示例2: 服务器GPU完整实验
python main.py --mode full --dataset amazon --epochs 10 --batch_size 32 --device cuda

# 示例3: 🆕 运行MCRL完整实验（推荐）
python main.py --mode mcrl --dataset amazon --epochs 10 --batch_size 32 --device cuda

# 示例4: 测试MCRL模型（快速版本）
python main.py --mode quick --model mcrl --epochs 3 --lr 5e-5

# 示例5: 自定义配置
python main.py --mode baseline --dataset movielens --epochs 15 --batch_size 128 --lr 2e-4 --seed 2024
```

#### 5. MCRL训练流程说明 🆕

MCRL完整实验包含以下步骤：

1. **PMAT单独训练**（作为基线）
   - 训练个性化语义ID生成器
   - 保存检查点：`checkpoints/MCRL_Exp_PMAT_only_*.pth`

2. **PMAT+MCRL联合训练**
   - 端到端训练三层对比学习
   - Layer 1: 用户偏好对比学习
   - Layer 2: 模态内对比学习
   - Layer 3: 模态间对比学习
   - 保存检查点：`checkpoints/MCRL_Exp_PMAT_MCRL_joint_*.pth`

3. **评估与对比**
   - 对比PMAT单独 vs PMAT+MCRL联合
   - 保存结果：`results/mcrl_experiment_*.json`

**训练监控**：
```bash
# 查看训练日志
tail -f logs/experiment.log

# 查看损失分解
# - PMAT损失: ID生成损失 + 语义一致性损失
# - MCRL损失: 用户偏好损失 + 模态内损失 + 模态间损失
# - 总损失: λ_pmat * L_pmat + λ_mcrl * L_mcrl
```

### 导入示例

```python
# 导入我们的创新模型
from our_models.pmat import PMAT
from our_models.mcrl import MCRL, PMATWithMCRL

# 导入经典基线模型
from baseline_models import Pctx, MMQ, FusID, RPG

# 导入2025最新基线模型（GitHub源码适配）
from baseline_models import PRISM, DGMRec, REARM

# 初始化模型
pmat = PMAT(config)
mcrl = MCRL(config)
joint_model = PMATWithMCRL(config)

# 初始化基线模型
prism = PRISM(config)
dgmrec = DGMRec(config)
rearm = REARM(config)
```

### 测试基线模型

```bash
# 测试所有基线模型（PRISM, DGMRec, REARM）
python test_baseline_models.py

# 测试单个基线模型
python main.py --mode quick --model prism --epochs 2
python main.py --mode quick --model dgmrec --epochs 2
python main.py --mode quick --model rearm --epochs 2
```

---

## 📊 实验时间规划（5周）

| 周次 | 任务 | 产出 |
|------|------|------|
| Week 1 | PMAT代码实现 + PRISM/AMMRM复现 | 可运行代码 |
| Week 2 | 创新点1实验 + 理论分析 | 实验结果 + 定理证明 |
| Week 3 | MCRL代码实现 + 三层对比学习实验 | 完整MCRL模块 |
| Week 4 | 整体系统实验 + 消融实验 | 全部实验数据 |
| Week 5 | 论文撰写 + 结果可视化 | 完整论文初稿 |

---

## 📚 参考文献

1. **PRISM (2025)**: Adaptive fusion methods for multimodal recommendation
2. **AMMRM (2025)**: User Preference Adaptive Fusion Module
3. **CoFiRec (2025)**: Coarse-to-Fine Tokenization for Generative Recommendation
4. **LETTER (2024)**: Learnable Item Tokenization for Generative Recommendation
5. **GRID (2025)**: Generative Recommendation with Semantic IDs: A Practitioner's Handbook

---

## 📝 更新日志

### 2026-01-22 (下午) - MCRL完整训练流程实现 ✅
- ✅ **实现MCRL完整训练流程**
  - 新增 `train_mcrl_model()` 函数：支持三层对比学习的端到端训练
  - 新增 `run_mcrl_experiment()` 函数：完整的MCRL实验流程
  - 新增 `sample_positive_negative_ids()` 函数：正负样本采样
  - 支持PMAT单独训练、PMAT+MCRL联合训练
- ✅ **修复PMATWithMCRL模型**
  - 修复forward方法参数兼容性问题
  - 修复PMAT输出缺少modal_features的问题
  - 添加MCRL参数可选支持（向后兼容）
- ✅ **完整测试验证**
  - 创建 `test_mcrl_training.py` 测试脚本
  - 4/4 测试全部通过：
    - ✓ MCRL模型初始化 (1.05M参数)
    - ✓ MCRL前向传播（三层对比损失正常计算）
    - ✓ MCRL反向传播（梯度计算正常）
    - ✓ PMAT+MCRL联合模型（6.15M参数，端到端训练）
- ✅ **命令行支持**
  - 新增 `--mode mcrl` 参数：运行MCRL完整实验
  - 使用方法：`python main.py --mode mcrl --epochs 10`

**技术细节**：
- MCRL三层对比学习损失：
  - Layer 1: 用户偏好对比学习 (User Preference CL)
  - Layer 2: 模态内对比学习 (Intra-modal CL)
  - Layer 3: 模态间对比学习 (Inter-modal CL)
- 联合训练损失：`L_total = λ_pmat * L_pmat + λ_mcrl * L_mcrl`
- 支持混合精度训练、梯度裁剪、学习率调度

### 2026-01-22 (上午) - 基线模型实现 + 对比实验完成
- ✅ **实现3个2025年最新基线模型**（从GitHub官方源码适配）
  - PRISM (WWW 2026) - 4.04M参数
  - DGMRec (SIGIR 2025) - 6.66M参数
  - REARM (MM 2025) - 6.27M参数
- ✅ **完成基线对比实验**（快速测试版本）
  - 运行8个模型：Pctx, MMQ, FusID, RPG, PRISM, DGMRec, REARM, PMAT
  - 结果保存：results/baseline_results.csv, baseline_results.json
  - 生成对比图表：baseline_top10_metrics.png
- ✅ **理论分析文档完整**
  - theory_analysis.md (351行)
  - 包含PMAT和MCRL的完整理论证明
- ✅ **创建基线模型总结文档**
  - BASELINE_MODELS_SUMMARY.md
  - 详细记录3个基线模型的实现和测试情况

### 2026-01-21 - 方案A实施 + 命令行参数支持
- ✅ 创建PMAT完整代码框架 (pmat.py)
- ✅ 创建MCRL完整代码框架 (mcrl.py)
- ✅ 更新配置文件，增加2025年基线
- ✅ 完成理论分析文档 (theory_analysis.md)
- ✅ 项目结构重组 (our_models/ + baseline_models/)
- ✅ 修复PMAT输出格式问题（logits vs 离散ID）
- ✅ **新增命令行参数支持**：支持quick/full模式切换
- ✅ 更新README，添加详细的命令行参数说明

### 下一步计划

**高优先级**：
- [ ] **运行MCRL完整实验**（10+ epochs，在服务器GPU上）
  ```bash
  python main.py --mode mcrl --epochs 10 --batch_size 32
  ```
- [ ] **运行完整的基线对比实验**（10+ epochs，在服务器GPU上）
  ```bash
  python main.py --mode baseline --epochs 10
  ```
- [ ] **运行消融实验**
  - PMAT消融：个性化权重 vs 动态更新
  - MCRL消融：三层对比学习的贡献
  ```bash
  python main.py --mode ablation --epochs 10
  ```

**中优先级**：
- [ ] **设计并实现可视化方案**
  - ID空间可视化（t-SNE/UMAP）
  - 模态权重分布可视化
  - 性能对比图表
  - MCRL三层损失曲线
- [ ] **运行超参数搜索实验**
  ```bash
  python main.py --mode hyper
  ```

**低优先级**：
- [ ] **实现AMMRM基线**（可选，已有3个2025基线）
- [ ] **撰写实验分析报告**

---

## 💬 联系方式

如有问题，请联系项目负责人。

---

**License**: MIT  
**Status**: 🚧 In Progress

