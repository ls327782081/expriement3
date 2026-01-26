# 基于个性化多模态语义ID的生成式推荐系统研究

> **研究生大论文实验项目**
> **更新日期**: 2026-01-26
> **状态**: 🚀 开发中

---

## 📋 目录

- [项目简介](#项目简介)
- [模型列表](#模型列表)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [文档](#文档)

---

## 项目简介

本项目实现了基于个性化多模态语义ID的生成式推荐系统，包含：

- **创新模型**: PMAT（个性化多模态自适应Tokenizer）、MCRL（多任务对比表示学习）
- **基线模型**: Pctx、PRISM、DGMRec（均已对齐官方实现）
- **完整训练框架**: 基于 `AbstractTrainableModel` 的统一训练流程
- **实验评估**: 推荐性能、ID质量、效率指标

### 核心创新

1. **PMAT**: 个性化语义ID生成
   - 个性化模态注意力权重分配
   - 兴趣感知的动态ID更新机制

2. **MCRL**: ID表征空间优化
   - 三层对比学习（用户偏好、模态内、模态间）
   - 多任务协同优化

---

## 模型列表

### 创新模型

| 模型 | 文件 | 说明 | 状态 |
|------|------|------|------|
| **PMAT** | `our_models/pmat.py` | 个性化多模态自适应Tokenizer | ✅ 已实现 |
| **MCRL** | `our_models/mcrl.py` | 多任务对比表示学习 | ✅ 已实现 |

### 基线模型

| 模型 | 会议 | 年份 | 文件 | 官方对齐 |
|------|------|------|------|----------|
| **Pctx** | SIGIR | 2023 | `baseline_models/pctx_aligned.py` | ✅ 完全对齐 |
| **PRISM** | WWW | 2026 | `baseline_models/prism.py` | ✅ 完全对齐 |
| **DGMRec** | SIGIR | 2025 | `baseline_models/dgmrec.py` | ✅ 完全对齐 |



---

## 项目结构

```
expriement3/
├── our_models/              # 创新模型
│   ├── pmat.py             # PMAT 框架
│   └── mcrl.py             # MCRL 方法
│
├── baseline_models/         # 基线模型
│   ├── pctx_aligned.py     # Pctx (SIGIR 2023)
│   ├── pctx_tokenizer_official.py  # Pctx Tokenizer
│   ├── prism.py            # PRISM (WWW 2026)
│   └── dgmrec.py           # DGMRec (SIGIR 2025)
│
├── data/                    # 数据集
├── checkpoints/             # 模型检查点
├── results/                 # 实验结果
├── logs/                    # 训练日志
│
├── base_model.py            # 抽象基类
├── config.py                # 配置文件
├── data_utils.py            # 数据处理
├── metrics.py               # 评估指标
├── main.py                  # 主入口
├── requirements.txt         # 依赖包
│
└── docs/                    # 文档
    ├── DEVELOPMENT_NOTES.md      # 开发笔记
    ├── MODELS_DOCUMENTATION.md   # 模型文档
    ├── BASELINE_ALIGNMENT_SUMMARY.md  # 基线对齐总结
    ├── BASELINE_MODELS_COMPARISON.md  # 基线对比
    └── PCTX_OFFICIAL_ALIGNMENT_COMPLETE.md  # Pctx对齐
```

---

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd expriement3

# 安装依赖
pip install -r requirements.txt

# Pctx 额外依赖
pip install -r requirements_pctx.txt
```

### 运行实验

```bash
# 快速测试（2 epochs）
python main.py --mode quick --model pmat --epochs 2

# 完整实验（10 epochs）
python main.py --mode full --dataset amazon --epochs 10 --device cuda

# 基线对比
python main.py --mode baseline --epochs 10

# MCRL 实验
python main.py --mode mcrl --epochs 10
```

### 使用模型

```python
from our_models import PMAT, MCRL
from baseline_models import PctxAligned, PRISM, DGMRec

# 初始化模型
pmat = PMAT(config)
prism = PRISM(config)

# 训练
pmat.train(train_dataloader, val_dataloader, num_epochs=10)

# 评估
metrics = pmat.evaluate(val_dataloader)
```

---

## 实验结果

### 基线对比

| 模型 | Recall@10 | NDCG@10 | 参数量 | 状态 |
|------|-----------|---------|--------|------|
| Pctx | - | - | - | ✅ 已实现 |
| PRISM | - | - | 4.04M | ✅ 已实现 |
| DGMRec | - | - | 6.66M | ✅ 已实现 |
| PMAT | - | - | - | ✅ 已实现 |
| MCRL | - | - | - | ✅ 已实现 |

*注: 完整实验结果将在服务器GPU上运行后更新*

---

## 文档

- **[DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md)**: 开发笔记，记录所有重要修改
- **[MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)**: 模型详细文档
- **[BASELINE_ALIGNMENT_SUMMARY.md](BASELINE_ALIGNMENT_SUMMARY.md)**: 基线模型对齐总结
- **[BASELINE_MODELS_COMPARISON.md](BASELINE_MODELS_COMPARISON.md)**: 基线模型对比分析
- **[PCTX_OFFICIAL_ALIGNMENT_COMPLETE.md](PCTX_OFFICIAL_ALIGNMENT_COMPLETE.md)**: Pctx 官方对齐报告

---

## 更新日志

### 2026-01-26 - 项目清理与文档整理 ✅
- ✅ 删除所有临时测试文件
- ✅ 删除临时文件夹（temp_dgmrec, temp_prism）
- ✅ 整理文档，合并重复内容
- ✅ 更新 README.md，使其更加简洁专业
- ✅ 创建统一的开发文档（DEVELOPMENT_NOTES.md）

### 2026-01-26 - 基线模型对齐 ✅
- ✅ PRISM 对齐官方实现
  - MLPReWeighting: 输出维度修正
  - AdaptiveFusionLayer: 专门设计的融合层
  - 添加完整的 forward 方法
- ✅ DGMRec 对齐官方实现
  - 编码器结构对齐
  - 生成器结构对齐
  - mge 方法简化
- ✅ 所有测试通过

### 2026-01-21 - Pctx 官方对齐 ✅
- ✅ PctxAligned 模型：纯 T5 包装
- ✅ PctxTokenizerOfficial：完整实现
- ✅ 所有测试通过

---

## 参考文献

1. **Pctx** (SIGIR 2023): Personalized Context-aware Recommendation
2. **PRISM** (WWW 2026): [GitHub](https://github.com/YutongLi2024/PRISM)
3. **DGMRec** (SIGIR 2025): [GitHub](https://github.com/ptkjw1997/DGMRec)

---

**License**: MIT
**Status**: 🚀 开发中
