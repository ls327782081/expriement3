# 快速开始指南

**更新日期**: 2026-01-26

---

## 🚀 5分钟快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
pip install -r requirements_pctx.txt  # Pctx额外依赖
```

### 2. 数据准备

数据已包含在 `data/` 文件夹中：
- `Video_Games_small.jsonl` - 小数据集（快速测试）
- `Video_Games.jsonl` - 完整数据集

### 3. 快速测试

```bash
# 快速测试（2 epochs，约5分钟）
python main.py --mode quick --epochs 2 --device cpu
```

### 4. 完整实验

```bash
# 在GPU上运行完整实验（约15小时）
python main.py --mode complete --epochs 10 --device cuda
```

---

## 📊 实验模式

### 基础实验

```bash
# 快速测试
python main.py --mode quick --epochs 2

# 完整训练
python main.py --mode full --epochs 10

# 基线对比
python main.py --mode baseline --epochs 10
```

### ⭐ 推荐模型实验（新增）

```bash
# PMAT推荐模型实验（使用真实用户历史）
python main.py --mode pmat_rec --epochs 10

# MCRL推荐模型实验（使用真实用户历史）
python main.py --mode mcrl_rec --epochs 10

# 快速测试模式
python main.py --mode pmat_rec --dataset mock
python main.py --mode mcrl_rec --dataset mock
```

### 高级实验

```bash
# 消融实验
python main.py --mode ablation --epochs 10

# 效率分析
python main.py --mode efficiency

# 鲁棒性分析
python main.py --mode robustness --epochs 10

# 多数据集验证
python main.py --mode multi_dataset --epochs 10
```

### 完整实验

```bash
# 运行所有实验（推荐在服务器GPU上运行）
python main.py --mode complete --epochs 10 --device cuda
```

---

## 📁 输出文件

### 结果文件
```
results/
├── baseline_results.csv              # 基线对比结果
├── baseline_results.json
├── pmat_rec_experiment_results.json  # ⭐ PMAT推荐模型结果
├── mcrl_rec_experiment_results.json  # ⭐ MCRL推荐模型结果
├── PMAT_ablation_results.json        # PMAT消融结果
├── MCRL_ablation_results.json        # MCRL消融结果
├── efficiency_analysis_results.json  # 效率分析结果
├── PMAT_robustness_results.json      # 鲁棒性分析结果
└── multi_dataset_results.json        # 多数据集结果
```

### 可视化图表
```
results/
├── baseline_top10_metrics.png        # 基线对比图
├── PMAT_ablation.png                 # PMAT消融图
├── MCRL_ablation.png                 # MCRL消融图
├── efficiency_analysis.png           # 效率分析图
├── PMAT_robustness.png               # 鲁棒性曲线
└── multi_dataset_comparison.png      # 多数据集对比图
```

### 模型检查点
```
checkpoints/
├── PMAT_best.pth                     # PMAT最佳模型
├── baseline_PctxAligned_best.pth     # Pctx最佳模型
├── baseline_PRISM_best.pth           # PRISM最佳模型
└── baseline_DGMRec_best.pth          # DGMRec最佳模型
```

---

## 🔧 常用命令

### 修改配置

```bash
# 修改数据集
python main.py --mode quick --category Beauty

# 修改batch size
python main.py --mode quick --batch-size 64

# 修改学习率
python main.py --mode quick --lr 0.001

# 使用GPU
python main.py --mode quick --device cuda
```

### 查看帮助

```bash
python main.py --help
```

---

## 📚 更多文档

- **[README.md](README.md)**: 项目主文档
- **[docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**: 详细实验指南
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**: 项目结构说明
- **[docs/MODELS_DOCUMENTATION.md](docs/MODELS_DOCUMENTATION.md)**: 模型文档

---

## ❓ 常见问题

### Q1: 如何快速测试？
```bash
python main.py --mode quick --epochs 2 --device cpu
```

### Q2: 如何运行完整实验？
```bash
python main.py --mode complete --epochs 10 --device cuda
```

### Q3: 如何只运行某个模型？
修改 `config.py` 中的 `baseline_models` 列表。

### Q4: 如何添加新数据集？
将数据放在 `data/` 文件夹，修改 `config.py` 中的 `category`。

### Q5: 实验结果在哪里？
所有结果保存在 `results/` 文件夹。

---

## 🎯 推荐工作流

### 本地开发
```bash
# 1. 快速测试代码
python main.py --mode quick --epochs 2 --device cpu

# 2. 验证功能正常
# 检查 results/ 文件夹是否有输出
```

### 服务器实验
```bash
# 1. 上传代码到服务器
# 2. 运行完整实验
python main.py --mode complete --epochs 10 --device cuda

# 3. 下载结果
# 下载 results/ 和 checkpoints/ 文件夹
```

---

**祝实验顺利！** 🚀

