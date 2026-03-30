"""
✅ 完全使用你的 PMAT 数据 & 划分
✅ 生成标准 .inter 文件
✅ 指定时间字段 timestamp
✅ 100% 适配 SASRec, BERT4Rec, GRU4Rec, NARM, STAMP
"""
import os
import pickle
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer
from data_utils import AmazonBooksProcessor


def prepare_pmat_for_recbole(dataset="Video_Games", quick_mode=False):
    # 加载你的数据
    cache_suffix = "_quick" if quick_mode else ""
    cache_file = f"../data/{dataset}{cache_suffix}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
    else:
        processor = AmazonBooksProcessor(dataset, quick_mode=quick_mode)
        data = processor.load_dataset_for_experiment()
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    # 正确构建交互数据
    rows = []
    all_seqs = {**data["train_sequences"], **data["val_sequences"], **data["test_sequences"]}

    for uid, seq_info in all_seqs.items():
        items = seq_info["item_indices"]
        times = seq_info["timestamps"]
        for iid, ts in zip(items, times):
            rows.append({
                "user_id": int(uid),
                "item_id": int(iid),
                "timestamp": float(ts)
            })

    # 写入 RecBole 官方格式 (使用 recbole_baselines/dataset 目录)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset", dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    inter_path = os.path.join(dataset_dir, f"{dataset}.inter")

    with open(inter_path, 'w', encoding='utf-8') as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for r in rows:
            f.write(f"{r['user_id']}\t{r['item_id']}\t{r['timestamp']}\n")

    print(f"✅ 已生成标准 RecBole 文件：{inter_path}")
    return inter_path


def run_model(model_name="SASRec", dataset="Video_Games", quick_mode=False):
    # 1. 生成标准数据文件
    prepare_pmat_for_recbole(dataset, quick_mode)

    # 2. 【关键】指定数据集目录 + 时间字段
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")  # 使用 recbole_baselines/dataset

    config_dict = {
        "dataset": dataset,
        "model": model_name,
        "data_path": dataset_dir,           # ✅ 强制指定数据集路径
        "time_field": "timestamp",         # ✅ 强制指定时间列
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "eval_type": "full",
        "metrics": ["Hit", "NDCG", "MRR"],
        "topk": [5, 10, 20],
        "max_seq_len": 50,
        "train_batch_size": 256,
        "eval_batch_size": 256,
        "learning_rate": 0.001,
        "epochs": 50,
        "loss_type": "CE",
        "train_neg_sample_args": None,
        "user_min_interactions": 0,
        "item_min_interactions": 0,
    }

    # 3. RecBole 官方标准流程
    config = Config(model=model_name, dataset=dataset, config_dict=config_dict)
    init_logger(config)

    dataset = create_dataset(config)
    train_data, val_data, test_data = data_preparation(config, dataset)

    model = get_model(model_name)(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], model_name)(config, model)

    trainer.fit(train_data, valid_data=val_data)
    res = trainer.evaluate(test_data)
    print(f"🎉 {model_name} 结果：", res)
    return res


if __name__ == "__main__":
    models = ["SASRec", "BERT4Rec", "GRU4Rec", "NARM", "STAMP"]
    for m in models:
        print(f"\n\n========== Training {m} ==========")
        run_model(model_name=m, dataset="Video_Games", quick_mode=True)