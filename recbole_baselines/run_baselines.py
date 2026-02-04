"""
运行RecBole序列推荐基线模型

支持的模型：
- SASRec: Self-Attentive Sequential Recommendation
- BERT4Rec: Bidirectional Encoder Representations from Transformer
- GRU4Rec: Session-based Recommendations with RNN

使用方法：
    # 运行所有基线
    python recbole_baselines/run_baselines.py --dataset Video_Games
    
    # 运行单个模型
    python recbole_baselines/run_baselines.py --dataset Video_Games --model SASRec
    
    # 快速模式（用于测试）
    python recbole_baselines/run_baselines.py --dataset Video_Games --quick_mode
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("RecBoleBaselines")


def run_recbole_model(model_name: str, dataset: str, config_dir: str, dataset_dir: str, logger):
    """
    运行单个RecBole模型
    
    Args:
        model_name: 模型名称 (SASRec, BERT4Rec, GRU4Rec)
        dataset: 数据集名称
        config_dir: 配置文件目录
        dataset_dir: 数据集目录
        logger: 日志记录器
    
    Returns:
        dict: 评估结果
    """
    try:
        from recbole.quick_start import run_recbole
    except ImportError:
        logger.error("RecBole not installed. Please install with: pip install recbole")
        logger.error("Or: pip install recbole==1.2.0")
        return None
    
    logger.info(f"Running {model_name} on {dataset}...")
    
    # 配置文件列表
    config_files = [
        os.path.join(config_dir, 'sequential_config.yaml'),
        os.path.join(config_dir, f'{model_name.lower()}_config.yaml')
    ]
    
    # 检查配置文件是否存在
    for cf in config_files:
        if not os.path.exists(cf):
            logger.error(f"Config file not found: {cf}")
            return None
    
    # 运行模型
    try:
        result = run_recbole(
            model=model_name,
            dataset=dataset,
            config_file_list=config_files,
            config_dict={
                'data_path': dataset_dir,
            }
        )
        
        logger.info(f"{model_name} completed!")
        logger.info(f"Best valid score: {result['best_valid_score']}")
        logger.info(f"Test result: {result['test_result']}")
        
        return {
            'model': model_name,
            'best_valid_score': float(result['best_valid_score']),
            'test_result': {k: float(v) for k, v in result['test_result'].items()}
        }
        
    except Exception as e:
        logger.error(f"Error running {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Run RecBole sequential baselines')
    parser.add_argument('--dataset', type=str, default='Video_Games',
                        help='Dataset name (default: Video_Games)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['SASRec', 'BERT4Rec', 'GRU4Rec'],
                        help='Model to run (default: all)')
    parser.add_argument('--quick_mode', action='store_true',
                        help='Use quick mode (for testing)')
    parser.add_argument('--output_dir', type=str, default='results/recbole',
                        help='Output directory for results')
    
    args = parser.parse_args()
    logger = setup_logger()
    
    # 目录设置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'config')
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # 检查数据集是否存在
    dataset_path = os.path.join(dataset_dir, args.dataset)
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please run convert_to_recbole.py first:")
        logger.error(f"  python recbole_baselines/convert_to_recbole.py --category {args.dataset}")
        return
    
    # 确定要运行的模型
    models = [args.model] if args.model else ['SASRec', 'BERT4Rec', 'GRU4Rec']
    
    # 运行模型并收集结果
    results = []
    for model_name in models:
        result = run_recbole_model(
            model_name=model_name,
            dataset=args.dataset,
            config_dir=config_dir,
            dataset_dir=dataset_dir,
            logger=logger
        )
        if result:
            results.append(result)
    
    # 保存结果
    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(args.output_dir, f'recbole_results_{timestamp}.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # 打印汇总
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for r in results:
            logger.info(f"\n{r['model']}:")
            for metric, value in r['test_result'].items():
                logger.info(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()

