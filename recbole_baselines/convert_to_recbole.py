"""
将当前数据格式转换为RecBole的atomic files格式

RecBole需要的文件格式：
- Video_Games.inter: user_id:token  item_id:token  timestamp:float
- 配置文件指定序列推荐设置

使用方法：
    python recbole_baselines/convert_to_recbole.py --category Video_Games --quick_mode
"""

import os
import sys
import argparse
import logging
import pickle

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import AmazonBooksProcessor


def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("RecBoleConverter")


def convert_to_recbole_format(category: str, quick_mode: bool = False, output_dir: str = None,
                              min_user_interactions: int = 5, min_item_interactions: int = 5):
    """
    将Amazon数据集转换为RecBole格式

    Args:
        category: 数据集类别 (如 Video_Games)
        quick_mode: 是否使用快速模式
        output_dir: 输出目录
        min_user_interactions: 用户最小交互数（过滤冷启动用户）
        min_item_interactions: 物品最小交互数（过滤冷启动物品）
    """
    logger = setup_logger()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'dataset', category)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Converting {category} dataset to RecBole format...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quick mode: {quick_mode}")
    logger.info(f"Min user interactions: {min_user_interactions}")
    logger.info(f"Min item interactions: {min_item_interactions}")

    # 加载原始数据
    processor = AmazonBooksProcessor(
        category=category,
        quick_mode=quick_mode,
        logger=logger
    )

    # 加载评论数据（包含user_id, item_id, timestamp）
    reviews_df, user_mapping_original, item_mapping_original = processor.load_reviews()

    # ======================================
    # 新增打印1：原始数据（你自己模型使用的数据集）关键指标
    # ======================================
    logger.info("=" * 80)
    logger.info("【关键对比1：你自己模型使用的原始数据集】")
    logger.info(f"  原始交互总数: {len(reviews_df)}")
    logger.info(f"  原始用户总数: {len(user_mapping_original)}")
    logger.info(f"  原始物品总数: {len(item_mapping_original)}")
    # 计算原始序列长度统计
    user_seq_len_original = reviews_df.groupby('user_id').size()
    logger.info(f"  原始用户平均序列长度: {user_seq_len_original.mean():.2f}")
    logger.info(f"  原始用户序列长度中位数: {user_seq_len_original.median():.2f}")
    logger.info(f"  原始用户序列长度最大值: {user_seq_len_original.max()}")
    logger.info(f"  原始用户序列长度最小值: {user_seq_len_original.min()}")
    logger.info(
        f"  原始中交互数<5的用户占比: {len(user_seq_len_original[user_seq_len_original < 5]) / len(user_seq_len_original) * 100:.2f}%")
    logger.info("=" * 80)

    logger.info(f"Loaded {len(reviews_df)} interactions (before filtering)")
    logger.info(f"Users: {len(user_mapping_original)}, Items: {len(item_mapping_original)}")

    # 过滤冷启动用户和物品（迭代过滤直到稳定）
    logger.info("Filtering cold-start users and items...")
    prev_len = 0
    iteration = 0
    while len(reviews_df) != prev_len:
        prev_len = len(reviews_df)
        iteration += 1

        # 统计用户和物品交互数
        user_counts = reviews_df['user_id'].value_counts()
        item_counts = reviews_df['item_id'].value_counts()

        # 过滤
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index

        reviews_df = reviews_df[
            reviews_df['user_id'].isin(valid_users) &
            reviews_df['item_id'].isin(valid_items)
        ]

        logger.info(f"  Iteration {iteration}: {len(reviews_df)} interactions, "
                   f"{len(valid_users)} users, {len(valid_items)} items")

    # 重新创建映射（过滤后）
    unique_users = reviews_df['user_id'].unique()
    unique_items = reviews_df['item_id'].unique()
    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}

    # ======================================
    # 新增打印2：RecBole使用的过滤后数据集关键指标
    # ======================================
    logger.info("=" * 80)
    logger.info("【关键对比2：RecBole使用的过滤后数据集】")
    logger.info(f"  过滤后交互总数: {len(reviews_df)}")
    logger.info(f"  过滤后用户总数: {len(user_mapping)}")
    logger.info(f"  过滤后物品总数: {len(item_mapping)}")
    # 计算过滤后序列长度统计
    user_seq_len_filtered = reviews_df.groupby('user_id').size()
    logger.info(f"  过滤后用户平均序列长度: {user_seq_len_filtered.mean():.2f}")
    logger.info(f"  过滤后用户序列长度中位数: {user_seq_len_filtered.median():.2f}")
    logger.info(f"  过滤后用户序列长度最大值: {user_seq_len_filtered.max()}")
    logger.info(f"  过滤后用户序列长度最小值: {user_seq_len_filtered.min()}")
    # 计算数据缩减比例
    user_reduce_ratio = (len(user_mapping_original) - len(user_mapping)) / len(user_mapping_original) * 100
    item_reduce_ratio = (len(item_mapping_original) - len(item_mapping)) / len(item_mapping_original) * 100
    inter_reduce_ratio = (len(reviews_df) / len(processor.load_reviews()[0])) * 100
    logger.info(
        f"  用户数缩减比例: {user_reduce_ratio:.2f}% (原始{len(user_mapping_original)} → 过滤后{len(user_mapping)})")
    logger.info(
        f"  物品数缩减比例: {item_reduce_ratio:.2f}% (原始{len(item_mapping_original)} → 过滤后{len(item_mapping)})")
    logger.info(
        f"  交互数保留比例: {inter_reduce_ratio:.2f}% (原始{len(processor.load_reviews()[0])} → 过滤后{len(reviews_df)})")
    logger.info("=" * 80)

    # ======================================
    # 新增打印3：ID映射规则对比
    # ======================================
    logger.info("=" * 80)
    logger.info("【关键对比3：ID映射规则】")
    logger.info(f"  RecBole使用的ID映射: 从0开始编码（RecBole内部会自动+1，最终从1开始）")
    logger.info(f"  你自己模型的ID映射: 从0开始编码（无+1，padding用num_items）")
    logger.info(f"  RecBole padding ID: 0（真实ID从1开始）")
    logger.info(f"  你自己模型padding ID: {len(item_mapping_original)}（真实ID从0开始）")
    logger.info("=" * 80)

    logger.info(f"After filtering: {len(reviews_df)} interactions")
    logger.info(f"Users: {len(user_mapping)}, Items: {len(item_mapping)}")

    # 创建 .inter 文件（使用内部ID）
    inter_file = os.path.join(output_dir, f'{category}.inter')

    with open(inter_file, 'w', encoding='utf-8') as f:
        # 写入header（RecBole格式）
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')

        # 写入数据（使用内部ID）
        for _, row in reviews_df.iterrows():
            user_id = user_mapping[row['user_id']]
            item_id = item_mapping[row['item_id']]
            timestamp = row['timestamp']
            f.write(f'{user_id}\t{item_id}\t{timestamp}\n')

    logger.info(f"Created {inter_file}")

    # 创建反向映射
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}

    # 保存映射关系（用于后续结果对齐）
    mapping_file = os.path.join(output_dir, 'mappings.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump({
            'user_mapping': user_mapping,
            'item_mapping': item_mapping,
            'reverse_user_mapping': reverse_user_mapping,
            'reverse_item_mapping': reverse_item_mapping
        }, f)
    
    logger.info(f"Saved mappings to {mapping_file}")
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"  Total interactions: {len(reviews_df)}")
    logger.info(f"  Total users: {len(user_mapping)}")
    logger.info(f"  Total items: {len(item_mapping)}")
    logger.info(f"  Output files:")
    logger.info(f"    - {inter_file}")
    logger.info(f"    - {mapping_file}")
    logger.info("=" * 60)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to RecBole format')
    parser.add_argument('--category', type=str, default='Video_Games',
                        help='Dataset category (default: Video_Games)')
    parser.add_argument('--quick_mode', action='store_true',
                        help='Use quick mode (sample data)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--min_user_interactions', type=int, default=5,
                        help='Minimum interactions per user (default: 5)')
    parser.add_argument('--min_item_interactions', type=int, default=5,
                        help='Minimum interactions per item (default: 5)')

    args = parser.parse_args()

    # quick_mode下使用更宽松的过滤阈值
    min_user = args.min_user_interactions
    min_item = args.min_item_interactions
    if args.quick_mode and min_user == 5 and min_item == 5:
        # quick_mode默认使用更宽松的阈值
        min_user = 2
        min_item = 2

    convert_to_recbole_format(
        category=args.category,
        quick_mode=args.quick_mode,
        output_dir=args.output_dir,
        min_user_interactions=min_user,
        min_item_interactions=min_item
    )


if __name__ == '__main__':
    main()

