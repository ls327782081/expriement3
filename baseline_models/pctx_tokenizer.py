"""
Pctx Tokenizer - 简化版本（渐进式对齐官方）

官方完整实现需要:
1. sentence-transformers 编码文本
2. PCA 降维
3. Faiss RQ 量化
4. interactionKey 生成 (userID-itemID-preorderedSeq)
5. merge_conflict 和 merge_low_frequency 策略

当前简化版本:
- 使用基于 item_id 的哈希映射（临时方案）
- 支持用户历史序列（对齐官方数据格式）
- 后续可以逐步替换为完整的语义量化
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from config import config


class PctxTokenizerSimplified:
    """
    Pctx Tokenizer 简化版
    
    TODO: 完整实现需要集成:
    - sentence-transformers
    - sklearn.decomposition.PCA
    - faiss (Residual Quantizer)
    """
    
    def __init__(self, codebook_size=256, n_codebooks=3, id_length=4, 
                 max_seq_len=20, n_user_tokens=1):
        """
        Args:
            codebook_size: 每个codebook的大小（官方默认256）
            n_codebooks: codebook数量（官方默认3）
            id_length: semantic ID长度（官方默认4）
            max_seq_len: 最大序列长度
            n_user_tokens: 用户token数量
        """
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.id_length = id_length
        self.max_seq_len = max_seq_len
        self.n_user_tokens = n_user_tokens
        
        # Token IDs
        self.pad_token_id = 0
        self.base_user_token = codebook_size * (n_codebooks + 1) + 1
        self.eos_token_id = self.base_user_token + n_user_tokens
        self.vocab_size = self.eos_token_id + 1
        
        # 临时映射: item_id -> semantic_ids
        # TODO: 替换为基于 sentence-transformers + Faiss RQ 的映射
        self.item_to_semantic = {}
        
        print(f"[PctxTokenizer] Initialized:")
        print(f"  - codebook_size: {codebook_size}")
        print(f"  - n_codebooks: {n_codebooks}")
        print(f"  - id_length: {id_length}")
        print(f"  - vocab_size: {self.vocab_size}")
        print(f"  - pad_token_id: {self.pad_token_id}")
        print(f"  - eos_token_id: {self.eos_token_id}")
    
    def _item_id_to_semantic_id(self, item_id: int, user_history: List[int] = None) -> List[int]:
        """
        将 item_id 转换为 semantic ID
        
        当前实现: 简单哈希（临时方案）
        TODO: 实现官方的 interactionKey -> semantic_ids 映射
        
        官方实现:
            interactionKey = f"{userID}-{itemID}-{preorderedSeq}"
            semantic_ids = self.interactionkey2sidTokens[interactionKey]
        
        Args:
            item_id: 物品ID
            user_history: 用户历史序列（用于个性化，当前未使用）
        
        Returns:
            semantic_ids: 长度为 id_length 的列表
        """
        # 临时方案: 使用确定性哈希
        # TODO: 替换为基于用户历史的个性化映射
        if item_id not in self.item_to_semantic:
            np.random.seed(item_id)
            semantic_ids = [
                np.random.randint(0, self.codebook_size) 
                for _ in range(self.id_length)
            ]
            self.item_to_semantic[item_id] = semantic_ids
        
        return self.item_to_semantic[item_id]
    
    def tokenize(self, user_id: int, item_sequence: List[int], 
                 target_item: int = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize 用户序列
        
        对齐官方格式:
            input_ids: 用户历史的 semantic IDs + user_token
            labels: 目标物品的 semantic IDs + eos_token
        
        Args:
            user_id: 用户ID
            item_sequence: 用户历史物品序列
            target_item: 目标物品（训练时需要）
        
        Returns:
            {
                'input_ids': tensor,
                'attention_mask': tensor,
                'labels': tensor (if target_item is not None)
            }
        """
        input_ids = []
        
        # 1. 编码用户历史序列
        for item_id in item_sequence[-self.max_seq_len:]:
            semantic_ids = self._item_id_to_semantic_id(item_id)
            input_ids.extend(semantic_ids)
        
        # 2. 添加 user token
        # 官方: user_token = base_user_token + (user_id % n_user_tokens)
        user_token = self.base_user_token + (user_id % self.n_user_tokens)
        input_ids.append(user_token)
        
        # 3. 添加 eos token
        input_ids.append(self.eos_token_id)
        
        # 4. 创建 attention_mask
        attention_mask = [1] * len(input_ids)
        
        # 5. 编码 labels（如果有目标物品）
        labels = None
        if target_item is not None:
            target_semantic_ids = self._item_id_to_semantic_id(target_item)
            labels = target_semantic_ids + [self.eos_token_id]
        
        # 转换为 tensor
        result = {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }
        
        if labels is not None:
            result['labels'] = torch.tensor([labels], dtype=torch.long)
        
        return result

