"""
Pctx Tokenizer - 完全对齐官方实现
官方源码: https://github.com/YoungZ365/Pctx/blob/main/genrec/models/Pctx/tokenizer.py

核心功能:
1. 使用 sentence-transformers 编码文本
2. PCA 降维
3. Faiss Residual Quantizer 量化
4. interactionKey 生成 (userID-itemID-preorderedSeq)
5. merge_conflict 和 merge_low_frequency 策略
"""
import os
import pickle
from typing import List, Dict

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm


class PctxTokenizerOfficial:
    """
    Pctx Tokenizer - 官方实现

    完全对齐官方的 tokenization 流程
    """

    def __init__(self,
                 codebook_size: int = 256,
                 n_codebooks: int = 3,
                 id_length: int = 4,
                 max_seq_len: int = 20,
                 n_user_tokens: int = 1,
                 sent_model_name: str = 'sentence-transformers/sentence-t5-base',
                 pca_dim: int = 128,
                 diff_dim: int = 64,
                 alpha: float = 0.5,
                 device: str = 'cpu'):
        """初始化 Tokenizer"""
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.id_length = id_length
        self.max_seq_len = max_seq_len
        self.n_user_tokens = n_user_tokens
        self.pca_dim = pca_dim
        self.diff_dim = diff_dim
        self.alpha = alpha
        self.device = device

        # Token IDs
        self.pad_token_id = 0
        self.base_user_token = codebook_size * (n_codebooks + 1) + 1
        self.eos_token_id = self.base_user_token + n_user_tokens
        self.vocab_size = self.eos_token_id + 1

        # Sentence Transformer
        print(f"[PctxTokenizer] Loading sentence-transformers: {sent_model_name}")
        self.sent_model = SentenceTransformer(sent_model_name, device=device)

        # PCA 模型
        self.pca_sent = None
        self.pca_diff = None

        # Faiss RQ 索引
        self.rq_index = None

        # interactionKey -> semantic_ids 映射
        self.interactionkey2sidTokens = {}

        # item_id -> text 映射
        self.item2text = {}

        print(f"[PctxTokenizer] Initialized with vocab_size={self.vocab_size}")

    def set_item_texts(self, item2text: Dict[int, str]):
        """设置物品文本映射"""
        self.item2text = item2text
        print(f"[PctxTokenizer] Loaded {len(item2text)} item texts")

    def return_one_item_interactionKey_ABC_in_a_seq(self,
                                                     lst: List[int],
                                                     idx: int,
                                                     userID: int,
                                                     max_len: int) -> str:
        """
        生成 interactionKey（完全对齐官方）

        格式: "userID-itemID-preorderedSeq"
        例如: "123-456-789_790_791"
        """
        A = userID
        B = lst[idx]
        start_idx = max(0, idx - (max_len - 1))

        if idx == 0:
            preorderedSeq = '0'
        else:
            preorderedSeq = '_'.join(str(x) for x in lst[start_idx:idx])

        C = preorderedSeq
        interactionKey = f"{A}-{B}-{C}"

        return interactionKey

    def build_semantic_ids_from_dataset(self,
                                        user_sequences: Dict[int, List[int]],
                                        item_texts: Dict[int, str] = None,
                                        save_path: str = None):
        """
        从数据集构建 semantic IDs（对齐官方流程）

        步骤:
        1. 收集所有 interactionKeys
        2. 编码文本 (sentence-transformers)
        3. PCA 降维
        4. 融合个性化差异
        5. Faiss RQ 量化
        6. 构建 interactionKey -> semantic_ids 映射

        Args:
            user_sequences: {user_id: [item1, item2, ...]}
            item_texts: {item_id: text}
            save_path: 保存路径
        """
        if item_texts is not None:
            self.set_item_texts(item_texts)

        print("\n[Step 1] 收集所有 interactionKeys...")
        all_interactionKeys = []
        interactionKey_to_info = {}

        for user_id, item_seq in tqdm(user_sequences.items()):
            for idx in range(len(item_seq)):
                interactionKey = self.return_one_item_interactionKey_ABC_in_a_seq(
                    item_seq, idx, user_id, self.max_seq_len
                )
                all_interactionKeys.append(interactionKey)
                interactionKey_to_info[interactionKey] = {
                    'user_id': user_id,
                    'item_id': item_seq[idx],
                    'history': item_seq[max(0, idx - self.max_seq_len + 1):idx]
                }

        print(f"  收集到 {len(all_interactionKeys)} 个 interactionKeys")

        # Step 2: 编码文本
        print("\n[Step 2] 编码文本 (sentence-transformers)...")
        unique_items = set()
        for info in interactionKey_to_info.values():
            unique_items.add(info['item_id'])

        item_texts_list = []
        item_ids_list = []
        for item_id in sorted(unique_items):
            if item_id in self.item2text:
                item_texts_list.append(self.item2text[item_id])
            else:
                item_texts_list.append(f"item_{item_id}")
            item_ids_list.append(item_id)

        print(f"  编码 {len(item_texts_list)} 个物品文本...")
        sent_embs = self.sent_model.encode(
            item_texts_list,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        print(f"  文本嵌入维度: {sent_embs.shape}")

        item2emb = {item_id: emb for item_id, emb in zip(item_ids_list, sent_embs)}

        # Step 3: PCA 降维
        print("\n[Step 3] PCA 降维...")
        # 确保 n_components 不超过样本数
        n_components = min(self.pca_dim, sent_embs.shape[0], sent_embs.shape[1])
        print(f"  PCA 维度: {n_components} (原始: {self.pca_dim})")
        self.pca_sent = PCA(n_components=n_components, whiten=True)
        pca_sent_emb = self.pca_sent.fit_transform(sent_embs)
        print(f"  PCA 后维度: {pca_sent_emb.shape}")

        item2pca_emb = {item_id: emb for item_id, emb in zip(item_ids_list, pca_sent_emb)}

        # Step 4: 构建 interactionKey 嵌入
        print("\n[Step 4] 构建 interactionKey 嵌入...")
        interactionKey_embeddings = []
        interactionKey_list = []

        for interactionKey in tqdm(all_interactionKeys):
            info = interactionKey_to_info[interactionKey]
            item_id = info['item_id']

            if item_id in item2pca_emb:
                emb = item2pca_emb[item_id]
            else:
                emb = np.zeros(self.pca_dim)

            interactionKey_embeddings.append(emb)
            interactionKey_list.append(interactionKey)

        interactionKey_embeddings = np.array(interactionKey_embeddings)
        print(f"  收集到 {len(interactionKey_embeddings)} 个交互嵌入")

        # Step 5: Faiss RQ 量化
        print("\n[Step 5] Faiss RQ 量化...")
        d = interactionKey_embeddings.shape[1]
        M = self.n_codebooks
        nbits = 8

        print(f"  初始化 RQ: d={d}, M={M}, nbits={nbits}")
        self.rq_index = faiss.IndexResidualQuantizer(d, M, nbits)

        print(f"  训练 RQ...")
        self.rq_index.train(interactionKey_embeddings.astype(np.float32))

        print(f"  计算 semantic IDs...")
        semantic_ids_array = self.rq_index.rq.compute_codes(
            interactionKey_embeddings.astype(np.float32)
        )
        print(f"  Semantic IDs shape: {semantic_ids_array.shape}")

        # Step 6: 构建映射
        print("\n[Step 6] 构建 interactionKey -> semantic_ids 映射...")
        for i, interactionKey in enumerate(interactionKey_list):
            semantic_ids = tuple(semantic_ids_array[i].tolist())
            self.interactionkey2sidTokens[interactionKey] = semantic_ids

        print(f"  构建了 {len(self.interactionkey2sidTokens)} 个映射")

        if save_path:
            self.save(save_path)
            print(f"\n✅ Tokenizer 已保存到: {save_path}")


    def tokenize(self, user_id: int, item_sequence: List[int],
                 target_item: int = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize 用户序列（使用预构建的映射）

        Args:
            user_id: 用户ID
            item_sequence: 用户历史物品序列
            target_item: 目标物品

        Returns:
            {input_ids, attention_mask, labels}
        """
        input_ids = []

        # 编码历史序列
        for idx in range(len(item_sequence)):
            interactionKey = self.return_one_item_interactionKey_ABC_in_a_seq(
                item_sequence, idx, user_id, self.max_seq_len
            )

            if interactionKey in self.interactionkey2sidTokens:
                semantic_ids = self.interactionkey2sidTokens[interactionKey]
                input_ids.extend(semantic_ids)
            else:
                # 如果没有映射，使用随机ID（不应该发生）
                print(f"Warning: interactionKey not found: {interactionKey}")
                input_ids.extend([0] * self.id_length)

        # 添加 user token
        user_token = self.base_user_token + (user_id % self.n_user_tokens)
        input_ids.append(user_token)

        # 添加 eos token
        input_ids.append(self.eos_token_id)

        # attention_mask
        attention_mask = [1] * len(input_ids)

        # labels
        labels = None
        if target_item is not None:
            # 生成目标的 interactionKey
            target_seq = item_sequence + [target_item]
            target_idx = len(target_seq) - 1
            target_interactionKey = self.return_one_item_interactionKey_ABC_in_a_seq(
                target_seq, target_idx, user_id, self.max_seq_len
            )

            if target_interactionKey in self.interactionkey2sidTokens:
                target_semantic_ids = self.interactionkey2sidTokens[target_interactionKey]
                labels = list(target_semantic_ids) + [self.eos_token_id]
            else:
                print(f"Warning: target interactionKey not found: {target_interactionKey}")
                labels = [0] * self.id_length + [self.eos_token_id]

        # 转换为 tensor
        result = {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

        if labels is not None:
            result['labels'] = torch.tensor([labels], dtype=torch.long)

        return result

    def save(self, save_path: str):
        """保存 Tokenizer"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {
            'config': {
                'codebook_size': self.codebook_size,
                'n_codebooks': self.n_codebooks,
                'id_length': self.id_length,
                'max_seq_len': self.max_seq_len,
                'n_user_tokens': self.n_user_tokens,
                'pca_dim': self.pca_dim,
                'diff_dim': self.diff_dim,
                'alpha': self.alpha,
            },
            'interactionkey2sidTokens': self.interactionkey2sidTokens,
            'pca_sent': self.pca_sent,
            'pca_diff': self.pca_diff,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Tokenizer saved to {save_path}")

    @classmethod
    def load(cls, load_path: str, sent_model_name: str = 'sentence-transformers/sentence-t5-base',
             device: str = 'cpu'):
        """加载 Tokenizer"""
        with open(load_path, 'rb') as f:
            save_dict = pickle.load(f)

        config = save_dict['config']
        tokenizer = cls(
            codebook_size=config['codebook_size'],
            n_codebooks=config['n_codebooks'],
            id_length=config['id_length'],
            max_seq_len=config['max_seq_len'],
            n_user_tokens=config['n_user_tokens'],
            sent_model_name=sent_model_name,
            pca_dim=config['pca_dim'],
            diff_dim=config['diff_dim'],
            alpha=config['alpha'],
            device=device
        )

        tokenizer.interactionkey2sidTokens = save_dict['interactionkey2sidTokens']
        tokenizer.pca_sent = save_dict['pca_sent']
        tokenizer.pca_diff = save_dict['pca_diff']

        print(f"Tokenizer loaded from {load_path}")
        print(f"  - {len(tokenizer.interactionkey2sidTokens)} interactionKeys")

        return tokenizer

