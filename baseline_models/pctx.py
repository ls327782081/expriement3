# Pctx模型 - 基于原始论文的两阶段训练流程
# 第一阶段：使用上游模型（如DuoRec）生成个性化语义ID
# 第二阶段：使用Pctx对生成的语义ID进行生成式推荐
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import AbstractTrainableModel  # 导入抽象基类
from config import config
from util import item_id_to_semantic_id, semantic_id_to_item_id
from transformers import T5Config, T5ForConditionalGeneration


class Pctx(AbstractTrainableModel):
    """
    Pctx: 个性化上下文感知推荐模型
    使用T5模型对预生成的语义ID进行生成式推荐
    """
    def __init__(self, use_pretrained_semantic_ids=True, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Pctx, self).__init__(device)  # 调用父类初始化
        self.use_pretrained_semantic_ids = use_pretrained_semantic_ids
        
        # 使用T5模型作为生成式推荐器
        t5_config = T5Config(
            num_layers=getattr(config, 'num_layers', 4), 
            num_decoder_layers=getattr(config, 'num_decoder_layers', 4),
            d_model=getattr(config, 'hidden_dim', 256),
            d_ff=getattr(config, 'mlp_dim', 1024),
            num_heads=getattr(config, 'attention_heads', 6),
            d_kv=getattr(config, 'd_kv', 64),
            dropout_rate=getattr(config, 'dropout', 0.1),
            activation_function=getattr(config, 'activation_function', 'relu'),
            vocab_size=config.codebook_size,  # 词汇表大小为码本大小
            pad_token_id=0,  # 填充token ID
            eos_token_id=1,  # 结束token ID
            decoder_start_token_id=0,
            feed_forward_proj=getattr(config, 'feed_forward_proj', 'relu'),
            is_decoder=False,
            tie_word_embeddings=False,
        )

        # T5条件生成模型，用于生成式推荐
        self.t5_model = T5ForConditionalGeneration(t5_config)

        # 用户和物品嵌入（用于构建上下文表示）
        self.user_emb = nn.Embedding(config.user_vocab_size, config.hidden_dim)
        self.item_emb = nn.Embedding(config.item_vocab_size, config.hidden_dim)
        
        # 多模态编码器（如果有的话）
        if hasattr(config, 'text_dim'):
            self.text_encoder = nn.Sequential(
                nn.Linear(config.text_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        
        if hasattr(config, 'visual_dim'):
            self.vision_encoder = nn.Sequential(
                nn.Linear(config.visual_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )

        # 优化器缓存
        self._optimizers = {}
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _get_optimizer_state_dict(self) -> dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._optimizers:
                self._optimizers[stage_id].load_state_dict(opt_state)

    def _train_one_batch(self, batch: any, stage_id: int, stage_kwargs: dict) -> tuple:
        """
        单batch训练逻辑
        Args:
            batch: 训练批次数据
            stage_id: 阶段ID
            stage_kwargs: 该阶段的自定义参数
        Returns:
            (batch_loss, batch_metrics)
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # 前向传播
        if "semantic_ids" in batch:
            # 使用预生成的语义ID进行训练
            outputs = self.forward(batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            # 如果没有语义ID，使用标准训练流程
            logits = self.forward(batch)  # 根据实际情况调整
            # 假设返回的是logits
            if isinstance(logits, dict) and 'logits' in logits:
                logits = logits['logits']
            elif hasattr(logits, 'logits'):
                logits = logits.logits
            elif torch.is_tensor(logits):
                # 如果直接返回logits张量
                pass
            else:
                # 默认处理：假设返回的是元组，第一个元素是logits
                logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            
            # 生成目标语义ID
            target = item_id_to_semantic_id(
                batch["item_id"], config.id_length, config.codebook_size
            ).to(self.device)
            
            # 计算损失
            criterion = torch.nn.CrossEntropyLoss()
            loss = 0
            for i in range(config.id_length):
                loss += criterion(logits[:, i, :], target[:, i])
            loss /= config.id_length

        # 计算指标
        if "semantic_ids" in batch:
            # 从outputs获取logits进行指标计算
            if hasattr(outputs, 'logits'):
                predictions = torch.argmax(outputs.logits, dim=-1)
                target = batch["labels"]
                correct = (predictions == target).float().sum()
                accuracy = correct / target.numel() if target.numel() > 0 else 0.0
            else:
                accuracy = 0.0  # 如果无法计算准确率，则返回0
        else:
            predictions = torch.argmax(logits, dim=-1)
            target = item_id_to_semantic_id(
                batch["item_id"], config.id_length, config.codebook_size
            ).to(self.device)
            correct = (predictions == target).float().sum()
            accuracy = correct / target.numel() if target.numel() > 0 else 0.0

        metrics = {'accuracy': accuracy}
        return loss, metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int, stage_kwargs: dict) -> dict:
        """单轮验证逻辑"""
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 移动数据到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # 前向传播
                if "semantic_ids" in batch:
                    outputs = self.forward(batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    if hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        target = batch["labels"]
                        correct = (predictions == target).float().sum()
                        total_correct += correct.item()
                        total_samples += target.numel() if hasattr(target, 'numel') else 0
                else:
                    logits = self.forward(batch)
                    if isinstance(logits, dict) and 'logits' in logits:
                        logits = logits['logits']
                    elif hasattr(logits, 'logits'):
                        logits = logits.logits
                    elif torch.is_tensor(logits):
                        pass
                    else:
                        logits = logits[0] if isinstance(logits, (tuple, list)) else logits
                        
                    # 生成目标语义ID
                    target = item_id_to_semantic_id(
                        batch["item_id"], config.id_length, config.codebook_size
                    ).to(self.device)
                    
                    # 计算损失
                    criterion = torch.nn.CrossEntropyLoss()
                    batch_loss = 0
                    for i in range(config.id_length):
                        batch_loss += criterion(logits[:, i, :], target[:, i])
                    batch_loss /= config.id_length
                    loss = batch_loss.item()

                    # 计算准确率
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == target).float().sum()
                    total_correct += correct.item()
                    total_samples += target.numel() if hasattr(target, 'numel') else 0

                total_loss += loss.item() if hasattr(loss, 'item') else loss

        avg_loss = total_loss / len(val_dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}

    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: 包含以下键的字典
                - user_id: 用户ID张量
                - item_id: 物品ID张量
                - input_ids: 上下文表示（如用户历史序列）
                - attention_mask: 注意力掩码
                - semantic_ids: 预生成的语义ID (batch_size, id_length) - 作为decoder输入
                - labels: 目标物品ID
                
        Returns:
            outputs: T5模型的输出，包含loss和logits
        """
        # 如果提供了预生成的语义ID，使用它们作为解码器输入
        if "semantic_ids" in batch:
            # 构建编码器输入（上下文表示）
            encoder_input_ids = batch["input_ids"]
            encoder_attention_mask = batch.get("attention_mask", torch.ones_like(encoder_input_ids))
            
            # 使用预生成的语义ID作为解码器输入
            decoder_input_ids = batch["semantic_ids"]
            
            # 准备标签（目标）
            labels = batch.get("labels", None)
            
            # T5模型前向传播
            outputs = self.t5_model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
            
            return outputs
        else:
            # 如果没有预生成的语义ID，需要先生成
            # 构建上下文表示
            user_emb = self.user_emb(batch["user_id"])  # (batch_size, hidden_dim)
            item_emb = self.item_emb(batch["item_id"])  # (batch_size, hidden_dim)
            
            # 如果有文本和视觉特征，也进行编码
            modality_features = [user_emb, item_emb]
            if "text_feat" in batch and hasattr(self, 'text_encoder'):
                text_emb = self.text_encoder(batch["text_feat"].float())
                modality_features.append(text_emb)
            if "vision_feat" in batch and hasattr(self, 'vision_encoder'):
                vision_emb = self.vision_encoder(batch["vision_feat"].float())
                modality_features.append(vision_emb)
            
            # 融合所有特征并重塑为适合T5输入的格式
            context_repr = torch.stack(modality_features, dim=1)  # (batch_size, num_features, hidden_dim)
            
            # 将context_repr转换为input_embeds用于T5
            # 需要将上下文信息编码为T5可接受的输入格式
            batch_size, num_features, hidden_dim = context_repr.shape
            input_embeds = context_repr.view(batch_size, num_features, hidden_dim)
            
            # 使用T5生成
            outputs = self.t5_model(
                inputs_embeds=input_embeds,
                return_dict=True
            )
            
            return outputs

    def train_step(self, batch, optimizer, criterion, device):
        """
        单步训练方法
        
        Args:
            batch: 训练批次数据
            optimizer: 优化器
            criterion: 损失函数
            device: 计算设备
            
        Returns:
            loss: 损失值
        """
        # 移动数据到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        if "semantic_ids" in batch:
            # 使用预生成的语义ID进行训练
            outputs = self.forward(batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            # 如果没有语义ID，使用标准训练流程
            logits = self.forward(batch)  # 根据实际情况调整
            # 假设返回的是logits
            if isinstance(logits, dict) and 'logits' in logits:
                logits = logits['logits']
            elif hasattr(logits, 'logits'):
                logits = logits.logits
            elif torch.is_tensor(logits):
                # 如果直接返回logits张量
                pass
            else:
                # 默认处理：假设返回的是元组，第一个元素是logits
                logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            
            # 生成目标语义ID
            target = item_id_to_semantic_id(
                batch["item_id"], config.id_length, config.codebook_size
            ).to(device)
            
            # 计算损失
            loss = 0
            for i in range(config.id_length):
                loss += criterion(logits[:, i, :], target[:, i])
            loss /= config.id_length
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def predict(self, batch, **kwargs):
        """
        预测方法
        
        Args:
            batch: 预测批次数据
            **kwargs: 其他参数
            
        Returns:
            predictions: 预测结果
        """
        # 使用beam search生成推荐
        top_k = kwargs.get('top_k', 10)
        num_beams = kwargs.get('num_beams', 10)
        
        # 使用T5的generate方法生成推荐
        with torch.no_grad():
            generated_sequences = self.t5_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                decoder_input_ids=batch.get("semantic_ids", None),  # 如果有预生成的语义ID
                max_length=top_k,
                num_beams=num_beams,
                num_return_sequences=min(top_k, num_beams),
                early_stopping=True,
                pad_token_id=0,
                eos_token_id=1
            )
        
        return generated_sequences

    def generate_recommendations(self, batch, top_k=10, num_beams=10):
        """
        生成推荐物品（使用beam search）
        
        Args:
            batch: 包含输入数据的字典
            top_k: 返回的推荐物品数量
            num_beams: beam search的beam数量
            
        Returns:
            generated_sequences: 生成的推荐序列
        """
        # 使用T5的generate方法生成推荐
        generated_sequences = self.t5_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            decoder_input_ids=batch.get("semantic_ids", None),  # 如果有预生成的语义ID
            max_length=top_k,
            num_beams=num_beams,
            num_return_sequences=min(top_k, num_beams),
            early_stopping=True,
            pad_token_id=0,
            eos_token_id=1
        )
        
        return generated_sequences
    
    def generate_semantic_ids_from_upstream(self, upstream_model_output):
        """
        从上游模型（如DuoRec）输出生成语义ID
        这是一个模拟函数，实际应用中会使用真正的上游模型输出
        
        Args:
            upstream_model_output: 上游模型的输出（如DuoRec的表示）
            
        Returns:
            semantic_ids: 生成的语义ID序列
        """
        # 这里假设上游模型输出是某种嵌入表示
        # 通过量化或其他方法转换为离散的语义ID
        batch_size = upstream_model_output.size(0)
        
        # 简单的线性投影后argmax得到语义ID
        projected = nn.Linear(upstream_model_output.size(-1), config.codebook_size * config.id_length)(
            upstream_model_output
        )
        projected = projected.view(batch_size, config.id_length, config.codebook_size)
        semantic_ids = torch.argmax(projected, dim=-1)  # (batch_size, id_length)
        
        return semantic_ids

    def train_with_duo_rec_integration(self, duo_rec_dataloader, pctx_dataloader, 
                                      optimizer, criterion, device, logger=None):
        """
        Pctx的完整两阶段训练流程：
        1. 使用DuoRec生成语义ID
        2. 使用生成的语义ID训练Pctx
        
        Args:
            duo_rec_dataloader: 用于训练DuoRec的数据加载器
            pctx_dataloader: 用于训练Pctx的数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            logger: 日志记录器
        """
        if logger:
            logger.info("开始Pctx的两阶段训练流程...")
        
        # 第一阶段：使用DuoRec生成语义ID（这里模拟）
        if logger:
            logger.info("第一阶段：准备语义ID（模拟DuoRec生成）")
        
        # 第二阶段：使用语义ID训练Pctx
        if logger:
            logger.info("第二阶段：使用语义ID训练Pctx模型")
        
        self.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(pctx_dataloader):
            # 确保数据在正确设备上
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # 确保批次中有语义ID
            if "semantic_ids" not in batch:
                # 如果没有语义ID，使用内部方法生成
                # 这里使用简化的逻辑
                with torch.no_grad():
                    # 生成语义ID（模拟从DuoRec获取）
                    user_emb = self.user_emb(batch["user_id"])
                    item_emb = self.item_emb(batch["item_id"])
                    interaction_repr = torch.cat([user_emb, item_emb], dim=-1)
                    
                    # 生成语义ID
                    semantic_logits = nn.Linear(interaction_repr.size(-1), config.codebook_size * config.id_length)(
                        interaction_repr
                    ).view(-1, config.id_length, config.codebook_size)
                    batch["semantic_ids"] = torch.argmax(semantic_logits, dim=-1)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.forward(batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if logger and batch_idx % 100 == 0:
                logger.info(f"Pctx训练 - Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if logger:
            logger.info(f"Pctx训练完成，平均损失: {avg_loss:.4f}")
        
        return avg_loss