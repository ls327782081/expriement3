"""
DGMRec: Disentangling and Generating Modalities for Recommendation
基于GitHub官方实现的适配版本
来源: https://github.com/ptkjw1997/DGMRec

核心创新:
1. 模态解耦 (Modality Disentanglement) - 分离通用特征和特定特征
2. 模态生成 (Modality Generation) - 为缺失模态生成特征
3. 互信息最小化 (MI Minimization) - 确保特征独立性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DGMRec(nn.Module):
    """
    DGMRec模型 - 简化适配版本
    来源: https://github.com/ptkjw1997/DGMRec
    """
    def __init__(self, config):
        super(DGMRec, self).__init__()
        self.embedding_dim = config.hidden_dim
        self.n_mm_layers = 2  # 模态图卷积层数
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(config.user_vocab_size, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(config.item_vocab_size, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # 模态编码器 - 通用特征 (General)
        self.image_encoder = nn.Linear(config.visual_dim, self.embedding_dim)
        self.text_encoder = nn.Linear(config.text_dim, self.embedding_dim)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # 模态编码器 - 特定特征 (Specific)
        self.image_encoder_s = nn.Linear(config.visual_dim, self.embedding_dim)
        self.text_encoder_s = nn.Linear(config.text_dim, self.embedding_dim)
        
        # 偏好过滤器
        self.image_preference = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_preference = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        # 解码器
        self.image_decoder = nn.Linear(self.embedding_dim * 2, config.visual_dim)
        self.text_decoder = nn.Linear(self.embedding_dim * 2, config.text_dim)
        
        # 生成器 - 特定特征
        self.image_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.text_gen = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # 生成器 - 跨模态转换
        self.image2text = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.text2image = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # 超参数
        self.lambda_1 = 0.1  # 解耦损失权重
        self.lambda_2 = 0.1  # 对齐损失权重
        self.infoNCETemp = 0.4
        
        self.act_g = nn.Tanh()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in [self.image_encoder, self.text_encoder, self.shared_encoder,
                      self.image_encoder_s, self.text_encoder_s,
                      self.image_preference, self.text_preference,
                      self.image_decoder, self.text_decoder]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                
        for module in [self.image_gen, self.text_gen, self.image2text, self.text2image]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
    
    def mge(self, image_feat, text_feat):
        """
        Modality Graph Encoder - 模态图编码器
        Args:
            image_feat: (batch, visual_dim)
            text_feat: (batch, text_dim)
        Returns:
            通用特征和特定特征
        """
        # 通用特征 (General)
        item_image_g = torch.sigmoid(
            self.shared_encoder(self.act_g(self.image_encoder(image_feat)))
        )
        item_text_g = torch.sigmoid(
            self.shared_encoder(self.act_g(self.text_encoder(text_feat)))
        )
        
        # 特定特征 (Specific)
        item_image_s = torch.sigmoid(self.image_encoder_s(image_feat))
        item_text_s = torch.sigmoid(self.text_encoder_s(text_feat))
        
        return item_image_g, item_text_g, item_image_s, item_text_s
    
    def InfoNCE(self, view1, view2, temperature=0.4):
        """InfoNCE对比损失"""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-8))
        return torch.mean(cl_loss)
    
    def perturb(self, x):
        """添加扰动"""
        noise = torch.rand_like(x)
        x = x + torch.sign(x) * F.normalize(noise, dim=-1) * 0.1
        return x
    
    def forward(self, user_ids, item_ids, visual_feat, text_feat):
        """
        Args:
            user_ids: (batch,)
            item_ids: (batch,)
            visual_feat: (batch, visual_dim)
            text_feat: (batch, text_dim)
        Returns:
            dict: 包含嵌入和损失
        """
        batch_size = user_ids.size(0)
        
        # 获取用户和物品嵌入
        user_emb = self.user_embedding(user_ids)
        item_id_emb = self.item_id_embedding(item_ids)
        
        # 模态图编码
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge(visual_feat, text_feat)
        
        # 解耦损失 - InfoNCE (通用特征应该相似)
        loss_InfoNCE = self.InfoNCE(item_image_g, item_text_g, temperature=self.infoNCETemp)
        
        # 用户偏好过滤
        image_filter = torch.tanh(self.image_preference(user_emb))
        text_filter = torch.tanh(self.text_preference(user_emb))
        
        # 应用过滤器
        item_image_g_filtered = item_image_g * image_filter
        item_text_g_filtered = item_text_g * text_filter
        item_image_s_filtered = item_image_s * image_filter
        item_text_s_filtered = item_text_s * text_filter
        
        # 生成损失
        item_image_g_gen = self.text2image(self.perturb(item_text_g_filtered))
        item_text_g_gen = self.image2text(self.perturb(item_image_g_filtered))
        item_image_s_gen = self.image_gen(self.perturb(image_filter))
        item_text_s_gen = self.text_gen(self.perturb(text_filter))
        
        loss_gen = F.mse_loss(item_image_s, item_image_s_gen)
        loss_gen += F.mse_loss(item_text_s, item_text_s_gen)
        loss_gen += F.mse_loss(item_text_g, item_text_g_gen)
        loss_gen += F.mse_loss(item_image_g, item_image_g_gen)
        
        # 重建损失
        image_concat = torch.cat([item_image_g_filtered, item_image_s_filtered], dim=1)
        text_concat = torch.cat([item_text_g_filtered, item_text_s_filtered], dim=1)
        
        image_recon = self.image_decoder(self.perturb(image_concat.detach()))
        text_recon = self.text_decoder(self.perturb(text_concat.detach()))
        
        loss_recon = F.mse_loss(image_recon, visual_feat) * 0.1
        loss_recon += F.mse_loss(text_recon, text_feat) * 0.1
        
        # 对齐损失
        loss_alignUI = self.InfoNCE(user_emb, item_id_emb, temperature=0.2)
        loss_alignUI += self.InfoNCE(
            item_image_g_filtered + item_text_g_filtered,
            item_id_emb,
            temperature=self.infoNCETemp
        )
        
        # 融合特征
        user_final = user_emb
        item_final = item_id_emb + (
            (item_image_g_filtered + item_text_g_filtered) / 2 +
            item_image_s_filtered + item_text_s_filtered
        ) / 3
        
        # 总损失
        loss_disentangle = self.lambda_1 * loss_InfoNCE
        loss_generation = loss_gen + loss_recon
        loss_align = self.lambda_2 * loss_alignUI
        
        total_loss = loss_disentangle + loss_generation + loss_align
        
        return {
            'user_embeddings': user_final,
            'item_embeddings': item_final,
            'total_loss': total_loss,
            'loss_disentangle': loss_disentangle,
            'loss_generation': loss_generation,
            'loss_align': loss_align,
        }
    
    def get_user_embedding(self, user_ids):
        """获取用户嵌入"""
        return self.user_embedding(user_ids)
    
    def get_item_embedding(self, item_ids, visual_feat, text_feat):
        """获取物品嵌入"""
        item_id_emb = self.item_id_embedding(item_ids)
        
        # 模态图编码
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge(visual_feat, text_feat)
        
        # 简化版本：直接融合
        item_final = item_id_emb + (
            (item_image_g + item_text_g) / 2 + item_image_s + item_text_s
        ) / 3
        
        return item_final
    
    def predict(self, user_ids, item_ids, visual_feat, text_feat):
        """预测评分"""
        user_emb = self.get_user_embedding(user_ids)
        item_emb = self.get_item_embedding(item_ids, visual_feat, text_feat)
        
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

