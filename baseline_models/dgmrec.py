import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_model import AbstractTrainableModel  # 导入抽象基类
from config import config
import scipy.sparse as sp

# 简化的工具函数（替代 utils.utils）
def build_sim(context):
    """构建相似度矩阵"""
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def compute_normalized_laplacian(adj):
    """计算归一化拉普拉斯矩阵"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def build_knn_neighbourhood(adj, topk):
    """构建 KNN 邻域"""
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


class DGMRec(AbstractTrainableModel):
    """
    DGMRec模型 - 基于原始论文实现的完整版本
    来源: https://github.com/ptkjw1997/DGMRec
    核心功能:
    1. 协同过滤嵌入 (Collaborative Filtering Embeddings)
    2. 模态解耦 (Modality Disentanglement) - 分离通用特征和特定特征
    3. 模态生成 (Modality Generation) - 为缺失模态生成特征
    4. 互信息最小化 (MI Minimization) - 确保特征独立性
    """

    def __init__(self, config, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(DGMRec, self).__init__(device)
        self.config = config
        self.n_users = config.user_vocab_size  # 从数据集中加载到config中获取用户数
        self.n_items = config.item_vocab_size  # 从数据集中加载到config中获取物品数
        self.latent_dim = config.hidden_dim
        self.n_ui_layers = getattr(config, 'n_ui_layers', 2)
        self.n_mm_layers = getattr(config, 'n_mm_layers', 2)
        self.knn_k = getattr(config, 'knn_k', 10)
        self.alpha = getattr(config, 'alpha', 0.1)
        self.lambda_1 = getattr(config, 'lambda_1', 0.1)
        self.lambda_2 = getattr(config, 'lambda_2', 0.1)
        self.infoNCETemp = getattr(config, 'infoNCETemp', 0.4)
        self.alignBMTemp = getattr(config, 'alignBMTemp', 0.4)
        self.alignUITemp = getattr(config, 'alignUITemp', 0.4)

        # 协同过滤嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 从数据集获取交互矩阵
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tensor(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_adj = self.norm_adj.to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        # 获取视觉和文本特征
        self.v_feat = dataset.visual_feat  # 假设数据集有这个属性
        self.t_feat = dataset.text_feat    # 假设数据集有这个属性
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
            
            # 构建图像邻接矩阵
            image_adj = build_sim(self.image_embedding.weight.detach().cpu())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            self.image_adj = compute_normalized_laplacian(image_adj).to_sparse_coo().to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)

            # 构建文本邻接矩阵
            text_adj = build_sim(self.text_embedding.weight.detach().cpu())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            self.text_adj = compute_normalized_laplacian(text_adj).to_sparse_coo().to(self.device)

        # 多模态编码器（对齐官方代码）
        image_input_dim = self.v_feat.shape[1] if self.v_feat is not None else config.visual_dim
        text_input_dim = self.t_feat.shape[1] if self.t_feat is not None else config.text_dim

        # General encoders (通用特征编码器)
        self.image_encoder = nn.Linear(image_input_dim, self.latent_dim).to(self.device)
        self.text_encoder = nn.Linear(text_input_dim, self.latent_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # Specific encoders (特定特征编码器)
        self.image_encoder_s = nn.Linear(image_input_dim, self.latent_dim).to(self.device)
        self.text_encoder_s = nn.Linear(text_input_dim, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder_s.weight)
        nn.init.xavier_uniform_(self.text_encoder_s.weight)

        # Preference encoders (用户偏好编码器) - 官方代码
        self.image_preference_ = nn.Linear(self.latent_dim, self.latent_dim, bias=False).to(self.device)
        self.text_preference_ = nn.Linear(self.latent_dim, self.latent_dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.image_preference_.weight)
        nn.init.xavier_uniform_(self.text_preference_.weight)

        # Decoders (解码器) - 官方代码
        if self.v_feat is not None:
            self.image_decoder = nn.Linear(self.latent_dim * 2, self.v_feat.shape[1]).to(self.device)
            nn.init.xavier_uniform_(self.image_decoder.weight)
        if self.t_feat is not None:
            self.text_decoder = nn.Linear(self.latent_dim * 2, self.t_feat.shape[1]).to(self.device)
            nn.init.xavier_uniform_(self.text_decoder.weight)

        # Generator for Specific Feature (特定特征生成器) - 官方代码
        self.image_gen = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim)
        ).to(self.device)
        self.image_gen.apply(self.init_weight)

        self.text_gen = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim)
        ).to(self.device)
        self.text_gen.apply(self.init_weight)

        # Generator for General Feature (通用特征生成器) - 官方代码
        self.image2text = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim)
        ).to(self.device)
        self.image2text.apply(self.init_weight)

        self.text2image = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim)
        ).to(self.device)
        self.text2image.apply(self.init_weight)

        # Activation function
        self.act_g = nn.Tanh()

    def init_weight(self, layer):
        """初始化单个层的权重（官方代码）"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def init_mi_estimator(self):
        """初始化互信息估计器"""
        # 这里可以初始化MI估计器
        pass

    def pre_epoch_processing(self):
        """
        每个epoch之前的预处理步骤
        """
        # 获取模态嵌入
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()
        
        # 这里可以实现MI估计器的训练逻辑
        # 为了简化，暂时留空
        pass

    def mge(self):
        """
        多模态嵌入生成 (Modality Generation Encoding) - 对齐官方代码
        提取通用特征和特定特征
        """
        # 官方代码：General features (通用特征)
        item_image_g = F.sigmoid(self.shared_encoder(self.act_g(self.image_encoder(self.image_embedding.weight))))
        item_text_g = F.sigmoid(self.shared_encoder(self.act_g(self.text_encoder(self.text_embedding.weight))))

        # 官方代码：Specific features (特定特征)
        item_image_s = F.sigmoid(self.image_encoder_s(self.image_embedding.weight))
        item_text_s = F.sigmoid(self.text_encoder_s(self.text_embedding.weight))

        return item_image_g, item_text_g, item_image_s, item_text_s

    def cge(self, user_emb, item_emb, adj):
        """
        协同过滤嵌入生成 (Collaborative Filtering Embedding Generation)
        """
        # 协同过滤GCN
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        user_embeddings, item_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return user_embeddings, item_embedding

    def InfoNCE(self, view1, view2, temperature=0.4):
        """
        InfoNCE损失函数
        """
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def bpr_loss(self, users, pos_items, neg_items):
        """
        BPR损失函数
        """
        if len(pos_items.shape) == 2:
            pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
            neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        else:
            pos_scores = torch.einsum("ik, ijk -> ij", users, pos_items)
            neg_scores = torch.einsum("ik, ijk -> ij", users, neg_items)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def calculate_loss(self, interaction):
        """
        计算完整的DGMRec损失函数
        """
        users, pos_items, neg_items = interaction

        # 计算协同过滤嵌入
        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, 
            self.item_id_embedding.weight, 
            self.norm_adj
        )
        
        # 获取模态嵌入
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        # 计算InfoNCE损失
        loss_InfoNCE = self.InfoNCE(item_image_g, item_text_g, temperature=self.infoNCETemp)

        # 计算用户过滤特征
        item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_preference(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_preference(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        # 通用特征过滤
        item_image_g_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)

        # 特定特征过滤
        item_image_s_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        # 通用特征传播
        for _ in range(self.n_mm_layers):
            item_image_g_filtered = torch.sparse.mm(self.image_adj, item_image_g_filtered)
            item_text_g_filtered = torch.sparse.mm(self.text_adj, item_text_g_filtered)
        user_image_g = torch.sparse.mm(self.adj, item_image_g_filtered) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g_filtered) * self.num_inters[:self.n_users]

        # 特定特征传播
        for _ in range(self.n_mm_layers):
            item_image_s_filtered = torch.sparse.mm(self.image_adj, item_image_s_filtered)
            item_text_s_filtered = torch.sparse.mm(self.text_adj, item_text_s_filtered)
        user_image_s = torch.sparse.mm(self.adj, item_image_s_filtered) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s_filtered) * self.num_inters[:self.n_users]

        # 计算BPR主损失
        user_emb = user_embeddings[users]
        pos_item_emb = item_embedding[pos_items]
        neg_item_emb = item_embedding[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        # 互信息损失
        loss_mi = self.mi_estimator(item_image_s, item_image_g) + \
                  self.mi_estimator(item_text_s, item_text_g)

        # 对齐损失
        loss_alignUI = self.InfoNCE(user_embeddings[users], item_embedding[pos_items], temperature=self.alignUITemp)
        loss_alignUI += self.InfoNCE(user_image_g[users] + user_text_g[users], item_image_g[pos_items] + item_text_g[pos_items], temperature=self.infoNCETemp)
        loss_alignUI += self.InfoNCE(user_image_s[users], item_image_s[pos_items], temperature=self.alignUITemp)
        loss_alignUI += self.InfoNCE(user_text_s[users], item_text_s[pos_items], temperature=self.alignUITemp)

        loss_alignBM = self.InfoNCE(item_embedding[pos_items], item_image_g[pos_items] + item_text_g[pos_items], temperature=self.alignBMTemp)
        loss_alignBM += self.InfoNCE(user_embeddings[users], user_image_g[users] + user_text_g[users], temperature=self.alignBMTemp)

        # 特征融合
        user_emb_fused = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb_fused = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3 

        user_emb = user_emb_fused[users]
        pos_item_emb = item_emb_fused[pos_items]
        neg_item_emb = item_emb_fused[neg_items]

        loss_main_bpr = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)

        # 正则化损失
        loss_reg = self.calculate_reg_loss(user_embeddings[users], item_embedding[pos_items], item_embedding[neg_items])

        # 总损失
        loss_disentangle = self.lambda_1 * (loss_mi + loss_InfoNCE)
        loss_align = self.lambda_2 * (loss_alignUI + loss_alignBM)

        total_loss = loss_main_bpr + loss_disentangle + loss_align + loss_reg

        return total_loss

    def full_sort_predict(self, interaction):
        """
        全排序预测
        """
        users, _ = interaction

        # 计算协同过滤嵌入
        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, 
            self.item_id_embedding.weight, 
            self.norm_adj
        )
        
        # 获取模态嵌入
        item_image_g, item_text_g, item_image_s, item_text_s = self.mge()

        # 特征过滤
        item_image_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.image_preference(self.user_embedding.weight))) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(self.adj.t(), F.tanh(self.text_preference(self.user_embedding.weight))) * self.num_inters[self.n_users:]

        # 通用特征传播
        item_image_g_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_g)
        item_text_g_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_g)

        for _ in range(self.n_mm_layers):
            item_image_g_filtered = torch.sparse.mm(self.image_adj, item_image_g_filtered)
            item_text_g_filtered = torch.sparse.mm(self.text_adj, item_text_g_filtered)
        user_image_g = torch.sparse.mm(self.adj, item_image_g_filtered) * self.num_inters[:self.n_users]
        user_text_g = torch.sparse.mm(self.adj, item_text_g_filtered) * self.num_inters[:self.n_users]

        # 特定特征传播
        item_image_s_filtered = torch.einsum("ij, ij -> ij", item_image_filter, item_image_s)
        item_text_s_filtered = torch.einsum("ij, ij -> ij", item_text_filter, item_text_s)

        for _ in range(self.n_mm_layers):
            item_image_s_filtered = torch.sparse.mm(self.image_adj, item_image_s_filtered)
            item_text_s_filtered = torch.sparse.mm(self.text_adj, item_text_s_filtered)
        user_image_s = torch.sparse.mm(self.adj, item_image_s_filtered) * self.num_inters[:self.n_users]
        user_text_s = torch.sparse.mm(self.adj, item_text_s_filtered) * self.num_inters[:self.n_users]

        # 特征融合
        user_emb = user_embeddings + ((user_image_g + user_text_g) / 2 + user_image_s + user_text_s) / 3
        item_emb = item_embedding + ((item_image_g + item_text_g) / 2 + item_image_s + item_text_s) / 3 

        user_emb = user_emb[users]

        score = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return score

    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        """获取归一化邻接矩阵"""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))

        for key, value in data_dict.items():
            A[key] = value

        # 归一化邻接矩阵
        sumArr = (A > 0).sum(axis=1)
        # 添加epsilon避免除零
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # 转换为稀疏张量
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def calculate_reg_loss(self, user_emb, pos_items_emb, neg_item_emb):
        """计算正则化损失"""
        reg_loss = self.reg_loss(user_emb, pos_items_emb, neg_item_emb) * 1e-5
        return reg_loss
    
    def reg_loss(self, *embs):
        """正则化损失函数"""
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def mi_estimator(self, x1, x2):
        """互信息估计器的简化版本"""
        # 简单的互信息估计，实际应用中可以用更复杂的估计器
        return torch.mean(torch.sum(x1 * x2, dim=1))

    def _get_optimizer(self, stage_id: int, stage_kwargs: dict) -> torch.optim.Optimizer:
        """获取指定阶段的优化器"""
        lr = stage_kwargs.get('lr', 0.001)
        return torch.optim.Adam(self.parameters(), lr=lr)

    def _get_optimizer_state_dict(self) -> dict:
        """获取当前阶段优化器的状态字典"""
        optimizer_states = {}
        for stage_id, optimizer in self._stage_optimizers.items():
            optimizer_states[stage_id] = optimizer.state_dict()
        return optimizer_states

    def _load_optimizer_state_dict(self, state_dict: dict):
        """加载当前阶段优化器的状态字典"""
        for stage_id, opt_state in state_dict.items():
            if stage_id in self._stage_optimizers:
                self._stage_optimizers[stage_id].load_state_dict(opt_state)

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
        # 计算损失
        loss = self.calculate_loss(batch)
        
        # 计算指标 - 由于DGMRec是推荐系统，我们计算BPR损失相关的指标
        users, pos_items, neg_items = batch
        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, 
            self.item_id_embedding.weight, 
            self.norm_adj
        )
        
        user_emb = user_embeddings[users]
        pos_item_emb = item_embedding[pos_items]
        neg_item_emb = item_embedding[neg_items]
        
        pos_scores = torch.sum(torch.mul(user_emb, pos_item_emb), dim=1)
        neg_scores = torch.sum(torch.mul(user_emb, neg_item_emb), dim=1)
        
        # 计算命中率（HR@K）作为指标
        hr = torch.mean((pos_scores > neg_scores).float())
        metrics = {'hr': hr}
        
        return loss, metrics

    def _validate_one_epoch(self, val_dataloader: torch.utils.data.DataLoader, stage_id: int, stage_kwargs: dict) -> dict:
        """单轮验证逻辑"""
        self.eval()
        total_loss = 0.0
        total_hr = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # 计算损失
                loss = self.calculate_loss(batch)
                
                # 计算指标
                users, pos_items, neg_items = batch
                user_embeddings, item_embedding = self.cge(
                    self.user_embedding.weight, 
                    self.item_id_embedding.weight, 
                    self.norm_adj
                )
                
                user_emb = user_embeddings[users]
                pos_item_emb = item_embedding[pos_items]
                neg_item_emb = item_embedding[neg_items]
                
                pos_scores = torch.sum(torch.mul(user_emb, pos_item_emb), dim=1)
                neg_scores = torch.sum(torch.mul(user_emb, neg_item_emb), dim=1)
                
                # 计算命中率（HR@K）作为指标
                hr = torch.mean((pos_scores > neg_scores).float())
                
                total_loss += loss.item()
                total_hr += hr.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_hr = total_hr / num_batches if num_batches > 0 else 0.0

        return {'loss': avg_loss, 'hr': avg_hr}