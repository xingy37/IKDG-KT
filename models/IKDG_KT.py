"""IKDG-KT: Interaction-Knowledge Dual-Level Dynamic Graph for KT."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TimeDualDecayEncoder
from models.kc_history_manager import KCHistoryManager

# ============================================================================
# IKDG-KT model
# ============================================================================

class IKDG_KT(nn.Module):
    """
    Interaction-Knowledge Dual-Level Dynamic Graph for Knowledge Tracing.

    The model combines event-level student-question encoding with time-aware
    knowledge-concept evolution on a heterogeneous S-Q-KC graph.
    """
    
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16,
                 num_neighbors: int = 50,
                 ablation='-1',
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 num_students: int = None,
                 num_questions: int = None,
                 num_kcs: int = None,
                 lambda_struct: float = 0.1,
                 lambda_decay: float = 0.1,
                 beta: float = 0.5,
                 epsilon: float = 1e-8,
                 result_embed_dim: int = 8,
                 use_attention: bool = True,
                 max_history_length: int = 10,
                 use_time_encoding_in_key: bool = True,
                 use_time_decay_in_attention: bool = False,
                 disable_struct_loss: bool = False,
                 fixed_gate_alpha: float = -1.0,
                 disable_result_embedding_in_key: bool = False,
                 disable_kc_in_fusion: bool = False,
                 history_aggregation: str = 'qkv',
                 ):
        """
        Initialize IKDG-KT.
        """
        super(IKDG_KT, self).__init__()
        
        # 基础配置
        self.num_neighbors = num_neighbors
        self.ablation = ablation
        self.device = device
        self.lambda_struct = lambda_struct
        self.max_history_length = max_history_length
        
        # 对照实验参数
        self.use_time_encoding_in_key = use_time_encoding_in_key
        self.use_time_decay_in_attention = use_time_decay_in_attention
        self.disable_struct_loss = disable_struct_loss
        self.disable_result_embedding_in_key = disable_result_embedding_in_key
        self.disable_kc_in_fusion = disable_kc_in_fusion
        if fixed_gate_alpha is not None and fixed_gate_alpha >= 0:
            if not (0.0 <= float(fixed_gate_alpha) <= 1.0):
                raise ValueError(f"fixed_gate_alpha must be in [0, 1], got {fixed_gate_alpha}")
            self.fixed_gate_alpha = float(fixed_gate_alpha)
        else:
            self.fixed_gate_alpha = None
        if history_aggregation not in {'qkv', 'uniform'}:
            raise ValueError(f"Unsupported history_aggregation: {history_aggregation}")
        self.history_aggregation = history_aggregation
        if not use_attention and self.history_aggregation != 'qkv':
            raise ValueError(
                "history_aggregation only applies when use_attention=True; "
                f"got use_attention={use_attention}, history_aggregation={history_aggregation}"
            )
        
        # 节点类型统计
        self.num_students = num_students
        self.num_questions = num_questions
        self.num_kcs = num_kcs
        
        # 加载原始特征
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        
        # 维度设置
        self.edge_dim = 64
        self.node_dim = 64
        self.time_dim = time_dim
        
        # ===== 原有投影层 =====
        self.projection_layer = nn.ModuleDict({
            'feature_Linear': nn.Linear(self.node_raw_features.shape[-1], self.node_dim),
            'edge': nn.Linear(1, self.node_dim),
            'time': nn.Linear(self.time_dim, self.node_dim),
            'struct': nn.Linear(1, self.node_dim),
        })
        
        # 输出层 (用于基础 DyGKT 部分)
        self.output_layer = nn.Linear(self.node_dim, self.node_dim)
        
        # 序列更新器 (S/Q 节点)
        self.src_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        
        # 时间编码器
        if self.ablation == 'dual':
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        # ===== 新增模块: KC 动态演化与预测 =====
        
        # 1. KC 状态存储 (Memory Bank for KCs)
        if self.num_kcs is not None:
            self.kc_embedding = nn.Embedding(self.num_kcs + 1, self.node_dim)
            nn.init.xavier_uniform_(self.kc_embedding.weight)
        
        # 2. KC 更新组件 - 注意力机制参数
        self.use_attention = use_attention and (self.num_kcs is not None)
        
        # 始终创建简单注意力机制的组件（用于向后兼容和回退）
        self.attn_W = nn.Linear(self.node_dim * 2, self.node_dim)
        self.attn_v = nn.Linear(self.node_dim, 1, bias=False)
        
        if self.use_attention:
            # QKV 注意力机制组件
            # Query: MLP_Q([h_c(t^-) || h_q(t)]) -> (128 -> 64)
            self.MLP_Q = nn.Sequential(
                nn.Linear(self.node_dim * 2, self.node_dim),
                nn.ReLU(),
                nn.Linear(self.node_dim, self.node_dim)
            )
            
            # Key: MLP_K([h_{q_i}(t_i) || Φ(Δt_i) || Embed(r_i)]) -> (64+16+8 -> 64)
            self.result_embed_dim = result_embed_dim
            self.result_embedding = nn.Embedding(2, result_embed_dim)  # 0 or 1
            self.time_encoder_kc = TimeEncoder(time_dim=self.time_dim)
            
            # 带时间编码的 MLP_K（完整版本）
            key_input_dim_with_time = self.node_dim + self.time_dim + result_embed_dim
            self.MLP_K = nn.Sequential(
                nn.Linear(key_input_dim_with_time, self.node_dim),
                nn.ReLU(),
                nn.Linear(self.node_dim, self.node_dim)
            )
            
            # 不带时间编码的 MLP_K（对照实验用）
            if not self.use_time_encoding_in_key:
                key_input_dim_no_time = self.node_dim + result_embed_dim
                self.MLP_K_no_time = nn.Sequential(
                    nn.Linear(key_input_dim_no_time, self.node_dim),
                    nn.ReLU(),
                    nn.Linear(self.node_dim, self.node_dim)
                )
            
            # Value: MLP_V(h_{q_i}(t_i)) -> (64 -> 64)
            self.MLP_V = nn.Sequential(
                nn.Linear(self.node_dim, self.node_dim),
                nn.ReLU(),
                nn.Linear(self.node_dim, self.node_dim)
            )
            
            # 时间衰减参数
            self.lambda_decay = lambda_decay
            self.beta = beta
            self.epsilon = epsilon
            
            # 当前交互权重计算: α_{c,0} = Sigmoid(v^T tanh(W [Q_c(t) || h_q(t)]))
            self.attn_W_gate = nn.Linear(self.node_dim * 2, self.node_dim)
            self.attn_v_gate = nn.Linear(self.node_dim, 1, bias=False)
        
        # KC 状态更新 GRU
        self.kc_gru = nn.GRUCell(self.node_dim, self.node_dim)
        
        # KC 历史管理器（可选，由外部设置）
        self.kc_history_manager = None
        self.target_trace_kc_id = None
        self.kc_trace_history = []
        
        # 3. 最终预测层 (融合 S, Q, KC)
        # Input: h_s (64) + h_q (64) + h_kc (64) -> Output (1)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.node_dim * 3, self.node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.node_dim, 1) # Output logit (before sigmoid)
        )
        
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
    
    def set_kc_history_manager(self, history_manager: KCHistoryManager):
        """设置 KC 历史交互序列管理器"""
        self.kc_history_manager = history_manager

    def _to_edge_ids_tensor(self, edge_ids):
        """
        Convert normalized internal edge ids to a device tensor and validate the
        index range before any GPU gather happens.
        """
        if torch.is_tensor(edge_ids):
            edge_ids_tensor = edge_ids.long().to(self.device)
        else:
            edge_ids_tensor = torch.as_tensor(edge_ids, dtype=torch.long, device=self.device)

        if edge_ids_tensor.numel() == 0:
            return edge_ids_tensor

        min_edge_id = int(edge_ids_tensor.min().item())
        max_edge_id = int(edge_ids_tensor.max().item())
        max_valid_edge_id = self.edge_raw_features.shape[0] - 1
        if min_edge_id < 0 or max_edge_id > max_valid_edge_id:
            raise IndexError(
                f"Normalized edge_ids out of range for IKDG-KT: "
                f"min={min_edge_id}, max={max_edge_id}, valid_range=0..{max_valid_edge_id}."
            )
        return edge_ids_tensor

    def _get_edge_feature_rows(self, edge_ids):
        """Gather edge feature rows using normalized internal edge ids."""
        edge_ids_tensor = self._to_edge_ids_tensor(edge_ids)
        return self.edge_raw_features[edge_ids_tensor]
    
    def is_student_question_edge(self, src_node_ids, dst_node_ids):
        """
        判断是否为 S-Q 边 (辅助方法，用于训练脚本过滤)
        """
        # 如果没有传入具体数量，默认全都是 S-Q 边（兼容旧逻辑）
        if self.num_students is None:
            return torch.ones(len(src_node_ids), dtype=torch.bool, device=self.device)
            
        src_is_student = torch.tensor(src_node_ids < self.num_students, device=self.device)
        return src_is_student

    def forward_with_kc(self, src_node_ids, edge_ids, node_interact_times, dst_node_ids, return_attention=False):
        """
        IKDG-KT core forward pass with QKV history aggregation.
        """
        batch_size = len(src_node_ids)
        
        # 1. 获取基础 DyGKT 嵌入
        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, edge_ids, node_interact_times, dst_node_ids
        )
        
        kc_emb = None
        struct_loss = torch.tensor(0.0, device=self.device)
        attention_weights = None
        
        # 2. 如果具备 KC 信息，进行 KC 更新逻辑
        if self.num_kcs is not None:
            # 2.1 获取当前 Batch 涉及的 KC 索引
            # 假设 node_raw_features 第0列存储了 skill_id (即 KC ID)
            dst_node_indices = torch.from_numpy(dst_node_ids).long().to(self.device)
            batch_kc_ids = self.node_raw_features[dst_node_indices, 0].long()

            # 防止数据中的占位/边界编码导致 KC embedding 索引越界。
            batch_kc_ids = torch.clamp(batch_kc_ids, min=0, max=self.num_kcs)
            
            # 2.2 获取 KC 的上一时刻状态 h_c(t-)
            prev_kc_emb = self.kc_embedding(batch_kc_ids)  # (B, D)
            
            if self.use_attention and self.kc_history_manager is not None:
                # ===== 使用完整的 QKV 注意力机制 =====
                kc_emb, struct_loss, attention_weights = self._update_kc_with_attention(
                    batch_kc_ids, prev_kc_emb, dst_emb, dst_node_ids, 
                    node_interact_times, edge_ids, batch_size
                )
            else:
                # ===== 使用简化的注意力机制（向后兼容） =====
                kc_emb, struct_loss, attention_weights = self._update_kc_simple(
                    batch_kc_ids, prev_kc_emb, dst_emb, batch_size
                )

        if return_attention:
            return src_emb, dst_emb, kc_emb, struct_loss, attention_weights
        return src_emb, dst_emb, kc_emb, struct_loss
    
    def _update_kc_with_attention(self, batch_kc_ids, prev_kc_emb, dst_emb, 
                                   dst_node_ids, node_interact_times, edge_ids, batch_size):
        """
        使用完整的 QKV 注意力机制更新 KC 状态（优化批处理版本）
        实现 DyGKT-attention.md 中的所有公式
        优化：批量处理所有样本，使用矩阵运算提高效率
        """
        # 步骤1: 生成 Query 向量（批处理）
        # Q_c(t) = MLP_Q([h_c(t^-) || h_q(t)])
        query_input = torch.cat([prev_kc_emb, dst_emb], dim=-1)  # (B, 2D)
        Q_c = self.MLP_Q(query_input)  # (B, D)
        
        # 步骤2: 按KC分组，批量处理历史交互（优化版本：方案2）
        if isinstance(batch_kc_ids, torch.Tensor):
            batch_kc_ids = torch.clamp(batch_kc_ids, min=0, max=self.num_kcs)
        else:
            batch_kc_ids = np.clip(batch_kc_ids, 0, self.num_kcs)
            batch_kc_ids = torch.from_numpy(batch_kc_ids).long().to(self.device)

        unique_kc_ids, inverse_indices = torch.unique(batch_kc_ids, return_inverse=True)
        max_history_len = self.max_history_length  # 限制最大历史长度，提高效率
        
        # 为每个样本准备历史数据（批处理）
        device = self.device
        node_interact_times_tensor = torch.from_numpy(node_interact_times).float().to(device)
        edge_feature_rows = self._get_edge_feature_rows(edge_ids)
        edge_results = edge_feature_rows[:, 0].long()
        
        # 预分配张量
        hist_embeddings_batch = torch.zeros(batch_size, max_history_len, self.node_dim, device=device)
        hist_time_deltas_batch = torch.zeros(batch_size, max_history_len, device=device)
        hist_results_batch = torch.zeros(batch_size, max_history_len, dtype=torch.long, device=device)
        hist_masks = torch.zeros(batch_size, max_history_len, dtype=torch.bool, device=device)
        
        # 优化1: 批量类型转换（一次性转换所有类型）
        if isinstance(batch_kc_ids, torch.Tensor):
            batch_kc_ids_np = batch_kc_ids.cpu().numpy()
        else:
            batch_kc_ids_np = np.array(batch_kc_ids)
        
        if isinstance(node_interact_times, np.ndarray):
            node_interact_times_np = node_interact_times
        else:
            node_interact_times_np = np.array(node_interact_times)
        
        # 优化2: 使用批量查询接口，为每个样本使用自己的时间获取历史
        # 构建 (kc_id, current_time) 对列表
        kc_time_pairs = [
            (int(batch_kc_ids_np[i]), float(node_interact_times_np[i]))
            for i in range(batch_size)
        ]
        
        # 批量获取历史（底层优化：为每个唯一的 (kc_id, time) 组合只查询一次）
        kc_time_histories = self.kc_history_manager.get_histories_batch_optimized(
            kc_time_pairs, max_length=max_history_len
        )
        
        # 优化3: 批量填充数据（优化版本：批量tensor创建）
        
        # 预分配numpy数组用于批量收集（避免在循环中重复分配）
        times_buffer = np.zeros((batch_size, max_history_len), dtype=np.float32)
        results_buffer = np.zeros((batch_size, max_history_len), dtype=np.int64)
        
        for i in range(batch_size):
            kc_id_int = int(batch_kc_ids_np[i])
            current_time_float = float(node_interact_times_np[i])
            # 使用每个样本自己的 (kc_id, time) 组合获取历史
            history = kc_time_histories.get((kc_id_int, current_time_float), [])
            
            # Fallback处理
            if len(history) == 0:
                result = int(edge_results[i].item())
                history = [(int(dst_node_ids[i]), current_time_float - 1e-6, result, dst_emb[i].detach().clone())]
            
            hist_len = min(len(history), max_history_len)
            if hist_len > 0:
                # 列表创建
                hist_emb_list = [history[j][3] for j in range(hist_len)]
                
                # Tensor stack操作
                hist_embeddings_batch[i, :hist_len] = torch.stack(hist_emb_list)
                
                # 批量收集到numpy数组（避免在循环中创建tensor）
                for j in range(hist_len):
                    times_buffer[i, j] = history[j][1]
                    results_buffer[i, j] = history[j][2]
                
                # Mask赋值
                hist_masks[i, :hist_len] = True
        
        # 批量转换为tensor（一次性操作，比循环中逐个转换快）
        times_tensor = torch.from_numpy(times_buffer).to(device)  # (B, max_history_len)
        results_tensor = torch.from_numpy(results_buffer).to(device)  # (B, max_history_len)
        
        # 批量计算时间间隔和赋值
        current_times_expanded = torch.from_numpy(node_interact_times_np).float().to(device).unsqueeze(1)  # (B, 1)
        hist_time_deltas_batch = current_times_expanded - times_tensor  # (B, max_history_len)
        hist_results_batch = results_tensor  # (B, max_history_len)
        
        # 应用mask（只保留有效位置）
        hist_time_deltas_batch = hist_time_deltas_batch * hist_masks.float()
        
        # 步骤3: 批量生成 Key 和 Value
        # 结果嵌入: (B, max_history_len) -> (B, max_history_len, result_embed_dim)
        if self.disable_result_embedding_in_key:
            result_embeddings = torch.zeros(
                batch_size, max_history_len, self.result_embed_dim, device=device
            )
        else:
            result_embeddings = self.result_embedding(hist_results_batch)  # (B, max_history_len, result_embed_dim)
        
        # 根据 use_time_encoding_in_key 决定是否在 Key 中包含时间编码
        if self.use_time_encoding_in_key:
            # 时间编码: TimeEncoder 期望输入为 (batch_size, seq_len)
            # hist_time_deltas_batch: (B, max_history_len)
            time_encodings = self.time_encoder_kc(hist_time_deltas_batch)  # (B, max_history_len, time_dim)
            # 生成 Key: (B, max_history_len, D+time_dim+result_embed_dim) -> (B, max_history_len, D)
            key_input = torch.cat([hist_embeddings_batch, time_encodings, result_embeddings], dim=-1)
            K_batch = self.MLP_K(key_input)  # (B, max_history_len, D)
        else:
            # 生成 Key（不包含时间编码）: (B, max_history_len, D+result_embed_dim) -> (B, max_history_len, D)
            key_input = torch.cat([hist_embeddings_batch, result_embeddings], dim=-1)
            K_batch = self.MLP_K_no_time(key_input)  # (B, max_history_len, D)
        
        # 生成 Value: (B, max_history_len, D) -> (B, max_history_len, D)
        V_batch = self.MLP_V(hist_embeddings_batch)  # (B, max_history_len, D)
        
        # 步骤4: 批量计算时序感知的注意力权重
        # 注意力分数: (B, 1, D) @ (B, D, max_history_len) -> (B, 1, max_history_len)
        Q_c_expanded = Q_c.unsqueeze(1)  # (B, 1, D)
        K_batch_transposed = K_batch.transpose(1, 2)  # (B, D, max_history_len)
        dot_product = torch.bmm(Q_c_expanded, K_batch_transposed)  # (B, 1, max_history_len)
        scaled_dot_product = dot_product.squeeze(1) / np.sqrt(self.node_dim)  # (B, max_history_len)
        
        if self.history_aggregation == 'uniform':
            attention_scores = torch.zeros_like(scaled_dot_product)
        else:
            # 根据 use_time_decay_in_attention 决定是否添加时间衰减项
            if self.use_time_decay_in_attention:
                # 时间衰减: (B, max_history_len)
                tau_batch = torch.exp(-self.lambda_decay * hist_time_deltas_batch)  # (B, max_history_len)
                # 时间衰减项
                time_decay_term = self.beta * torch.log(tau_batch + self.epsilon)  # (B, max_history_len)
                attention_scores = scaled_dot_product + time_decay_term  # (B, max_history_len)
            else:
                # 不使用时间衰减，仅使用点积相似度
                attention_scores = scaled_dot_product  # (B, max_history_len)
        
        # 应用mask，将无效位置设为负无穷
        attention_scores = attention_scores.masked_fill(~hist_masks, float('-inf'))
        
        # Softmax 归一化
        alpha_batch = F.softmax(attention_scores, dim=1)  # (B, max_history_len)
        
        # 步骤5: 批量历史交互聚合
        # (B, max_history_len, 1) * (B, max_history_len, D) -> (B, max_history_len, D) -> (B, D)
        m_c_batch = torch.sum(alpha_batch.unsqueeze(-1) * V_batch, dim=1)  # (B, D)
        
        # 步骤6: 批量计算当前交互权重
        if self.fixed_gate_alpha is None:
            gate_input = torch.cat([Q_c, dst_emb], dim=-1)  # (B, 2D)
            gate_hidden = torch.tanh(self.attn_W_gate(gate_input))  # (B, D)
            alpha_c_0_batch = torch.sigmoid(self.attn_v_gate(gate_hidden))  # (B, 1)
        else:
            alpha_c_0_batch = torch.full(
                (batch_size, 1), self.fixed_gate_alpha, device=device, dtype=dst_emb.dtype
            )
        
        # 步骤7: 批量最终聚合消息
        M_c_batch = alpha_c_0_batch * dst_emb + (1 - alpha_c_0_batch) * m_c_batch  # (B, D)
        
        # 步骤8: 按 KC ID 分组聚合（如果同一 KC 有多个样本）
        unique_aggregated_msgs = torch.zeros(len(unique_kc_ids), self.node_dim, device=device)
        unique_aggregated_msgs.index_add_(0, inverse_indices, M_c_batch)
        
        # 获取唯一的 KC 上一时刻状态
        unique_prev_kc_emb = self.kc_embedding(unique_kc_ids)
        
        # 步骤9: GRU 更新 KC 状态
        unique_updated_kc_emb = self.kc_gru(unique_aggregated_msgs, unique_prev_kc_emb)  # (Num_Unique, D)
        self._record_kc_trace(unique_kc_ids, unique_updated_kc_emb)
        
        # 映射回 Batch 维度
        kc_emb = unique_updated_kc_emb[inverse_indices]  # (B, D)
        
        # 步骤10: 批量更新历史管理器（优化版本：批量准备数据）
        # 使用步骤2中已经转换好的numpy数组（batch_kc_ids_np和node_interact_times_np）
        # 如果不存在（理论上不应该），则重新转换
        try:
            batch_kc_ids_np_step10 = batch_kc_ids_np
            node_interact_times_np_step10 = node_interact_times_np
        except NameError:
            # 如果变量不存在，重新转换
            if isinstance(batch_kc_ids, torch.Tensor):
                batch_kc_ids_np_step10 = batch_kc_ids.cpu().numpy()
            else:
                batch_kc_ids_np_step10 = np.array(batch_kc_ids)
            
            if isinstance(node_interact_times, np.ndarray):
                node_interact_times_np_step10 = node_interact_times
            else:
                node_interact_times_np_step10 = np.array(node_interact_times)
        
        # 批量转换dst_node_ids和edge_results
        if isinstance(dst_node_ids, np.ndarray):
            dst_node_ids_np = dst_node_ids
        else:
            dst_node_ids_np = np.array(dst_node_ids)
        
        edge_results_np = edge_results.cpu().numpy()
        
        # 优化2: 批量detach和clone embeddings（一次性操作，比循环中逐个操作快）
        dst_emb_detached = dst_emb.detach().clone()  # 批量detach和clone
        
        # 优化3: 批量更新历史管理器
        for i in range(batch_size):
            kc_id_int = int(batch_kc_ids_np_step10[i])
            q_id_int = int(dst_node_ids_np[i])
            current_time_float = float(node_interact_times_np_step10[i])
            result = int(edge_results_np[i])
            
            self.kc_history_manager.add_interaction(
                kc_id_int, q_id_int, current_time_float, result, dst_emb_detached[i],
                max_history=self.max_history_length
            )
        
        # 计算结构损失（对比学习）
        shuffled_indices = torch.randperm(batch_size, device=device)
        neg_kc_ids = batch_kc_ids[shuffled_indices]
        neg_kc_emb = self.kc_embedding(neg_kc_ids)
        
        pos_score = torch.sum(dst_emb * kc_emb, dim=-1)
        neg_score = torch.sum(dst_emb * neg_kc_emb, dim=-1)
        
        margin = 1.0
        struct_loss = torch.clamp(margin - pos_score + neg_score, min=0.0).mean()
        
        return kc_emb, struct_loss, alpha_batch
    
    def _update_kc_simple(self, batch_kc_ids, prev_kc_emb, dst_emb, batch_size):
        """
        简化的 KC 更新方法（向后兼容，不使用历史交互序列）
        """
        # Query: q_c(t) = MaxPool(h_c(t-), h_q(t))
        query_vec = torch.max(torch.stack([prev_kc_emb, dst_emb], dim=1), dim=1)[0]  # (B, D)
        
        # Attention Score: e = v^T tanh(W [q_c || h_q])
        concat_input = torch.cat([query_vec, dst_emb], dim=-1)  # (B, 2D)
        attn_hidden = torch.tanh(self.attn_W(concat_input))  # (B, D)
        attn_scores = self.attn_v(attn_hidden).squeeze(-1)  # (B,)
        
        # 采用 Sigmoid 近似隶属强度 alpha
        alpha = torch.sigmoid(attn_scores).unsqueeze(-1)  # (B, 1)
        
        # 构建加权消息
        weighted_msg = alpha * dst_emb  # (B, D)
        
        # 聚合消息
        unique_kc_ids, inverse_indices = torch.unique(batch_kc_ids, return_inverse=True)
            
        aggregated_msgs = torch.zeros(len(unique_kc_ids), self.node_dim, device=self.device)
        aggregated_msgs.index_add_(0, inverse_indices, weighted_msg)
            
        # 获取唯一的 KC 上一时刻状态
        unique_prev_kc_emb = self.kc_embedding(unique_kc_ids)
            
        # GRU 更新
        unique_updated_kc_emb = self.kc_gru(aggregated_msgs, unique_prev_kc_emb)  # (Num_Unique, D)
        self._record_kc_trace(unique_kc_ids, unique_updated_kc_emb)
        
        # 映射回 Batch 维度
        kc_emb = unique_updated_kc_emb[inverse_indices]  # (B, D)
        
        # 辅助任务 Loss
        shuffled_indices = torch.randperm(batch_size, device=self.device)
        neg_kc_ids = batch_kc_ids[shuffled_indices]
        neg_kc_emb = self.kc_embedding(neg_kc_ids)
            
        pos_score = torch.sum(dst_emb * kc_emb, dim=-1)
        neg_score = torch.sum(dst_emb * neg_kc_emb, dim=-1)
            
        margin = 1.0
        struct_loss = torch.clamp(margin - pos_score + neg_score, min=0.0).mean()
        
        return kc_emb, struct_loss, None

    def _record_kc_trace(self, unique_kc_ids, unique_updated_kc_emb):
        if self.target_trace_kc_id is None:
            return

        target_kc_id = int(self.target_trace_kc_id)
        trace_mask = unique_kc_ids == target_kc_id
        if not torch.any(trace_mask):
            return

        trace_index = torch.nonzero(trace_mask, as_tuple=False)[0, 0]
        self.kc_trace_history.append(
            unique_updated_kc_emb[trace_index].detach().cpu().numpy().copy()
        )



    def gated_predictor(self, src_emb, dst_emb, kc_emb=None):
        """
        预测层: y = Sigmoid(MLP(h_s + h_q + h_kc))
        """
        if kc_emb is None or self.disable_kc_in_fusion:
            # Fallback if KC not available
            zero_kc = torch.zeros_like(src_emb)
            features = torch.cat([src_emb, dst_emb, zero_kc], dim=-1)
        else:
            features = torch.cat([src_emb, dst_emb, kc_emb], dim=-1)
            
        return self.fusion_layer(features)

    def predict_with_kc(self, src_node_ids, edge_ids, node_interact_times, dst_node_ids, return_attention=False):
        """
        用于评估的预测接口
        """
        if return_attention:
            src_emb, dst_emb, kc_emb, _, attention_weights = self.forward_with_kc(
                src_node_ids, edge_ids, node_interact_times, dst_node_ids, return_attention=True
            )
        else:
            src_emb, dst_emb, kc_emb, _ = self.forward_with_kc(
                src_node_ids, edge_ids, node_interact_times, dst_node_ids
            )
            attention_weights = None
        logits = self.gated_predictor(src_emb, dst_emb, kc_emb)
        return logits, attention_weights

    # ========================================================================
    # 原有 DyGKT 方法保留
    # ========================================================================

    def compute_src_dst_node_temporal_embeddings(self, 
                                                  src_node_ids: np.ndarray, 
                                                  edge_ids: np.ndarray,
                                                  node_interact_times: np.ndarray, 
                                                  dst_node_ids: np.ndarray):
        """
        原有 DyGKT 方法
        """
        
        # ===== 步骤1: 采样历史邻居 =====
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                src_node_ids, node_interact_times, self.num_neighbors
            )
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(
                dst_node_ids, node_interact_times, self.num_neighbors
            )
        
        # ===== 步骤2: 将当前节点添加到邻居序列末尾 =====
        src_neighbor_node_ids = np.concatenate(
            (src_neighbor_node_ids, src_node_ids[:, np.newaxis]), axis=1
        )
        src_neighbor_edge_ids = np.concatenate(
            (src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), 
            axis=1
        )
        src_neighbor_times = np.concatenate(
            (src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1
        )
        
        dst_neighbor_node_ids = np.concatenate(
            (dst_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1
        )
        dst_neighbor_edge_ids = np.concatenate(
            (dst_neighbor_edge_ids, np.zeros((len(dst_node_ids), 1)).astype(np.longlong)), 
            axis=1
        )
        dst_neighbor_times = np.concatenate(
            (dst_neighbor_times, node_interact_times[:, np.newaxis]), axis=1
        )
        
        # ===== 步骤3: 计算结构特征 =====
        src_nodes_neighbor_co_occurrence_features = (
            torch.from_numpy(src_neighbor_node_ids[:, :-1]) == 
            torch.from_numpy(dst_node_ids).unsqueeze(1).repeat(1, self.num_neighbors)
        ).unsqueeze(-1).float().to(self.device)
        
        dst_nodes_neighbor_co_occurrence_features = (
            torch.from_numpy(dst_neighbor_node_ids[:, :-1]) == 
            torch.from_numpy(src_node_ids).unsqueeze(1).repeat(1, self.num_neighbors)
        ).unsqueeze(-1).float().to(self.device)
        
        src_node_skill = self.node_raw_features[torch.from_numpy(src_neighbor_node_ids)][:, :-1, 0].long().to(self.device)
        dst_node_skill = self.node_raw_features[torch.from_numpy(dst_neighbor_node_ids)][:, -1, 0].long().to(self.device).unsqueeze(1).repeat(1, self.num_neighbors)
        src_nodes_neighbor_skill_features = (src_node_skill == dst_node_skill).unsqueeze(-1).float()
        
        a = 0 if self.ablation == 'counter' else 1
        
        src_nodes_neighbor_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer['struct'](a * dst_nodes_neighbor_co_occurrence_features)
        src_nodes_neighbor_skill_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_skill_features)
        
        # ===== 步骤4: 提取特征 =====
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = \
            self.get_features(node_interact_times, src_neighbor_edge_ids, src_neighbor_node_ids, src_neighbor_times)
            
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = \
            self.get_features(node_interact_times, dst_neighbor_edge_ids, dst_neighbor_node_ids, dst_neighbor_times)
        
        # ===== 步骤5: 融合特征 =====
        src_nodes_features = (src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features)
        dst_nodes_features = (dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)
        
        # ===== 步骤6: GRU 更新 =====
        src_node_embeddings = self.src_node_updater.update(
            src_nodes_features[:, :-1, :] + 
            src_nodes_neighbor_skill_struct_features + 
            src_nodes_neighbor_struct_features
        ) + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[:, -1, :]
        
        if self.ablation in ['q_qid', 'q_kid']:
            dst_node_embeddings = dst_nodes_neighbor_node_raw_features[:, -1]
        else:
            dst_node_embeddings = self.dst_node_updater.update(
                (dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[:, :-1, :] + 
                dst_nodes_neighbor_struct_features
            ) + dst_nodes_features[:, -1, :]
        
        # ===== 步骤7: 输出层 =====
        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        return src_node_embeddings, dst_node_embeddings
    
    def get_features(self, 
                     node_interact_times: np.ndarray,
                     nodes_edge_ids: np.ndarray,
                     nodes_neighbor_ids: np.ndarray, 
                     nodes_neighbor_times: np.ndarray):
        """
        提取特征辅助函数
        """
        # Node features
        if self.ablation in ['embed', 'q_kid']:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Embed'](
                self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)].to(self.device)[:, :, 0].long()
            )
        elif self.ablation == 'q_qid':
            nodes_neighbor_node_raw_features = self.projection_layer['node'](
                torch.from_numpy(nodes_neighbor_ids).to(self.device)
            )
        else:
            nodes_neighbor_node_raw_features = self.projection_layer['feature_Linear'](
                self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)].to(self.device)
            )
        
        # Time features
        if self.ablation == 'dual':
            nodes_neighbor_time_features = self.time_encoder(
                torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device)
            )
        else:
            nodes_neighbor_time_features = self.time_encoder(
                torch.from_numpy(nodes_neighbor_times).float().to(self.device)
            )
        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        
        # Edge features
        nodes_edge_raw_features = self.projection_layer['edge'](
            self._get_edge_feature_rows(nodes_edge_ids)[:, :, 0].unsqueeze(-1)
        )
        
        # Ablations
        if self.ablation == 'time':
            nodes_neighbor_time_features *= 0
        elif self.ablation == 'skill':
            nodes_neighbor_node_raw_features *= 0
            
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features


# ============================================================================
# DyKT_Seq 序列更新器类
# ============================================================================

class DyKT_Seq(nn.Module):
    """
    GRU-based序列更新器
    """
    def __init__(self, edge_dim: int, node_dim: int):
        super(DyKT_Seq, self).__init__()
        self.hid_node_updater = nn.GRU(
            input_size=node_dim, # Fixed: edge_dim -> node_dim based on usage
            hidden_size=node_dim, 
            batch_first=True
        )
    
    def update(self, x):
        outputs, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, dim=0)
