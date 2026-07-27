"""
KC 历史交互序列管理器
维护每个知识概念（KC）的历史交互序列，用于注意力机制计算
优化版本：使用bisect保持有序，避免每次添加时排序
"""

import numpy as np
import torch
import bisect
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class KCHistoryManager:
    """
    管理每个 KC 的历史交互序列
    存储格式: {(q_i, t_i, r_i, h_{q_i}(t_i))}
    """
    
    def __init__(self, device: str = 'cuda:0'):
        """
        初始化 KC 历史管理器
        :param device: 设备类型
        """
        self.device = device
        # 存储格式: {kc_id: [(q_i, t_i, r_i, h_{q_i}(t_i)), ...]}
        self.history: Dict[int, List[Tuple[int, float, int, torch.Tensor]]] = defaultdict(list)
        
    def add_interaction(self, kc_id: int, question_id: int, timestamp: float, 
                       result: int, question_embedding: torch.Tensor, max_history: int = 100):
        """
        添加一次交互到历史序列
        :param kc_id: 知识概念 ID
        :param question_id: 题目 ID
        :param timestamp: 时间戳
        :param result: 答题结果 (0 或 1)
        :param question_embedding: 题目嵌入向量
        :param max_history: 最大历史长度，超过此长度会删除最旧的记录
        """
        # 确保 embedding 在正确的设备上
        if question_embedding.device != torch.device(self.device):
            question_embedding = question_embedding.to(self.device)
        
        # 断开梯度连接并创建独立副本（避免反向传播错误和内存问题）
        if question_embedding.requires_grad:
            question_embedding = question_embedding.detach().clone()
        else:
            question_embedding = question_embedding.clone()
        
        # 使用bisect插入，保持有序（O(log n)插入，而不是O(n log n)排序）
        # 这样历史列表始终保持按时间戳排序
        new_item = (question_id, timestamp, result, question_embedding)
        
        # 如果列表为空或新时间戳最大，直接追加（最快情况）
        if len(self.history[kc_id]) == 0 or timestamp >= self.history[kc_id][-1][1]:
            self.history[kc_id].append(new_item)
        else:
            # 使用bisect找到插入位置（二分查找，O(log n)）
            insert_pos = bisect.bisect_left(self.history[kc_id], timestamp, key=lambda x: x[1])
            self.history[kc_id].insert(insert_pos, new_item)
        
        # 限制历史长度，只保留最近的 max_history 个（避免内存无限增长）
        if len(self.history[kc_id]) > max_history:
            self.history[kc_id] = self.history[kc_id][-max_history:]
    
    def get_history(self, kc_id: int, current_time: float, 
                   max_length: Optional[int] = None) -> List[Tuple[int, float, int, torch.Tensor]]:
        """
        获取 KC 在指定时间之前的历史交互序列
        优化：使用二分查找，O(log n)而不是O(n)
        :param kc_id: 知识概念 ID
        :param current_time: 当前时间戳
        :param max_length: 最大历史长度（None 表示不限制）
        :return: 历史交互列表，按时间排序
        """
        if kc_id not in self.history:
            return []
        
        history_list = self.history[kc_id]
        if len(history_list) == 0:
            return []
        
        # 使用二分查找找到第一个时间 >= current_time 的位置（O(log n)）
        # 由于列表已排序，可以直接切片，不需要过滤
        insert_pos = bisect.bisect_left(history_list, current_time, key=lambda x: x[1])
        
        # 直接切片获取时间 < current_time 的所有历史（O(1)切片操作）
        history = history_list[:insert_pos]
        
        # 如果指定了最大长度，只返回最近的 max_length 个
        if max_length is not None and len(history) > max_length:
            history = history[-max_length:]
        
        return history
    
    def get_history_batch(self, kc_ids: np.ndarray, current_times: np.ndarray,
                        max_length: Optional[int] = None) -> Dict[int, List[Tuple[int, float, int, torch.Tensor]]]:
        """
        批量获取多个 KC 的历史交互序列
        :param kc_ids: KC ID 数组 (batch_size,)
        :param current_times: 当前时间戳数组 (batch_size,)
        :param max_length: 最大历史长度
        :return: 字典 {kc_id: history_list}
        """
        batch_history = {}
        for kc_id, current_time in zip(kc_ids, current_times):
            kc_id = int(kc_id)
            current_time = float(current_time)
            batch_history[kc_id] = self.get_history(kc_id, current_time, max_length)
        
        return batch_history
    
    def get_histories_batch_optimized(self, kc_time_pairs: List[Tuple[int, float]],
                                      max_length: Optional[int] = None) -> Dict[Tuple[int, float], List[Tuple[int, float, int, torch.Tensor]]]:
        """
        优化的批量获取历史交互序列方法
        为每个唯一的 (kc_id, current_time) 组合批量获取历史，避免重复查询
        
        :param kc_time_pairs: (kc_id, current_time) 元组列表
        :param max_length: 最大历史长度
        :return: 字典 {(kc_id, current_time): history_list}
        
        优化策略：
        1. 按 KC 分组，减少对同一 KC 的重复访问
        2. 为每个唯一的 (kc_id, time) 组合只查询一次
        3. 利用历史列表已排序的特性，使用二分查找加速
        """
        # 去重：为每个唯一的 (kc_id, time) 组合只查询一次
        unique_pairs = {}
        pair_to_indices = {}
        for idx, (kc_id, current_time) in enumerate(kc_time_pairs):
            kc_id_int = int(kc_id)
            current_time_float = float(current_time)
            pair_key = (kc_id_int, current_time_float)
            
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = None  # 占位符，稍后填充
                pair_to_indices[pair_key] = []
            pair_to_indices[pair_key].append(idx)
        
        # 批量查询：为每个唯一的 (kc_id, time) 组合获取历史
        results = {}
        for (kc_id_int, current_time_float) in unique_pairs.keys():
            history = self.get_history(kc_id_int, current_time_float, max_length)
            results[(kc_id_int, current_time_float)] = history
        
        return results
    
    def clear(self):
        """清空所有历史记录"""
        self.history.clear()
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息
        :return: 统计字典
        """
        total_kcs = len(self.history)
        total_interactions = sum(len(h) for h in self.history.values())
        avg_interactions = total_interactions / total_kcs if total_kcs > 0 else 0
        
        return {
            'total_kcs': total_kcs,
            'total_interactions': total_interactions,
            'avg_interactions_per_kc': avg_interactions
        }

