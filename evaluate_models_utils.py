import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.metrics import get_link_classification_metrics
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)
    
    # Initialize the concept-history manager when evaluating IKDG-KT.
    if model_name == 'IKDG-KT':
        from models.kc_history_manager import KCHistoryManager
        if not hasattr(model[0], 'kc_history_manager') or model[0].kc_history_manager is None:
            device = next(model[0].parameters()).device
            kc_history_manager = KCHistoryManager(device=str(device))
            model[0].set_kc_history_manager(kc_history_manager)

    model.eval()
    
    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_predicts, evaluate_labels = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_edge_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],\
                evaluate_data.labels[evaluate_data_indices]
           
            if model_name == 'IKDG-KT':
                # IKDG-KT evaluates only student-question response edges.
                predicts_batch, _ = model[0].predict_with_kc(
                    src_node_ids=batch_src_node_ids,
                    edge_ids=batch_edge_ids,
                    node_interact_times=batch_node_interact_times,
                    dst_node_ids=batch_dst_node_ids
                )
                predicts_batch = predicts_batch.squeeze(dim=-1).sigmoid()
                
                is_sq = model[0].is_student_question_edge(batch_src_node_ids, batch_dst_node_ids)
                
                if is_sq.any():
                    predicts = predicts_batch[is_sq]
                    # Fix: batch_edge_labels is numpy, is_sq is cuda tensor
                    all_labels_tensor = torch.tensor(batch_edge_labels, dtype=torch.float32, device=predicts.device)
                    labels = all_labels_tensor[is_sq]
                    
                    loss = loss_func(input=predicts, target=labels)
                    evaluate_losses.append(loss.item())
                    
                    evaluate_predicts.append(predicts)
                    evaluate_labels.append(labels)
                
                evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch')
                continue

            if model_name in ['DyGKT','DyGKT_KC','DACE','QIKT','IEKT','IPKT','DIMKT','DKT','AKT','CTNCM','simpleKT']:
                
                batch_src_node_embeddings,batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          edge_ids = batch_edge_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          dst_node_ids=batch_dst_node_ids)

            elif model_name in ['TGAT']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

               
            elif model_name in ['TGN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            

            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            predicts = model[1](batch_src_node_embeddings,batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.tensor(batch_edge_labels, dtype=torch.float32,device=predicts.device)            
            
            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())

            evaluate_predicts.append(predicts)
            evaluate_labels.append(labels)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_predict = torch.cat(evaluate_predicts, dim=0)
        evaluate_label = torch.cat(evaluate_labels, dim=0)

        evaluate_metrics.append(get_link_classification_metrics(predicts=evaluate_predict, labels=evaluate_label))

    return evaluate_losses, evaluate_metrics


def evaluate_model_link_classification_by_edge_type(
    model_name: str, 
    model: nn.Module, 
    neighbor_sampler: NeighborSampler, 
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler, 
    evaluate_data: Data, 
    loss_func: nn.Module,
    num_students: int = None,
    num_questions: int = None,
    num_neighbors: int = 20, 
    time_gap: int = 2000
):
    """
    按边类型分别评估模型性能
    
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_students: int, number of student nodes (用于判断边类型)
    :param num_questions: int, number of question nodes (用于判断边类型)
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
        dict: {
            'all': {'losses': [...], 'metrics': [...]},
            'sq': {'losses': [...], 'metrics': [...]},  # 学生-问题边
            'qkc': {'losses': [...], 'metrics': [...]}  # 问题-KC边
        }
    """
    # Ensures the random sampler uses a fixed seed for evaluation
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    # 处理不同的模型格式
    if isinstance(model, nn.Sequential):
        # 如果是 Sequential，第一个元素是 backbone
        model[0].set_neighbor_sampler(neighbor_sampler)
    elif hasattr(model, 'set_neighbor_sampler'):
        # 如果是直接的模型对象，直接调用
        model.set_neighbor_sampler(neighbor_sampler)
    else:
        raise ValueError(f"模型格式不支持: {type(model)}")
    
    model.eval()
    
    # 判断边类型的辅助函数
    def is_student_question_edge(src_ids, dst_ids, num_students, num_questions):
        """判断是否为学生-问题边"""
        if num_students is None or num_questions is None:
            return np.zeros(len(src_ids), dtype=bool)
        question_start = num_students
        question_end = num_students + num_questions
        return (src_ids < question_start) & (dst_ids >= question_start) & (dst_ids < question_end)
    
    def is_question_kc_edge(src_ids, dst_ids, num_students, num_questions):
        """判断是否为问题-KC边"""
        if num_students is None or num_questions is None:
            return np.zeros(len(src_ids), dtype=bool)
        question_start = num_students
        question_end = num_students + num_questions
        return (src_ids >= question_start) & (src_ids < question_end) & (dst_ids >= question_end)
    
    with torch.no_grad():
        # 存储所有边的评估结果
        all_losses, all_predicts, all_labels = [], [], []
        # 存储 S-Q 边的评估结果
        sq_losses, sq_predicts, sq_labels = [], [], []
        # 存储 Q-KC 边的评估结果
        qkc_losses, qkc_predicts, qkc_labels = [], [], []
        
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_edge_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices],\
                evaluate_data.labels[evaluate_data_indices]
            
            # 判断边类型
            is_sq = is_student_question_edge(batch_src_node_ids, batch_dst_node_ids, num_students, num_questions)
            is_qkc = is_question_kc_edge(batch_src_node_ids, batch_dst_node_ids, num_students, num_questions)
            
            # 获取模型组件
            if isinstance(model, nn.Sequential):
                backbone = model[0]
                predictor = model[1]
            else:
                # 如果不是 Sequential，假设模型有这些方法
                backbone = model
                predictor = model  # 需要根据实际情况调整
            
            # 计算节点嵌入
            if model_name in ['DyGKT','DyGKT_KC','IKDG-KT','DACE','QIKT','IEKT','IPKT','DIMKT','DKT','AKT','CTNCM','simpleKT']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        edge_ids=batch_edge_ids,
                        node_interact_times=batch_node_interact_times,
                        dst_node_ids=batch_dst_node_ids
                    )
            elif model_name in ['TGAT']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors
                    )
            elif model_name in ['TGN']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                        edges_are_positive=True,
                        num_neighbors=num_neighbors
                    )
            elif model_name in ['DyGFormer']:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times
                    )
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            
            # 获取预测结果
            if isinstance(model, nn.Sequential):
                predicts = predictor(batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            else:
                # 如果不是 Sequential，需要根据实际情况调整
                predicts = predictor(batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            labels = torch.tensor(batch_edge_labels, dtype=torch.float32, device=predicts.device)
            
            # 计算所有边的损失
            loss = loss_func(input=predicts, target=labels)
            all_losses.append(loss.item())
            all_predicts.append(predicts)
            all_labels.append(labels)
            
            # 分别处理 S-Q 边
            if np.any(is_sq):
                sq_mask = torch.from_numpy(is_sq).bool().to(predicts.device)
                sq_predict = predicts[sq_mask]
                sq_label = labels[sq_mask]
                sq_loss = loss_func(input=sq_predict, target=sq_label)
                sq_losses.append(sq_loss.item())
                sq_predicts.append(sq_predict)
                sq_labels.append(sq_label)
            
            # 分别处理 Q-KC 边
            if np.any(is_qkc):
                qkc_mask = torch.from_numpy(is_qkc).bool().to(predicts.device)
                qkc_predict = predicts[qkc_mask]
                qkc_label = labels[qkc_mask]
                qkc_loss = loss_func(input=qkc_predict, target=qkc_label)
                qkc_losses.append(qkc_loss.item())
                qkc_predicts.append(qkc_predict)
                qkc_labels.append(qkc_label)
            
            # 更新进度条
            sq_count = is_sq.sum()
            qkc_count = is_qkc.sum()
            other_count = len(is_sq) - sq_count - qkc_count
            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate batch {batch_idx + 1}, loss: {loss.item():.4f}, '
                f'S-Q: {sq_count}, Q-KC: {qkc_count}, Other: {other_count}'
            )
        
        # 确定设备（从第一个batch的predicts获取）
        device = all_predicts[0].device if all_predicts else torch.device('cpu')
        
        # 计算所有边的指标
        all_predict = torch.cat(all_predicts, dim=0) if all_predicts else torch.tensor([], device=device)
        all_label = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([], device=device)
        all_metrics = [get_link_classification_metrics(predicts=all_predict, labels=all_label)] if len(all_predict) > 0 else []
        
        # 计算 S-Q 边的指标
        sq_predict = torch.cat(sq_predicts, dim=0) if sq_predicts else torch.tensor([], device=device)
        sq_label = torch.cat(sq_labels, dim=0) if sq_labels else torch.tensor([], device=device)
        sq_metrics = [get_link_classification_metrics(predicts=sq_predict, labels=sq_label)] if len(sq_predict) > 0 else []
        
        # 计算 Q-KC 边的指标
        qkc_predict = torch.cat(qkc_predicts, dim=0) if qkc_predicts else torch.tensor([], device=device)
        qkc_label = torch.cat(qkc_labels, dim=0) if qkc_labels else torch.tensor([], device=device)
        qkc_metrics = [get_link_classification_metrics(predicts=qkc_predict, labels=qkc_label)] if len(qkc_predict) > 0 else []
        
        # 返回结果
        result = {
            'all': {
                'losses': all_losses,
                'metrics': all_metrics,
                'num_edges': len(all_predict)
            },
            'sq': {
                'losses': sq_losses,
                'metrics': sq_metrics,
                'num_edges': len(sq_predict)
            },
            'qkc': {
                'losses': qkc_losses,
                'metrics': qkc_metrics,
                'num_edges': len(qkc_predict)
            }
        }
        
        return result
