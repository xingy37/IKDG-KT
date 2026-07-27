#!/usr/bin/env python3
"""
使用示例：按边类型分别评估模型性能

这个脚本展示了如何使用 evaluate_model_link_classification_by_edge_type 函数
来分别评估 S-Q 边和 Q-KC 边的性能。
"""

import pickle
import os
import torch
import torch.nn as nn
from evaluate_models_utils import evaluate_model_link_classification_by_edge_type
from utils.DataLoader import get_link_classification_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from models.DyGKT_KC import DyGKT_KC
from models.DyGKT import DyGKT
from models.modules import MergeLayer

def load_mapping_info(dataset_name):
    """加载映射信息"""
    mapping_info_path = f'./processed_data/{dataset_name}/mapping_info.pkl'
    if os.path.exists(mapping_info_path):
        with open(mapping_info_path, 'rb') as f:
            mapping_info = pickle.load(f)
            num_students = len(mapping_info.get('student_to_node_id', {}))
            num_questions = len(mapping_info.get('question_to_node_id', {}))
            num_kcs = len(mapping_info.get('kc_to_node_id', {}))
            return num_students, num_questions, num_kcs
    return None, None, None

def evaluate_by_edge_type_example(dataset_name='dbe_kt22_higher', model_path=None):
    """
    示例：按边类型分别评估模型
    
    Args:
        dataset_name: 数据集名称
        model_path: 模型路径（如果为None，则创建一个新模型用于演示）
    """
    # 1. 加载数据
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_classification_data(dataset_name=dataset_name, val_ratio=0.15, test_ratio=0.15)
    
    # 2. 加载映射信息
    print("\n" + "=" * 60)
    print("步骤2: 加载节点类型映射信息")
    print("=" * 60)
    num_students, num_questions, num_kcs = load_mapping_info(dataset_name)
    if num_students is None:
        print("警告: 无法加载映射信息，将无法区分边类型")
    else:
        print(f"学生节点数: {num_students}")
        print(f"问题节点数: {num_questions}")
        print(f"KC节点数: {num_kcs}")
    
    # 3. 创建或加载模型
    print("\n" + "=" * 60)
    print("步骤3: 创建/加载模型")
    print("=" * 60)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if model_path and os.path.exists(model_path):
        # 从路径判断模型类型
        if 'DyGKT_KC' in model_path:
            model_name = 'DyGKT_KC'
            print(f"从 {model_path} 加载 DyGKT_KC 模型...")
            dynamic_backbone = DyGKT_KC(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=0.5,
                num_neighbors=50,
                device=device,
                ablation='-1',
                num_students=num_students,
                num_questions=num_questions,
                num_kcs=num_kcs
            )
        elif 'DyGKT' in model_path:
            model_name = 'DyGKT'
            print(f"从 {model_path} 加载 DyGKT 模型...")
            dynamic_backbone = DyGKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=0.5,
                num_neighbors=50,
                device=device,
                ablation='-1'
            )
        else:
            raise ValueError(f"无法从路径判断模型类型: {model_path}")
        
        link_predictor = MergeLayer(input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = model.to(device)
        
        # 加载模型权重
        try:
            loaded_data = torch.load(model_path, map_location=device, weights_only=False)
            # 检查是否是 state_dict 还是完整模型
            if isinstance(loaded_data, dict):
                # 是 state_dict
                model.load_state_dict(loaded_data)
                print("成功加载模型权重 (state_dict)")
            elif isinstance(loaded_data, nn.Module):
                # 是完整模型，检查格式
                if isinstance(loaded_data, nn.Sequential):
                    model = loaded_data.to(device)
                    print("成功加载完整模型 (Sequential)")
                else:
                    # 如果不是 Sequential，尝试提取组件
                    print("警告: 加载的模型不是 Sequential 格式，尝试使用 state_dict 方式...")
                    model.load_state_dict(loaded_data.state_dict())
                    print("成功加载模型权重 (从完整模型提取 state_dict)")
            else:
                # 尝试作为 state_dict 加载
                try:
                    model.load_state_dict(loaded_data)
                    print("成功加载模型权重 (尝试作为 state_dict)")
                except Exception as e2:
                    print(f"无法加载模型: {e2}")
                    raise
        except Exception as e:
            print(f"警告: 加载模型时出错: {e}")
            print("将使用未训练的模型进行评估")
    else:
        # 创建新模型（仅用于演示）
        model_name = 'DyGKT_KC'
        print("创建新模型（仅用于演示）...")
        dynamic_backbone = DyGKT_KC(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            dropout=0.5,
            num_neighbors=50,
            device=device,
            ablation='-1',
            num_students=num_students,
            num_questions=num_questions,
            num_kcs=num_kcs
        )
        link_predictor = MergeLayer(input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = model.to(device)
    
    # 4. 准备评估数据
    print("\n" + "=" * 60)
    print("步骤4: 准备评估数据")
    print("=" * 60)
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data, 
        sample_neighbor_strategy='recent',
        time_scaling_factor=1.0, 
        seed=1
    )
    
    # 准备所有数据集的 DataLoader 和 NegativeEdgeSampler
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))), 
        batch_size=200, 
        shuffle=False
    )
    
    new_node_val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_val_data.src_node_ids))), 
        batch_size=200, 
        shuffle=False
    )
    
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))), 
        batch_size=200, 
        shuffle=False
    )
    
    new_node_test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.src_node_ids))), 
        batch_size=200, 
        shuffle=False
    )
    
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, 
        dst_node_ids=full_data.dst_node_ids, 
        seed=0
    )
    
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids, 
        dst_node_ids=new_node_val_data.dst_node_ids, 
        seed=1
    )
    
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, 
        dst_node_ids=full_data.dst_node_ids, 
        seed=2
    )
    
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids, 
        dst_node_ids=new_node_test_data.dst_node_ids, 
        seed=3
    )
    
    loss_func = nn.BCELoss()
    
    # 5. 执行按边类型的评估 - 对所有数据集进行评估
    print("\n" + "=" * 60)
    print("步骤5: 执行按边类型的评估")
    print("=" * 60)
    
    # 评估 validation 集
    print("\n评估 Validation 集...")
    val_results = evaluate_model_link_classification_by_edge_type(
        model_name=model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        evaluate_idx_data_loader=val_idx_data_loader,
        evaluate_neg_edge_sampler=val_neg_edge_sampler,
        evaluate_data=val_data,
        loss_func=loss_func,
        num_students=num_students,
        num_questions=num_questions,
        num_neighbors=50
    )
    
    # 评估 new node validation 集
    print("\n评估 New Node Validation 集...")
    new_node_val_results = evaluate_model_link_classification_by_edge_type(
        model_name=model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        evaluate_idx_data_loader=new_node_val_idx_data_loader,
        evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
        evaluate_data=new_node_val_data,
        loss_func=loss_func,
        num_students=num_students,
        num_questions=num_questions,
        num_neighbors=50
    )
    
    # 评估 test 集
    print("\n评估 Test 集...")
    test_results = evaluate_model_link_classification_by_edge_type(
        model_name=model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        evaluate_idx_data_loader=test_idx_data_loader,
        evaluate_neg_edge_sampler=test_neg_edge_sampler,
        evaluate_data=test_data,
        loss_func=loss_func,
        num_students=num_students,
        num_questions=num_questions,
        num_neighbors=50
    )
    
    # 评估 new node test 集
    print("\n评估 New Node Test 集...")
    new_node_test_results = evaluate_model_link_classification_by_edge_type(
        model_name=model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        evaluate_idx_data_loader=new_node_test_idx_data_loader,
        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
        evaluate_data=new_node_test_data,
        loss_func=loss_func,
        num_students=num_students,
        num_questions=num_questions,
        num_neighbors=50
    )
    
    # 6. 输出学生-问题边（S-Q边）的指标
    print("\n" + "=" * 60)
    print("步骤6: 学生-问题边 (S-Q) 评估结果")
    print("=" * 60)
    
    def get_sq_metrics(results_dict):
        """从结果字典中提取S-Q边的指标"""
        if results_dict['sq']['num_edges'] > 0 and results_dict['sq']['metrics']:
            metrics = results_dict['sq']['metrics'][0]
            return {
                'average_precision': metrics.get('average_precision', 0.0),
                'roc_auc': metrics.get('roc_auc', 0.0)
            }
        return {
            'average_precision': 0.0,
            'roc_auc': 0.0
        }
    
    # 提取各数据集的S-Q边指标
    val_sq_metrics = get_sq_metrics(val_results)
    new_node_val_sq_metrics = get_sq_metrics(new_node_val_results)
    test_sq_metrics = get_sq_metrics(test_results)
    new_node_test_sq_metrics = get_sq_metrics(new_node_test_results)
    
    # 输出指标
    print("\n【Validation 集 - S-Q边】")
    print(f"  average_precision: {val_sq_metrics['average_precision']:.4f}")
    print(f"  roc_auc: {val_sq_metrics['roc_auc']:.4f}")
    print(f"  边数量: {val_results['sq']['num_edges']}")
    
    print("\n【New Node Validation 集 - S-Q边】")
    print(f"  average_precision: {new_node_val_sq_metrics['average_precision']:.4f}")
    print(f"  roc_auc: {new_node_val_sq_metrics['roc_auc']:.4f}")
    print(f"  边数量: {new_node_val_results['sq']['num_edges']}")
    
    print("\n【Test 集 - S-Q边】")
    print(f"  average_precision: {test_sq_metrics['average_precision']:.4f}")
    print(f"  roc_auc: {test_sq_metrics['roc_auc']:.4f}")
    print(f"  边数量: {test_results['sq']['num_edges']}")
    
    print("\n【New Node Test 集 - S-Q边】")
    print(f"  average_precision: {new_node_test_sq_metrics['average_precision']:.4f}")
    print(f"  roc_auc: {new_node_test_sq_metrics['roc_auc']:.4f}")
    print(f"  边数量: {new_node_test_results['sq']['num_edges']}")
    
    # 7. 汇总输出（按用户要求的格式）
    print("\n" + "=" * 60)
    print("步骤7: 学生-问题边指标汇总")
    print("=" * 60)
    print(f"validate average_precision: {val_sq_metrics['average_precision']:.4f}")
    print(f"validate roc_auc: {val_sq_metrics['roc_auc']:.4f}")
    print(f"new node validate average_precision: {new_node_val_sq_metrics['average_precision']:.4f}")
    print(f"new node validate roc_auc: {new_node_val_sq_metrics['roc_auc']:.4f}")
    print(f"test average_precision: {test_sq_metrics['average_precision']:.4f}")
    print(f"test roc_auc: {test_sq_metrics['roc_auc']:.4f}")
    print(f"new node test average_precision: {new_node_test_sq_metrics['average_precision']:.4f}")
    print(f"new node test roc_auc: {new_node_test_sq_metrics['roc_auc']:.4f}")
    
    # 返回所有结果
    all_results = {
        'val': val_results,
        'new_node_val': new_node_val_results,
        'test': test_results,
        'new_node_test': new_node_test_results,
        'sq_metrics': {
            'val': val_sq_metrics,
            'new_node_val': new_node_val_sq_metrics,
            'test': test_sq_metrics,
            'new_node_test': new_node_test_sq_metrics
        }
    }
    
    return all_results

if __name__ == "__main__":
    # 使用示例
    results = evaluate_by_edge_type_example(
        dataset_name='assist17_higher',
        model_path='/private/DyGKT-main/saved_models/DyGKT/assist17_higher/DyGKT_seed0/DyGKT_seed0.pkl'  # 设置为模型路径以评估已训练的模型
    )
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

