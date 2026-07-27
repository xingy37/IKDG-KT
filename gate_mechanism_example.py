#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
门控机制融合问题-KC边特征示例代码
用于演示门控机制的计算过程
"""

import torch
import torch.nn as nn
import numpy as np


def visualize_gate_mechanism():
    """
    可视化门控机制的计算过程
    """
    print("=" * 80)
    print("门控机制融合问题-KC边特征示例")
    print("=" * 80)
    
    # ===== 1. 模拟数据 =====
    print("\n【步骤1】模拟数据")
    print("-" * 80)
    
    num_qkc_edges = 3  # 3条问题-KC边
    node_dim = 8  # 节点嵌入维度
    
    # 模拟基础嵌入（来自GRU更新器）
    question_base_emb = torch.randn(num_qkc_edges, node_dim)
    kc_base_emb = torch.randn(num_qkc_edges, node_dim)
    
    # 模拟全局特征（来自问题-KC边的特殊特征）
    qkc_global_feature = torch.randn(num_qkc_edges, node_dim)
    
    print(f"问题节点基础嵌入形状: {question_base_emb.shape}")
    print(f"问题节点基础嵌入（第一条边，前4个维度）: {question_base_emb[0, :4]}")
    print(f"\n全局特征形状: {qkc_global_feature.shape}")
    print(f"全局特征（第一条边，前4个维度）: {qkc_global_feature[0, :4]}")
    
    # ===== 2. 创建门控层 =====
    print("\n【步骤2】创建门控层")
    print("-" * 80)
    
    gate_layer = nn.Sequential(
        nn.Linear(node_dim, node_dim),
        nn.Sigmoid()
    )
    
    print(f"门控层结构: Linear({node_dim}, {node_dim}) → Sigmoid()")
    print(f"参数数量: {sum(p.numel() for p in gate_layer.parameters())}")
    
    # ===== 3. 计算门控值 =====
    print("\n【步骤3】计算门控值")
    print("-" * 80)
    
    # 问题节点的门控值
    gate_q = gate_layer(question_base_emb)
    print(f"问题节点门控值形状: {gate_q.shape}")
    print(f"问题节点门控值（第一条边，前4个维度）: {gate_q[0, :4]}")
    print(f"门控值范围: [{gate_q.min():.4f}, {gate_q.max():.4f}]")
    print("  → 门控值在[0, 1]范围内，表示全局特征的权重")
    
    # KC节点的门控值
    gate_kc = gate_layer(kc_base_emb)
    print(f"\nKC节点门控值形状: {gate_kc.shape}")
    print(f"KC节点门控值（第一条边，前4个维度）: {gate_kc[0, :4]}")
    
    # ===== 4. 融合特征 =====
    print("\n【步骤4】融合特征（门控机制）")
    print("-" * 80)
    
    # 问题节点融合
    question_emb = gate_q * qkc_global_feature + (1 - gate_q) * question_base_emb
    print(f"问题节点最终嵌入形状: {question_emb.shape}")
    print(f"问题节点最终嵌入（第一条边，前4个维度）: {question_emb[0, :4]}")
    
    # 展示融合过程（第一条边，维度0）
    print("\n融合过程示例（第一条边，维度0）:")
    print(f"  基础嵌入[0, 0] = {question_base_emb[0, 0]:.4f}")
    print(f"  全局特征[0, 0] = {qkc_global_feature[0, 0]:.4f}")
    print(f"  门控值[0, 0] = {gate_q[0, 0]:.4f}")
    print(f"  最终嵌入[0, 0] = {gate_q[0, 0]:.4f} × {qkc_global_feature[0, 0]:.4f} + {1 - gate_q[0, 0]:.4f} × {question_base_emb[0, 0]:.4f}")
    print(f"                = {gate_q[0, 0] * qkc_global_feature[0, 0]:.4f} + {(1 - gate_q[0, 0]) * question_base_emb[0, 0]:.4f}")
    print(f"                = {question_emb[0, 0]:.4f}")
    
    # KC节点融合
    kc_emb = gate_kc * qkc_global_feature + (1 - gate_kc) * kc_base_emb
    print(f"\nKC节点最终嵌入形状: {kc_emb.shape}")
    print(f"KC节点最终嵌入（第一条边，前4个维度）: {kc_emb[0, :4]}")
    
    # ===== 5. 对比：门控机制 vs 直接相加 =====
    print("\n【步骤5】对比：门控机制 vs 直接相加")
    print("-" * 80)
    
    # 直接相加（消融实验方法）
    question_emb_add = question_base_emb + qkc_global_feature
    kc_emb_add = kc_base_emb + qkc_global_feature
    
    print("门控机制方法:")
    print(f"  问题节点嵌入（第一条边，前4个维度）: {question_emb[0, :4]}")
    print("\n直接相加方法:")
    print(f"  问题节点嵌入（第一条边，前4个维度）: {question_emb_add[0, :4]}")
    print("\n差异:")
    diff = question_emb - question_emb_add
    print(f"  差异（第一条边，前4个维度）: {diff[0, :4]}")
    print("  → 可以看到两种方法的结果不同，门控机制能够自适应调整融合比例")
    
    # ===== 6. 门控值的语义解释 =====
    print("\n【步骤6】门控值的语义解释")
    print("-" * 80)
    
    # 创建一个简单的示例
    base_emb_example = torch.tensor([[0.8, 0.2, 0.9, 0.1]])
    global_feat_example = torch.tensor([[0.3, 0.7, 0.4, 0.6]])
    
    # 手动设置门控值（用于演示）
    gate_example = torch.tensor([[0.9, 0.3, 0.8, 0.2]])
    
    final_emb_example = gate_example * global_feat_example + (1 - gate_example) * base_emb_example
    
    print("示例:")
    print(f"  基础嵌入: {base_emb_example[0].tolist()}")
    print(f"  全局特征: {global_feat_example[0].tolist()}")
    print(f"  门控值:   {gate_example[0].tolist()}")
    print(f"  最终嵌入: {final_emb_example[0].tolist()}")
    print("\n解释:")
    print("  维度0: gate=0.9 → 90%使用全局特征(0.3)，10%使用基础嵌入(0.8)")
    print(f"        结果: 0.9×0.3 + 0.1×0.8 = {0.9*0.3 + 0.1*0.8:.2f}")
    print("  维度1: gate=0.3 → 30%使用全局特征(0.7)，70%使用基础嵌入(0.2)")
    print(f"        结果: 0.3×0.7 + 0.7×0.2 = {0.3*0.7 + 0.7*0.2:.2f}")
    print("  维度2: gate=0.8 → 80%使用全局特征(0.4)，20%使用基础嵌入(0.9)")
    print(f"        结果: 0.8×0.4 + 0.2×0.9 = {0.8*0.4 + 0.2*0.9:.2f}")
    print("  维度3: gate=0.2 → 20%使用全局特征(0.6)，80%使用基础嵌入(0.1)")
    print(f"        结果: 0.2×0.6 + 0.8×0.1 = {0.2*0.6 + 0.8*0.1:.2f}")
    print("\n结论:")
    print("  - 门控值越大，全局特征的贡献越大")
    print("  - 门控值越小，基础嵌入的贡献越大")
    print("  - 门控机制允许模型为每个维度学习最优的融合比例")
    
    # ===== 7. 门控值的梯度流 =====
    print("\n【步骤7】门控值的梯度流（训练过程）")
    print("-" * 80)
    
    # 设置requires_grad=True以便计算梯度
    question_base_emb.requires_grad_(True)
    qkc_global_feature.requires_grad_(True)
    gate_layer.train()
    
    # 前向传播
    gate_q = gate_layer(question_base_emb)
    question_emb = gate_q * qkc_global_feature + (1 - gate_q) * question_base_emb
    
    # 模拟损失（假设是预测值）
    loss = question_emb.sum()
    
    # 反向传播
    loss.backward()
    
    print("梯度信息:")
    print(f"  基础嵌入的梯度形状: {question_base_emb.grad.shape}")
    print(f"  全局特征的梯度形状: {qkc_global_feature.grad.shape}")
    print(f"  门控层权重的梯度: 存在（已通过反向传播计算）")
    print("\n说明:")
    print("  - 梯度会流经门控层，更新门控层的权重")
    print("  - 门控层会学习：什么时候全局特征更重要")
    print("  - 如果全局特征对预测有帮助，门控值会增大")
    print("  - 如果基础嵌入更重要，门控值会减小")


def demonstrate_gate_learning():
    """
    演示门控机制如何学习
    """
    print("\n" + "=" * 80)
    print("门控机制学习过程演示")
    print("=" * 80)
    
    node_dim = 4
    num_samples = 10
    
    # 创建简单的训练数据
    # 场景1：基础嵌入强，全局特征弱 → 门控值应该小
    base_emb_strong = torch.randn(num_samples, node_dim) * 2  # 放大基础嵌入
    global_feat_weak = torch.randn(num_samples, node_dim) * 0.5  # 缩小全局特征
    
    # 场景2：基础嵌入弱，全局特征强 → 门控值应该大
    base_emb_weak = torch.randn(num_samples, node_dim) * 0.5
    global_feat_strong = torch.randn(num_samples, node_dim) * 2
    
    # 创建门控层
    gate_layer = nn.Sequential(
        nn.Linear(node_dim, node_dim),
        nn.Sigmoid()
    )
    
    print("\n场景1：基础嵌入强，全局特征弱")
    print("-" * 80)
    gate1 = gate_layer(base_emb_strong)
    print(f"平均门控值: {gate1.mean():.4f}")
    print("  → 理想情况下，门控值应该较小（减少全局特征的贡献）")
    
    print("\n场景2：基础嵌入弱，全局特征强")
    print("-" * 80)
    gate2 = gate_layer(base_emb_weak)
    print(f"平均门控值: {gate2.mean():.4f}")
    print("  → 理想情况下，门控值应该较大（增加全局特征的贡献）")
    
    print("\n注意:")
    print("  - 由于是随机初始化，门控值可能不符合预期")
    print("  - 通过训练，门控层会学习到正确的模式")
    print("  - 训练后，场景1的门控值应该 < 场景2的门控值")


if __name__ == '__main__':
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行示例
    visualize_gate_mechanism()
    demonstrate_gate_learning()
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)




