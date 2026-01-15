# mclp.py（修复版）
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

# 导入create_more模块
from create_more import MCLPDatasetGenerator, build_mclp_graph, load_dataset, save_dataset
from self_supervised_mclp import SelfSupervisedMCLPWrapper

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class SelfSupervisedMLP(nn.Module):
    """基于自监督嵌入的MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def prepare_self_supervised_data(instance, self_supervised_model, device):
    """使用自监督模型准备数据"""
    # 获取图数据
    try:
        graph = build_mclp_graph(instance)
    except Exception as e:
        print(f"图构建失败: {e}")
        # 创建简单图
        points = instance['points']
        n = len(points)
        
        # 简单邻接：每个节点连接最近的3个邻居
        from create_more import _pairwise_euclidean
        dist_matrix = _pairwise_euclidean(points, points, device)
        
        edge_list = []
        for i in range(n):
            # 获取最近的3个邻居（排除自己）
            distances = dist_matrix[i].clone()
            distances[i] = float('inf')  # 排除自己
            _, indices = torch.topk(distances, min(3, n-1), largest=False)
            for j in indices:
                edge_list.append([i, j.item()])
                edge_list.append([j.item(), i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        
        graph = type('Graph', (), {
            'x': points.to(device).float(),
            'edge_index': edge_index,
            'edge_attr': None,
            'distance_matrix': dist_matrix
        })()
    
    graph.x = graph.x.float().to(device)
    
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        graph.edge_index = graph.edge_index.long().to(device)
    
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr.float().to(device)
    
    # 获取自监督嵌入
    self_supervised_model.model.eval()
    
    with torch.no_grad():
        try:
            embeddings, _ = self_supervised_model.model(
                graph.x, graph.edge_index, graph.edge_attr
            )
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 使用随机嵌入作为后备
            embeddings = torch.randn(len(graph.x), 32, device=device)
    
    # 计算度
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        if graph.edge_index.shape[1] > 0:
            unique, counts = torch.unique(graph.edge_index[0], return_counts=True)
            degree = torch.zeros(len(graph.x), device=device)
            degree[unique] = counts.float()
        else:
            degree = torch.zeros(len(graph.x), device=device)
    else:
        degree = torch.zeros(len(graph.x), device=device)
    
    return {
        'embeddings': embeddings,
        'degree': degree,
        'dist_matrix': graph.distance_matrix.float() if hasattr(graph, 'distance_matrix') else None,
        'coverage_radius': instance.get('coverage_radius'),
        'total_weights': instance.get('total_weights'),
        'instance': instance,
        'graph': graph
    }

def main():
    # 加载数据
    print("正在加载MCLP数据集...")
    try:
        dataset_file = 'mclp_beijing_test_20.pkl'
        if not os.path.exists(dataset_file):
            print("生成新的数据集...")
            generator = MCLPDatasetGenerator(
                num_nodes=100,
                num_instances=10,
                device=device
            )
            train_dataset = generator.generate_dataset()
            save_dataset(train_dataset, dataset_file)
        else:
            train_dataset = load_dataset(dataset_file)
        
        print(f"成功加载数据集，包含 {len(train_dataset)} 个实例")
        
        # 使用小部分数据进行测试
        if len(train_dataset) > 5:
            train_dataset = train_dataset[:3]  # 只用3个实例测试
            print(f"使用前3个实例进行测试")
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 加载自监督模型
    print("\n加载自监督模型...")
    self_supervised = SelfSupervisedMCLPWrapper(device=device)
    
    try:
        # 获取输入维度
        if len(train_dataset) > 0:
            test_instance = train_dataset[0]
            # 简单估计输入维度
            if hasattr(test_instance, 'points'):
                input_dim = 5  # 坐标(2) + 权重(3)
            else:
                input_dim = 5
        else:
            input_dim = 5
            
        self_supervised.initialize_model(input_dim)
        
        # 尝试加载预训练权重
        model_path = 'self_supervised_mclp_simple.pth'
        if os.path.exists(model_path):
            self_supervised.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print("自监督模型加载成功")
        else:
            print("未找到预训练模型，使用随机初始化")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    K = 5  # 设施数量，减少以简化问题
    
    all_results = []
    max_instances_to_process = min(2, len(train_dataset))  # 只处理2个实例
    
    print(f"\n将处理前 {max_instances_to_process} 个实例")
    print(f"设施数量: {K}")
    print("-" * 60)
    
    for index, instance in enumerate(train_dataset[:max_instances_to_process]):
        start_time = time.time()
        
        print(f"\n处理实例 {index}: {instance['name']}")
        print(f"  节点数: {len(instance['points'])}")
        
        try:
            # 准备数据
            data = prepare_self_supervised_data(instance, self_supervised, device)
            
            # 创建并训练MLP
            mlp = SelfSupervisedMLP(
                input_dim=data['embeddings'].shape[1],
                hidden_dim=32,
                output_dim=1
            ).to(device)
            
            optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
            
            # 快速训练
            print("  训练MLP...")
            num_epochs = 50
            
            for epoch in range(num_epochs):
                mlp.train()
                
                # 前向传播
                logits = mlp(data['embeddings']).squeeze()
                probs = torch.sigmoid(logits)
                
                # 计算损失
                if data['dist_matrix'] is not None and data['coverage_radius'] is not None:
                    # 计算每个节点的覆盖潜力
                    coverage_potential = torch.sum(
                        data['dist_matrix'] <= data['coverage_radius'], 
                        dim=1
                    ).float()
                    
                    # 归一化
                    if coverage_potential.max() > 0:
                        coverage_potential = coverage_potential / coverage_potential.max()
                    
                    # 鼓励选择覆盖潜力高的节点
                    loss = -torch.mean(probs * coverage_potential)
                else:
                    # 使用度数作为指导
                    if data['degree'].max() > 0:
                        degree_norm = data['degree'] / data['degree'].max()
                        loss = -torch.mean(probs * degree_norm)
                    else:
                        loss = -torch.mean(probs)
                
                # 设施数量约束
                facility_count = torch.sum(probs)
                count_penalty = torch.abs(facility_count - K) / K
                
                total_loss = loss + 0.1 * count_penalty
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0 and epoch < num_epochs - 1:
                    print(f"    迭代 {epoch}: 损失={total_loss.item():.4f}")
            
            # 求解
            mlp.eval()
            with torch.no_grad():
                logits = mlp(data['embeddings']).squeeze()
                probs = torch.sigmoid(logits)
                
                # 选择概率最高的K个节点
                selected_indices = torch.topk(probs, min(K, len(probs))).indices.cpu().numpy()
                
                # 计算覆盖率
                if data['dist_matrix'] is not None:
                    coverage_radius = data['coverage_radius'] or data['dist_matrix'].max().item() * 0.15
                    
                    if len(selected_indices) > 0:
                        dist_to_selected = data['dist_matrix'][:, selected_indices]
                        min_dist = torch.min(dist_to_selected, dim=1)[0]
                        covered_mask = (min_dist <= coverage_radius)
                        
                        if data['total_weights'] is not None:
                            coverage = torch.sum(data['total_weights'][covered_mask]).item()
                            total_demand = torch.sum(data['total_weights']).item()
                        else:
                            coverage = torch.sum(covered_mask.float()).item()
                            total_demand = len(instance['points'])
                    else:
                        coverage = 0
                        total_demand = len(instance['points'])
                    
                    coverage_percentage = (coverage / total_demand * 100) if total_demand > 0 else 0
                    
                    result = {
                        'instance_id': index,
                        'instance_name': instance['name'],
                        'selected_indices': selected_indices.tolist(),
                        'coverage': coverage,
                        'total_demand': total_demand,
                        'coverage_percentage': coverage_percentage,
                        'n_nodes': len(instance['points']),
                        'K': K,
                        'time': time.time() - start_time
                    }
                    all_results.append(result)
                    
                    print(f"  结果: 需求覆盖 = {coverage:.2f}/{total_demand:.2f} ({coverage_percentage:.1f}%)")
                    print(f"  耗时: {result['time']:.2f}秒")
                    
                    # 可视化结果
                    try:
                        visualize_result(instance, selected_indices, result)
                    except Exception as e:
                        print(f"  可视化失败: {e}")
                        
        except Exception as e:
            print(f"  处理实例失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    if all_results:
        print("\n" + "="*70)
        print("自监督MCLP求解完成!")
        print("="*70)
        
        # 计算统计
        avg_coverage = np.mean([r['coverage_percentage'] for r in all_results])
        avg_time = np.mean([r['time'] for r in all_results])
        
        print(f"平均覆盖率: {avg_coverage:.1f}%")
        print(f"平均耗时: {avg_time:.2f}秒")
        print(f"处理实例数: {len(all_results)}")
        
        # 保存结果
        results_summary = {
            'all_results': all_results,
            'summary': {
                'avg_coverage_percentage': avg_coverage,
                'avg_time': avg_time,
                'num_instances': len(all_results),
                'K': K
            }
        }
        
        result_file = 'self_supervised_mclp_results_simple.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(results_summary, f)
        
        print(f"\n结果已保存到: {result_file}")

def visualize_result(instance, selected_indices, result):
    """可视化单个结果"""
    if not instance or not selected_indices.size:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    points = instance['points'].cpu().numpy()
    selected_points = points[selected_indices]
    
    # 1. 点分布图
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=20, alpha=0.6, label='候选点')
    ax1.scatter(selected_points[:, 0], selected_points[:, 1], 
                s=100, c='red', marker='X', label=f'选定设施(K={result["K"]})', 
                edgecolors='black', linewidth=1.5)
    
    # 绘制服务半径
    coverage_radius = instance.get('coverage_radius', np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.15)
    for idx in selected_indices:
        circle = plt.Circle(points[idx], coverage_radius, 
                           color='red', alpha=0.1, fill=True)
        ax1.add_patch(circle)
    
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title(f'实例 {instance["name"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 覆盖分析
    ax2 = axes[1]
    if 'distance_matrix' in instance and instance['distance_matrix'] is not None:
        dist_matrix = instance['distance_matrix'].cpu().numpy()
        
        # 计算覆盖情况
        coverage_mask = np.min(dist_matrix[:, selected_indices], axis=1) <= coverage_radius
        covered_points = points[coverage_mask]
        uncovered_points = points[~coverage_mask]
        
        ax2.scatter(covered_points[:, 0], covered_points[:, 1], 
                   s=30, c='green', alpha=0.6, label='已覆盖点')
        ax2.scatter(uncovered_points[:, 0], uncovered_points[:, 1], 
                   s=30, c='gray', alpha=0.4, label='未覆盖点')
        ax2.scatter(selected_points[:, 0], selected_points[:, 1], 
                   s=100, c='red', marker='X', label='设施点', edgecolors='black', linewidth=1.5)
        
        # 添加覆盖统计
        stats_text = f"""
        节点数: {result['n_nodes']}
        设施数: {result['K']}
        覆盖率: {result['coverage_percentage']:.1f}%
        耗时: {result['time']:.2f}秒
        """
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title('覆盖分析')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # 显示简单统计
        ax2.text(0.5, 0.5, f"覆盖率: {result['coverage_percentage']:.1f}%", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('结果统计')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'mclp_result_{instance["name"]}.png', dpi=150, bbox_inches='tight')
    plt.show()

# 简化的批量处理函数
def simple_batch_mode():
    """简化版批量处理"""
    print("\n简化版批量处理")
    print("="*60)
    
    # 生成数据
    generator = MCLPDatasetGenerator(
        num_nodes=50,
        num_instances=5,
        device=device
    )
    
    dataset = generator.generate_dataset()
    
    # 加载模型
    self_supervised = SelfSupervisedMCLPWrapper(device=device)
    
    try:
        self_supervised.initialize_model(5)
        model_path = 'self_supervised_mclp_simple.pth'
        if os.path.exists(model_path):
            self_supervised.model.load_state_dict(torch.load(model_path, map_location=device))
            print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试不同K值
    K_options = [3, 5, 7]
    
    for K in K_options:
        print(f"\n测试 K={K}")
        print("-" * 40)
        
        for i, instance in enumerate(dataset[:2]):  # 只测试2个实例
            try:
                selected_indices, coverage = self_supervised.solve_mclp(instance, K=K)
                n_nodes = len(instance['points'])
                coverage_percentage = coverage / n_nodes * 100
                print(f"  实例 {i}: 覆盖率 = {coverage_percentage:.1f}%")
            except Exception as e:
                print(f"  实例 {i} 失败: {e}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("自监督MCLP求解系统")
    print("="*80)
    
    # 选择运行模式
    print("\n选择运行模式:")
    print("1. 标准求解模式 (推荐)")
    print("2. 简化批量处理模式")
    
    try:
        choice = input("请输入选择 (1/2): ").strip()
    except:
        choice = "1"
    
    if choice == "1":
        main()
    elif choice == "2":
        simple_batch_mode()
    else:
        print("无效选择，使用标准模式")
        main()