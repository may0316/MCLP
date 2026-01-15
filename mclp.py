import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from create_more import MCLPDatasetGenerator, build_mclp_graph, load_dataset, save_dataset
from self_supervised_mclp import SelfSupervisedMCLPWrapper

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MCLPOptimizer(nn.Module):
    """MCLP优化器（基于FLP的梯度优化）"""
    def __init__(self, embedding_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits.squeeze()
    
    def solve(self, embeddings, K, temperature=10.0, num_samples=10):
        """求解MCLP（连续松弛+随机舍入）"""
        self.eval()
        with torch.no_grad():
            # 计算每个节点的得分
            logits = self(embeddings)
            probs = torch.sigmoid(logits)
            
            best_indices = None
            best_coverage = -1
            
            # 多次采样取最佳
            for _ in range(num_samples):
                # Gumbel-Softmax采样
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs)))
                gumbel_logits = (logits + gumbel_noise) / temperature
                gumbel_probs = torch.sigmoid(gumbel_logits)
                
                # 选择概率最高的K个
                _, candidate_indices = torch.topk(gumbel_probs, min(K, len(probs)))
                
                # 计算覆盖分数（使用嵌入相似度作为代理）
                candidate_embeddings = embeddings[candidate_indices]
                similarities = torch.matmul(embeddings, candidate_embeddings.T)
                max_similarities, _ = torch.max(similarities, dim=1)
                coverage_score = torch.sum(max_similarities).item()
                
                if coverage_score > best_coverage:
                    best_coverage = coverage_score
                    best_indices = candidate_indices.cpu().numpy()
            
            return best_indices, best_coverage
def train_optimizer(embeddings, K, optimizer_model, num_iterations=500, lr=0.01):
    """训练优化器"""
    opt_optimizer = torch.optim.Adam(optimizer_model.parameters(), lr=lr)
    losses = []
    
    n = embeddings.shape[0]
    
    for iteration in range(num_iterations):
        # 前向传播
        logits = optimizer_model(embeddings)
        probs = torch.sigmoid(logits)
        
        # 设施数量约束损失
        facility_count = torch.sum(probs)
        count_loss = torch.abs(facility_count - K) / K
        
        # 覆盖分散损失（鼓励分散分布）
        if hasattr(embeddings, 'device'):
            device = embeddings.device
        else:
            device = 'cpu'
            
        # 计算节点间的相似度
        similarities = torch.matmul(embeddings, embeddings.T)
        
        # 创建掩码排除对角线
        diag_mask = torch.eye(n, device=device).bool()
        non_diag_mask = ~diag_mask
        
        # 只考虑非对角线元素
        if n > 1:
            # 获取非对角线元素的索引
            row_indices, col_indices = torch.where(non_diag_mask)
            
            # 计算非对角线相似度的平均值
            non_diag_similarities = similarities[row_indices, col_indices]
            
            # 惩罚选择相似节点 - 使用加权方法
            # 先展开probs矩阵
            probs_i = probs[row_indices]
            probs_j = probs[col_indices]
            
            dispersion_loss = torch.mean(probs_i * probs_j * non_diag_similarities)
        else:
            dispersion_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss = count_loss + 0.3 * dispersion_loss
        
        # 反向传播
        opt_optimizer.zero_grad()
        total_loss.backward()
        opt_optimizer.step()
        
        losses.append(total_loss.item())
        
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}: Loss={total_loss.item():.4f}, Count={facility_count.item():.2f}")
    
    return losses

def main():
    # 加载数据
    print("正在加载文旅MCLP数据集...")
    try:
        dataset_file = 'mclp_tourism_test_50.pkl'
        if not os.path.exists(dataset_file):
            print("生成新的测试数据集...")
            generator = MCLPDatasetGenerator(
                num_nodes=200,
                num_instances=20,
                device=device,
                include_tourism_features=True,
                tourism_hotspots=8
            )
            dataset = generator.generate_dataset()
            save_dataset(dataset, dataset_file)
        else:
            dataset = load_dataset(dataset_file)
        
        print(f"成功加载数据集，包含 {len(dataset)} 个实例")
        
        # 使用部分数据进行测试
        if len(dataset) > 10:
            dataset = dataset[:5]
            print(f"使用前5个实例进行测试")
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 加载预训练模型
    print("\n加载预训练模型...")
    wrapper = SelfSupervisedMCLPWrapper(device=device)
    
    try:
        # 初始化模型
        test_instance = dataset[0]
        test_graph = build_mclp_graph(test_instance)
        input_dim = test_graph.x.shape[1]
        
        wrapper.initialize_model(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=64
        )
        
        # 加载预训练权重
        model_path = 'moco_mclp_pretrained.pth'
        if os.path.exists(model_path):
            wrapper.model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
            print("预训练模型加载成功")
        else:
            print("警告：未找到预训练模型，使用随机初始化")
            
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    K = 10  # 设施数量
    all_results = []
    
    print(f"\n开始求解MCLP（K={K}）...")
    print("="*70)
    
    for index, instance in enumerate(dataset):
        start_time = time.time()
        
        print(f"\n处理实例 {index}: {instance['name']}")
        print(f"  节点数: {len(instance['points'])}")
        print(f"  覆盖半径: {instance.get('coverage_radius', 'N/A'):.2f}")
        
        try:
            # 1. 构建图
            graph = build_mclp_graph(instance)
            
            # 2. 获取自监督嵌入
            x = graph.x.to(device).float()
            edge_index = graph.edge_index.to(device).long()
            edge_weight = graph.edge_attr.to(device).float() if graph.edge_attr is not None else None
            dist_feat = graph.dist_row_sum.to(device).float() if hasattr(graph, 'dist_row_sum') else torch.ones(x.shape[0], 1, device=device)
            degree_feat = graph.degree.to(device).float() if hasattr(graph, 'degree') else torch.ones(x.shape[0], 1, device=device)
            
            # 获取嵌入
            wrapper.model.eval()
            with torch.no_grad():
                embeddings, _, _ = wrapper.model(
                    0, x, edge_index, edge_weight, dist_feat, degree_feat, x.shape[0]
                )
            
            # 3. 训练优化器
            print("  训练优化器...")
            optimizer_model = MCLPOptimizer(embedding_dim=embeddings.shape[1]).to(device)
            train_optimizer(embeddings, K, optimizer_model, num_iterations=500, lr=0.01)
            
            # 4. 求解
            print("  求解MCLP...")
            selected_indices, _ = optimizer_model.solve(embeddings, K, num_samples=20)
            
            # 5. 计算实际覆盖率
            if hasattr(graph, 'distance_matrix') and hasattr(graph, 'coverage_radius'):
                dist_matrix = graph.distance_matrix
                coverage_radius = graph.coverage_radius
                
                if len(selected_indices) > 0:
                    dist_to_selected = dist_matrix[:, selected_indices]
                    min_dist = torch.min(dist_to_selected, dim=1)[0]
                    covered_mask = (min_dist <= coverage_radius)
                    
                    # 如果有权重，计算加权覆盖率
                    if hasattr(graph, 'total_weights') and graph.total_weights is not None:
                        coverage = torch.sum(graph.total_weights[covered_mask]).item()
                        total_demand = torch.sum(graph.total_weights).item()
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
                    'coverage_radius': coverage_radius,
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
        print("文旅MCLP求解完成!")
        print("="*70)
        
        # 计算统计
        coverages = [r['coverage_percentage'] for r in all_results]
        times = [r['time'] for r in all_results]
        
        avg_coverage = np.mean(coverages)
        avg_time = np.mean(times)
        std_coverage = np.std(coverages)
        
        print(f"平均覆盖率: {avg_coverage:.1f}% ± {std_coverage:.1f}%")
        print(f"平均耗时: {avg_time:.2f}秒")
        print(f"处理实例数: {len(all_results)}")
        print(f"最佳覆盖率: {max(coverages):.1f}%")
        print(f"最差覆盖率: {min(coverages):.1f}%")
        
        # 保存结果
        results_summary = {
            'all_results': all_results,
            'summary': {
                'avg_coverage_percentage': avg_coverage,
                'std_coverage_percentage': std_coverage,
                'avg_time': avg_time,
                'num_instances': len(all_results),
                'K': K
            }
        }
        
        result_file = 'mclp_tourism_results.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(results_summary, f)
        
        # 可视化统计
        visualize_statistics(all_results)
        
        print(f"\n结果已保存到: {result_file}")

def visualize_result(instance, selected_indices, result):
    """可视化单个结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    points = instance['points'].cpu().numpy()
    selected_points = points[selected_indices]
    
    # 1. 点分布和设施选择
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=20, alpha=0.6, c='blue', label='候选点')
    ax1.scatter(selected_points[:, 0], selected_points[:, 1], 
                s=150, c='red', marker='X', label=f'选定设施(K={result["K"]})', 
                edgecolors='black', linewidth=2, zorder=10)
    
    # 绘制服务半径
    coverage_radius = result.get('coverage_radius', np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.15)
    for idx in selected_indices:
        circle = plt.Circle(points[idx], coverage_radius, 
                           color='red', alpha=0.08, fill=True, zorder=1)
        ax1.add_patch(circle)
    
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title(f'设施选址 - {instance["name"]}')
    ax1.legend(loc='upper right')
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
                   s=30, c='green', alpha=0.7, label=f'已覆盖点 ({len(covered_points)})')
        ax2.scatter(uncovered_points[:, 0], uncovered_points[:, 1], 
                   s=30, c='gray', alpha=0.4, label=f'未覆盖点 ({len(uncovered_points)})')
        ax2.scatter(selected_points[:, 0], selected_points[:, 1], 
                   s=150, c='red', marker='X', label='设施点', edgecolors='black', linewidth=2)
        
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        ax2.set_title('覆盖分析')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "无距离矩阵数据", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
    
    # 3. 统计信息
    ax3 = axes[2]
    ax3.axis('off')
    
    stats_text = f"""
    实例信息:
    ----------
    节点数: {result['n_nodes']}
    设施数: {result['K']}
    覆盖半径: {coverage_radius:.2f}
    
    覆盖结果:
    ----------
    覆盖点数: {int(result['coverage'])}
    总点数: {int(result['total_demand'])}
    覆盖率: {result['coverage_percentage']:.1f}%
    
    性能:
    ----------
    求解时间: {result['time']:.2f}秒
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'mclp_tourism_result_{instance["name"]}.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_statistics(all_results):
    """可视化统计结果"""
    if len(all_results) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 覆盖率分布
    ax1 = axes[0]
    coverages = [r['coverage_percentage'] for r in all_results]
    instances = [r['instance_name'] for r in all_results]
    
    x_pos = np.arange(len(coverages))
    bars = ax1.bar(x_pos, coverages, color='skyblue', edgecolor='black')
    
    # 标注数值
    for i, (bar, cov) in enumerate(zip(bars, coverages)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{cov:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('实例')
    ax1.set_ylabel('覆盖率 (%)')
    ax1.set_title('各实例覆盖率对比')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.split('_')[-1] for name in instances], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # 2. 覆盖率vs节点数散点图
    ax2 = axes[1]
    nodes = [r['n_nodes'] for r in all_results]
    
    ax2.scatter(nodes, coverages, s=100, c='orange', edgecolors='black', alpha=0.7)
    
    # 添加趋势线
    if len(nodes) > 1:
        z = np.polyfit(nodes, coverages, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(nodes), max(nodes), 100)
        ax2.plot(x_range, p(x_range), 'r--', alpha=0.7, label='趋势线')
    
    ax2.set_xlabel('节点数')
    ax2.set_ylabel('覆盖率 (%)')
    ax2.set_title('覆盖率与节点数的关系')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mclp_tourism_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("文旅场景MCLP求解系统（基于自监督对比学习）")
    print("="*80)
    
    main()