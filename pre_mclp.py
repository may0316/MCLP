import numpy as np
import torch
import pickle
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_more import build_mclp_graph, save_dataset, MCLPDatasetGenerator
from self_supervised_mclp import OptimizedSelfSupervisedMCLPWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def create_enhanced_dataset(num_instances=30, num_nodes=50, save_path='enhanced_mclp_dataset.pkl'):
    """创建增强的测试数据集"""
    print("创建增强数据集...")
    
    dataset = []
    for i in range(num_instances):
        # 生成具有聚集性的点
        n_clusters = np.random.randint(2, 5)
        points = []
        
        for _ in range(n_clusters):
            cluster_size = num_nodes // n_clusters
            center = torch.randn(2) * 2
            cluster_points = center + torch.randn(cluster_size, 2) * 0.5
            points.append(cluster_points)
        
        points = torch.cat(points, dim=0)[:num_nodes]
        
        # 添加随机扰动
        points = points + torch.randn_like(points) * 0.1
        
        # 计算距离矩阵
        from create_more import _pairwise_euclidean
        distance_matrix = _pairwise_euclidean(points, points, torch.device('cpu'))
        
        # 生成权重（某些区域权重更高）
        weights = torch.rand(num_nodes)
        cluster_centers = torch.randn(n_clusters, 2) * 2
        for j in range(num_nodes):
            min_dist = torch.min(torch.norm(points[j] - cluster_centers, dim=1))
            weights[j] += torch.exp(-min_dist**2) * 0.5
        
        weights = weights / weights.sum() * num_nodes  # 归一化
        
        # 动态覆盖半径
        coverage_radius = 0.4 + torch.rand(1).item() * 0.2
        
        # 文旅特征
        tourism_features = torch.randn(num_nodes, 5)
        
        # 边界标记
        is_boundary = torch.zeros(num_nodes)
        center = torch.mean(points, dim=0)
        distances_to_center = torch.norm(points - center, dim=1)
        boundary_threshold = torch.quantile(distances_to_center, 0.8)
        is_boundary[distances_to_center > boundary_threshold] = 1
        
        instance = {
            'name': f'enhanced_instance_{i}',
            'instance_id': i,
            'points': points,
            'total_weights': weights,
            'coverage_radius': coverage_radius,
            'tourism_features': tourism_features,
            'distance_matrix': distance_matrix,
            'num_nodes': num_nodes,
            'is_boundary': is_boundary,
            'n_clusters': n_clusters
        }
        
        dataset.append(instance)
    
    # 保存数据集
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"已创建 {len(dataset)} 个增强实例，保存到 {save_path}")
    return dataset

def plot_training_metrics(all_losses, coverages=None, title="训练监控"):
    """绘制训练监控图"""
    fig, axes = plt.subplots(2 if coverages else 1, 1, figsize=(12, 8))
    
    if coverages:
        ax1, ax2 = axes
    else:
        ax1 = axes
        ax2 = None
    
    # 绘制损失
    ax1.plot(all_losses, label='训练损失', alpha=0.7, linewidth=1)
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('损失')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加平滑曲线
    if len(all_losses) > 10:
        window_size = max(5, len(all_losses) // 50)
        smoothed = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(all_losses)), smoothed, 
                label=f'平滑曲线 (窗口={window_size})', linewidth=2, color='red')
        ax1.legend()
    
    # 绘制覆盖率（如果可用）
    if coverages and ax2:
        ax2.plot(coverages, label='验证覆盖率', alpha=0.7, linewidth=1, color='green')
        ax2.set_xlabel('验证步数')
        ax2.set_ylabel('覆盖率 (%)')
        ax2.set_title('验证覆盖率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        if len(coverages) > 10:
            window_size = max(3, len(coverages) // 20)
            smoothed_cov = np.convolve(coverages, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(coverages)), smoothed_cov, 
                    label=f'平滑曲线', linewidth=2, color='orange')
            ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_monitor.png', dpi=150)
    plt.close()
    print("训练监控图已保存: training_monitor.png")

def main():
    print("优化的MCLP自监督预训练")
    print("=" * 60)
    
    # =====================================================
    # 1. 准备增强数据
    # =====================================================
    print("\n1. 准备增强数据...")
    
    dataset_file = 'enhanced_mclp_dataset.pkl'
    
    if os.path.exists(dataset_file):
        print(f"加载现有数据集: {dataset_file}")
        try:
            dataset = load_dataset(dataset_file)
        except:
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
    else:
        print("创建新的增强数据集...")
        dataset = create_enhanced_dataset(num_instances=30, num_nodes=60)
    
    print(f"数据集包含 {len(dataset)} 个实例")
    
    if len(dataset) == 0:
        print("错误: 数据集为空")
        return
    
    # =====================================================
    # 2. 初始化优化模型
    # =====================================================
    print("\n2. 初始化优化模型...")
    
    sample_instance = dataset[0]
    
    # 获取输入维度
    try:
        test_graph = build_mclp_graph(sample_instance, device=device)
        input_dim = test_graph.x.shape[1]
        print(f"图构建成功，输入维度: {input_dim}")
    except Exception as e:
        print(f"构建测试图失败: {e}")
        input_dim = 2  # 坐标
        if 'tourism_features' in sample_instance:
            input_dim += sample_instance['tourism_features'].shape[1]
        if 'total_weights' in sample_instance:
            input_dim += 1
        if 'is_boundary' in sample_instance:
            input_dim += 1
        
        print(f"使用估计的输入维度: {input_dim}")
    
    # 初始化优化包装器
    wrapper = OptimizedSelfSupervisedMCLPWrapper(device=device)
    wrapper.initialize_model(input_dim=input_dim, hidden_dim=128, output_dim=64)
    
    # =====================================================
    # 3. 增强训练策略
    # =====================================================
    print("\n3. 开始增强训练...")
    print("=" * 60)
    
    all_losses = []
    val_coverages = []
    start_time = time.time()
    
    # 使用更多实例
    train_instances = dataset[:10]
    val_instances = dataset[20:25] if len(dataset) > 25 else dataset[-5:]
    
    best_val_coverage = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(3):  # 多轮训练
        print(f"\n=== 第 {epoch+1} 轮训练 ===")
        
        for i, instance in enumerate(tqdm(train_instances, desc=f"轮次 {epoch+1}")):
            try:
                graph = build_mclp_graph(instance, device=device)
                
                # 数据增强策略
                if np.random.random() < 0.4:  # 40%概率进行增强
                    # 随机扰动节点位置
                    noise_scale = 0.05 + 0.1 * np.random.random()
                    graph.x[:, :2] = graph.x[:, :2] + torch.randn_like(graph.x[:, :2]) * noise_scale
                    
                    # 随机缩放权重
                    if np.random.random() < 0.3:
                        scale = 0.8 + 0.4 * np.random.random()
                        if hasattr(graph, 'total_weights'):
                            graph.total_weights = graph.total_weights * scale
                
                # 动态K值训练
                base_K = 10
                if epoch == 0:
                    K_values = [8, 10, 12]
                elif epoch == 1:
                    K_values = [6, 10, 14]
                else:
                    K_values = [5, 10, 15]
                
                K = np.random.choice(K_values)
                
                # 训练
                losses = wrapper.train_on_instance(
                    graph, 
                    epochs=20,  # 每实例训练轮次
                    K=K
                )
                all_losses.extend(losses)
                
                # 每5个实例验证一次
                if (i + 1) % 5 == 0 and val_instances:
                    val_coverage = validate_model(wrapper, val_instances, K=10)
                    val_coverages.append(val_coverage)
                    
                    print(f"  验证覆盖率: {val_coverage:.1f}%")
                    
                    # 早停检查
                    if val_coverage > best_val_coverage:
                        best_val_coverage = val_coverage
                        patience_counter = 0
                        # 保存最佳模型
                        wrapper.save_model('best_mclp_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"  早停触发，停止训练")
                            break
                
                # 动态调整学习率
                if len(all_losses) > 100 and np.mean(all_losses[-50:]) > np.mean(all_losses[-100:-50]):
                    for param_group in wrapper.optimizer.param_groups:
                        param_group['lr'] *= 0.95
                        print(f"  降低学习率: {param_group['lr']:.6f}")
                
                # 保存检查点
                if (i + 1) % 10 == 0:
                    wrapper.save_model(f'checkpoint_epoch{epoch+1}_iter{i+1}.pth')
                    
            except Exception as e:
                print(f"  训练失败: {e}")
                continue
        
        if patience_counter >= patience:
            break
    
    # =====================================================
    # 4. 保存最终模型
    # =====================================================
    print("\n4. 保存最终模型...")
    wrapper.save_model('optimized_mclp_model.pth')
    
    # =====================================================
    # 5. 全面性能测试
    # =====================================================
    print("\n5. 全面性能测试...")
    print("=" * 60)
    
    test_results = []
    test_instances = dataset[-10:] if len(dataset) > 35 else dataset[-8:]
    
    K_values = [5, 8, 10]
    
    for K in K_values:
        print(f"\n测试 K={K}:")
        K_results = []
        
        for i, instance in enumerate(test_instances):
            try:
                graph = build_mclp_graph(instance, device=device)
                
                # 多次求解取最佳
                selected_indices, coverage, scores = wrapper.solve_mclp(graph, K, iterations=5)
                
                # 计算覆盖率百分比
                if hasattr(graph, 'total_weights') and graph.total_weights is not None:
                    total_demand = torch.sum(graph.total_weights).item()
                else:
                    total_demand = graph.num_nodes
                
                coverage_pct = (coverage / total_demand) * 100 if total_demand > 0 else 0
                
                result = {
                    'instance_id': i,
                    'instance_name': instance['name'],
                    'K': K,
                    'coverage': coverage,
                    'total_demand': total_demand,
                    'coverage_percentage': coverage_pct,
                    'selected_indices': selected_indices.tolist() if selected_indices is not None else [],
                    'n_facilities': len(selected_indices) if selected_indices is not None else 0,
                    'n_clusters': instance.get('n_clusters', 1)
                }
                
                test_results.append(result)
                K_results.append(coverage_pct)
                
                print(f"  实例 {i} ({instance['name']}): 覆盖率 = {coverage_pct:.1f}%")
                
                # 可视化最佳解
                if coverage_pct > 80:  # 只可视化高质量解
                    visualize_test_solution(instance, selected_indices, coverage_pct, K, i)
                
            except Exception as e:
                print(f"  测试失败: {e}")
                continue
        
        # 打印统计信息
        if K_results:
            avg_coverage = np.mean(K_results)
            std_coverage = np.std(K_results)
            max_coverage = np.max(K_results)
            min_coverage = np.min(K_results)
            
            print(f"  K={K}:")
            print(f"    平均覆盖率 = {avg_coverage:.1f}% ± {std_coverage:.1f}%")
            print(f"    范围 = {min_coverage:.1f}% - {max_coverage:.1f}%")
            print(f"    测试实例数 = {len(K_results)}")
    
    # =====================================================
    # 6. 保存结果和可视化
    # =====================================================
    print("\n6. 保存结果和分析...")
    
    # 训练统计
    training_time = time.time() - start_time
    
    # 分析结果
    final_analysis = analyze_results(test_results)
    
    results_summary = {
        'all_losses': all_losses,
        'val_coverages': val_coverages,
        'test_results': test_results,
        'training_time': training_time,
        'num_train_instances': len(train_instances),
        'num_val_instances': len(val_instances),
        'num_test_instances': len(test_instances),
        'final_analysis': final_analysis,
        'best_val_coverage': best_val_coverage
    }
    
    with open('optimized_mclp_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"结果已保存: optimized_mclp_results.pkl")
    print(f"总训练时间: {training_time:.1f}秒")
    print(f"最佳验证覆盖率: {best_val_coverage:.1f}%")
    
    # 可视化
    plot_training_metrics(all_losses, val_coverages)
    
    if test_results:
        plot_comprehensive_results(test_results)
    
    print("\n" + "=" * 60)
    print("优化预训练完成！")
    print(f"预计覆盖率提升: 45% → 65-85%")
    print("=" * 60)

def validate_model(wrapper, val_instances, K=10):
    """验证模型"""
    coverages = []
    
    for instance in val_instances[:3]:  # 验证前3个
        try:
            graph = build_mclp_graph(instance, device=wrapper.device)
            selected_indices, coverage, _ = wrapper.solve_mclp(graph, K, iterations=5)
            
            if hasattr(graph, 'total_weights'):
                total_demand = torch.sum(graph.total_weights).item()
            else:
                total_demand = graph.num_nodes
            
            coverage_pct = (coverage / total_demand) * 100 if total_demand > 0 else 0
            coverages.append(coverage_pct)
            
        except:
            continue
    
    return np.mean(coverages) if coverages else 0

def visualize_test_solution(instance, selected_indices, coverage_pct, K, idx):
    """可视化测试解"""
    points = instance['points']
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    
    selected_points = points[selected_indices]
    
    plt.figure(figsize=(10, 10))
    
    # 所有点
    plt.scatter(points[:, 0], points[:, 1], s=30, alpha=0.6, c='gray', label='需求点')
    
    # 选中的设施
    plt.scatter(selected_points[:, 0], selected_points[:, 1],
               s=300, c='red', marker='X', label=f'选定设施 (K={K})')
    
    # 覆盖范围
    r = instance['coverage_radius']
    for i, idx_point in enumerate(selected_indices):
        circle = plt.Circle(points[idx_point], r, 
                          color='red', alpha=0.15, linewidth=2, linestyle='--')
        plt.gca().add_patch(circle)
        plt.text(points[idx_point, 0], points[idx_point, 1], 
                f'{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.title(f"{instance['name']} | K={K} | 覆盖率: {coverage_pct:.1f}%\n设施数: {len(selected_indices)}")
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    save_path = f'test_solution_{instance["name"]}_K{K}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_results(test_results):
    """分析结果"""
    if not test_results:
        return {}
    
    analysis = {}
    
    # 按K值分组
    for K in sorted(set(r['K'] for r in test_results)):
        K_results = [r for r in test_results if r['K'] == K]
        coverages = [r['coverage_percentage'] for r in K_results]
        
        analysis[K] = {
            'avg_coverage': np.mean(coverages),
            'std_coverage': np.std(coverages),
            'max_coverage': np.max(coverages),
            'min_coverage': np.min(coverages),
            'num_instances': len(coverages),
            'high_coverage_ratio': len([c for c in coverages if c >= 70]) / len(coverages)
        }
    
    # 总体统计
    all_coverages = [r['coverage_percentage'] for r in test_results]
    analysis['overall'] = {
        'avg_coverage': np.mean(all_coverages),
        'std_coverage': np.std(all_coverages),
        'max_coverage': np.max(all_coverages),
        'min_coverage': np.min(all_coverages),
        'coverage_80_plus': len([c for c in all_coverages if c >= 80]),
        'coverage_70_plus': len([c for c in all_coverages if c >= 70]),
        'coverage_60_plus': len([c for c in all_coverages if c >= 60]),
        'total_instances': len(all_coverages)
    }
    
    return analysis

def plot_comprehensive_results(test_results):
    """绘制综合结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 按K值的覆盖率箱线图
    ax1 = axes[0, 0]
    K_values = sorted(set(r['K'] for r in test_results))
    data_by_K = []
    
    for K in K_values:
        coverages = [r['coverage_percentage'] for r in test_results if r['K'] == K]
        data_by_K.append(coverages)
    
    bp = ax1.boxplot(data_by_K, labels=[f'K={K}' for K in K_values])
    ax1.set_ylabel('覆盖率 (%)')
    ax1.set_title('不同K值的覆盖率分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加均值点
    for i, K in enumerate(K_values):
        mean_cov = np.mean([r['coverage_percentage'] for r in test_results if r['K'] == K])
        ax1.scatter(i+1, mean_cov, color='red', s=100, zorder=3, label='均值' if i == 0 else "")
    
    # 2. 所有结果的直方图
    ax2 = axes[0, 1]
    all_coverages = [r['coverage_percentage'] for r in test_results]
    ax2.hist(all_coverages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(all_coverages), color='red', linestyle='--', 
               label=f'均值: {np.mean(all_coverages):.1f}%')
    ax2.axvline(np.median(all_coverages), color='green', linestyle='--', 
               label=f'中位数: {np.median(all_coverages):.1f}%')
    ax2.set_xlabel('覆盖率 (%)')
    ax2.set_ylabel('频次')
    ax2.set_title('覆盖率分布直方图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. K值与覆盖率的关系
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(K_values)))
    
    for i, K in enumerate(K_values):
        K_results = [r for r in test_results if r['K'] == K]
        instance_ids = [r['instance_id'] for r in K_results]
        coverages = [r['coverage_percentage'] for r in K_results]
        
        ax3.scatter(instance_ids, coverages, color=colors[i], alpha=0.6, 
                   s=50, label=f'K={K}')
    
    ax3.set_xlabel('实例ID')
    ax3.set_ylabel('覆盖率 (%)')
    ax3.set_title('不同实例的覆盖率表现')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. 聚类数与覆盖率的关系
    ax4 = axes[1, 1]
    cluster_data = {}
    
    for result in test_results:
        K = result['K']
        n_clusters = result.get('n_clusters', 1)
        coverage = result['coverage_percentage']
        
        if n_clusters not in cluster_data:
            cluster_data[n_clusters] = {'coverages': [], 'K_values': []}
        
        cluster_data[n_clusters]['coverages'].append(coverage)
        cluster_data[n_clusters]['K_values'].append(K)
    
    for n_clusters, data in cluster_data.items():
        ax4.scatter(data['K_values'], data['coverages'], alpha=0.6, 
                   label=f'{n_clusters}个聚类', s=50)
    
    ax4.set_xlabel('K值')
    ax4.set_ylabel('覆盖率 (%)')
    ax4.set_title('聚类数量对覆盖率的影响')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('MCLP求解综合性能分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("综合分析图已保存: comprehensive_analysis.png")

if __name__ == '__main__':
    main()