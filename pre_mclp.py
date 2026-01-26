# pre_mclp.py (修复版)
import numpy as np
import torch
import pickle
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_more import build_mclp_graph, save_dataset, MCLPDatasetGenerator
from self_supervised_mclp import StableSelfSupervisedMCLPWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def create_simple_dataset(num_instances=20, num_nodes=50, save_path='simple_mclp_dataset.pkl'):
    """创建简单的测试数据集（包含所有必要字段）"""
    print("创建简单数据集...")
    
    dataset = []
    for i in range(num_instances):
        # 随机生成点
        points = torch.randn(num_nodes, 2)
        
        # 计算距离矩阵
        from create_more import _pairwise_euclidean
        distance_matrix = _pairwise_euclidean(points, points, torch.device('cpu'))
        
        # 随机生成权重
        weights = torch.rand(num_nodes)
        
        # 固定的覆盖半径
        coverage_radius = 0.4
        
        # 简单的文旅特征（可选）
        tourism_features = torch.randn(num_nodes, 3)
        
        instance = {
            'name': f'simple_instance_{i}',
            'instance_id': i,
            'points': points,
            'total_weights': weights,
            'coverage_radius': coverage_radius,
            'tourism_features': tourism_features,
            'distance_matrix': distance_matrix,  # 关键：包含距离矩阵
            'num_nodes': num_nodes
        }
        
        dataset.append(instance)
    
    # 保存数据集
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"已创建 {len(dataset)} 个简单实例，保存到 {save_path}")
    return dataset

def plot_training_loss(losses, title="训练损失"):
    """绘制训练损失"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='训练损失', alpha=0.7)
    plt.xlabel('训练步数')
    plt.ylabel('损失')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加平滑曲线
    if len(losses) > 10:
        window_size = max(5, len(losses) // 20)
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(losses)), smoothed, label='平滑曲线', linewidth=2)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.close()
    print("训练损失图已保存: training_loss.png")

def main():
    print("稳定的MCLP自监督预训练")
    print("=" * 60)
    
    # =====================================================
    # 1. 准备数据
    # =====================================================
    print("\n1. 准备数据...")
    
    dataset_file = 'simple_mclp_dataset.pkl'
    
    if os.path.exists(dataset_file):
        print(f"加载现有数据集: {dataset_file}")
        try:
            dataset = load_dataset(dataset_file)
        except:
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
    else:
        print("创建新的数据集...")
        dataset = create_simple_dataset(num_instances=20, num_nodes=50)
    
    print(f"数据集包含 {len(dataset)} 个实例")
    
    if len(dataset) == 0:
        print("错误: 数据集为空")
        return
    
    # 检查第一个实例是否有必要字段
    sample_instance = dataset[0]
    required_fields = ['points', 'total_weights', 'coverage_radius']
    for field in required_fields:
        if field not in sample_instance:
            print(f"警告: 实例缺少字段 '{field}'")
    
    # =====================================================
    # 2. 初始化模型
    # =====================================================
    print("\n2. 初始化模型...")
    
    # 获取输入维度
    try:
        test_graph = build_mclp_graph(sample_instance, device=device)
        input_dim = test_graph.x.shape[1]
        print(f"图构建成功，输入维度: {input_dim}")
    except Exception as e:
        print(f"构建测试图失败: {e}")
        # 估计输入维度
        input_dim = 2  # 坐标
        if 'tourism_features' in sample_instance:
            input_dim += sample_instance['tourism_features'].shape[1]
        if 'total_weights' in sample_instance:
            input_dim += 1
        
        print(f"使用估计的输入维度: {input_dim}")
    
    # 初始化包装器
    wrapper = StableSelfSupervisedMCLPWrapper(device=device)
    wrapper.initialize_model(input_dim=input_dim)
    
    # =====================================================
    # 3. 训练模型
    # =====================================================
    print("\n3. 开始训练...")
    print("=" * 40)
    
    all_losses = []
    start_time = time.time()
    
    # 使用前10个实例训练
    train_instances = dataset[:10]
    
    for i, instance in enumerate(tqdm(train_instances, desc="训练实例")):
        print(f"\n训练实例 {i+1}/{len(train_instances)}: {instance['name']}")
        
        try:
            # 构建图
            graph = build_mclp_graph(instance, device=device)
            print(f"  图构建成功: {graph.num_nodes}个节点, {graph.edge_index.shape[1]}条边")
            
            # 训练
            losses = wrapper.train_on_instance(graph, epochs=30, K=10)
            all_losses.extend(losses)
            
            avg_loss = np.mean(losses)
            print(f"  平均损失: {avg_loss:.4f}")
            
            # 每训练2个实例保存一次检查点
            if (i + 1) % 2 == 0:
                wrapper.save_model(f'checkpoint_{i+1}.pth')
                
        except Exception as e:
            print(f"  训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =====================================================
    # 4. 保存最终模型
    # =====================================================
    print("\n4. 保存最终模型...")
    wrapper.save_model('stable_mclp_model.pth')
    
    # =====================================================
    # 5. 测试模型
    # =====================================================
    print("\n5. 测试模型性能...")
    print("=" * 40)
    
    test_results = []
    test_instances = dataset[-5:] if len(dataset) > 5 else dataset
    
    K_values = [5, 8, 10]
    
    for K in K_values:
        print(f"\n测试 K={K}:")
        K_coverages = []
        
        for i, instance in enumerate(test_instances):
            try:
                graph = build_mclp_graph(instance, device=device)
                
                # 求解
                selected_indices, coverage, scores = wrapper.solve_mclp(graph, K)
                
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
                    'selected_indices': selected_indices.tolist(),
                    'n_facilities': len(selected_indices)
                }
                
                test_results.append(result)
                K_coverages.append(coverage_pct)
                
                print(f"  实例 {i}: 覆盖率 = {coverage_pct:.1f}%")
                
            except Exception as e:
                print(f"  测试失败: {e}")
                continue
        
        # 打印统计
        if K_coverages:
            avg_coverage = np.mean(K_coverages)
            std_coverage = np.std(K_coverages)
            print(f"  K={K}: 平均覆盖率 = {avg_coverage:.1f}% ± {std_coverage:.1f}%")
    
    # =====================================================
    # 6. 保存结果和可视化
    # =====================================================
    print("\n6. 保存结果...")
    
    results_summary = {
        'all_losses': all_losses,
        'test_results': test_results,
        'training_time': time.time() - start_time,
        'num_train_instances': len(train_instances),
        'num_test_instances': len(test_instances)
    }
    
    with open('stable_mclp_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"结果已保存: stable_mclp_results.pkl")
    print(f"训练时间: {results_summary['training_time']:.1f}秒")
    
    # 可视化
    if all_losses:
        plot_training_loss(all_losses)
    
    if test_results:
        # 绘制测试结果
        plt.figure(figsize=(10, 6))
        
        # 按K值分组
        K_values = sorted(set(r['K'] for r in test_results))
        
        for K in K_values:
            coverages = [r['coverage_percentage'] for r in test_results if r['K'] == K]
            if coverages:
                plt.scatter([K] * len(coverages), coverages, label=f'K={K}', alpha=0.6, s=50)
                
                # 添加平均值
                avg = np.mean(coverages)
                plt.axhline(y=avg, color='red' if K == 10 else 'blue', alpha=0.3, linestyle='--')
                plt.text(K, avg, f'{avg:.1f}%', ha='center', va='bottom')
        
        plt.xlabel('K值')
        plt.ylabel('覆盖率 (%)')
        plt.title('MCLP求解性能测试')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('test_results.png', dpi=150)
        plt.close()
        print("测试结果图已保存: test_results.png")
    
    print("\n" + "=" * 60)
    print("预训练完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()