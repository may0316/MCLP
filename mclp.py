# mclp.py (稳定版)
import time
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from create_more import build_mclp_graph, load_dataset
from self_supervised_mclp import StableSelfSupervisedMCLPWrapper

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def visualize_solution(instance, selected_indices, coverage_pct, K, save_path=None):
    """可视化解决方案"""
    points = instance['points']
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    
    selected_points = points[selected_indices]
    
    plt.figure(figsize=(8, 8))
    
    # 所有点
    plt.scatter(points[:, 0], points[:, 1], s=20, alpha=0.5, c='gray', label='需求点')
    
    # 选中的设施
    plt.scatter(selected_points[:, 0], selected_points[:, 1],
               s=200, c='red', marker='X', label=f'选定设施 (K={K})')
    
    # 覆盖范围
    r = instance['coverage_radius']
    for idx in selected_indices:
        circle = plt.Circle(points[idx], r, color='red', alpha=0.1, linewidth=0)
        plt.gca().add_patch(circle)
    
    # 标记覆盖的点
    if isinstance(points, np.ndarray):
        points_tensor = torch.from_numpy(points)
    else:
        points_tensor = points
    
    dist_matrix = torch.cdist(points_tensor, points_tensor)
    covered_mask = torch.any(
        dist_matrix[:, selected_indices] <= r,
        dim=1
    ).cpu().numpy()
    
    covered_points = points[covered_mask]
    if len(covered_points) > 0:
        plt.scatter(covered_points[:, 0], covered_points[:, 1],
                   s=40, alpha=0.7, c='green', label='覆盖点')
    
    plt.title(f"{instance.get('name', '实例')} | K={K} | 覆盖率: {coverage_pct:.1f}%")
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    print("\n" + "=" * 80)
    print("文旅MCLP求解（稳定版）")
    print("=" * 80)
    
    # --------------------------------------------------------
    # 1. 加载数据
    # --------------------------------------------------------
    print("\n1. 加载数据...")
    
    dataset_files = ['simple_mclp_dataset.pkl', 'mclp_tourism_train_improved.pkl']
    dataset = None
    
    for file in dataset_files:
        if os.path.exists(file):
            print(f"加载数据集: {file}")
            try:
                dataset = load_dataset(file)
                break
            except:
                with open(file, 'rb') as f:
                    dataset = pickle.load(f)
                break
    
    if dataset is None:
        print("未找到数据集，使用默认测试数据...")
        # 创建简单测试数据
        dataset = []
        for i in range(3):
            points = torch.randn(50, 2)
            dataset.append({
                'name': f'test_{i}',
                'points': points,
                'weights': torch.ones(50),
                'coverage_radius': 0.4
            })
    
    print(f"加载了 {len(dataset)} 个实例")
    
    # 使用前3个实例测试
    test_instances = dataset[:3]
    
    # --------------------------------------------------------
    # 2. 初始化模型
    # --------------------------------------------------------
    print("\n2. 初始化模型...")
    
    wrapper = StableSelfSupervisedMCLPWrapper(device=device)
    
    # 获取输入维度
    test_instance = test_instances[0]
    try:
        test_graph = build_mclp_graph(test_instance)
        input_dim = test_graph.x.shape[1]
    except:
        # 简单估计
        input_dim = test_instance['points'].shape[1]
        if 'tourism_features' in test_instance:
            input_dim += test_instance['tourism_features'].shape[1]
    
    wrapper.initialize_model(input_dim=input_dim)
    
    # 尝试加载预训练模型
    model_files = ['stable_mclp_model.pth', 'moco_mclp_pretrained.pth']
    model_loaded = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"加载预训练模型: {model_file}")
            wrapper.load_model(model_file)
            model_loaded = True
            break
    
    if not model_loaded:
        print("未找到预训练模型，进行快速训练...")
        # 快速训练
        print("快速训练中...")
        for i, instance in enumerate(test_instances[:2]):
            try:
                graph = build_mclp_graph(instance)
                wrapper.train_on_instance(graph, epochs=20, K=10)
            except:
                continue
    
    # --------------------------------------------------------
    # 3. 求解MCLP
    # --------------------------------------------------------
    print("\n3. 求解MCLP...")
    print("=" * 50)
    
    results = []
    K = 10  # 默认K值
    
    for idx, instance in enumerate(test_instances):
        print(f"\n实例 {idx}: {instance.get('name', '未知')}")
        
        try:
            # 构建图
            graph = build_mclp_graph(instance)
            
            start_time = time.time()
            
            # 求解
            selected_indices, coverage, scores = wrapper.solve_mclp(graph, K)
            
            # 计算覆盖率
            if hasattr(graph, 'total_weights') and graph.total_weights is not None:
                total_demand = torch.sum(graph.total_weights).item()
            else:
                total_demand = graph.num_nodes
            
            coverage_pct = (coverage / total_demand) * 100 if total_demand > 0 else 0
            
            # 记录结果
            result = {
                'instance_id': idx,
                'instance_name': instance.get('name', f'instance_{idx}'),
                'K': K,
                'selected_indices': selected_indices.tolist(),
                'coverage': coverage,
                'total_demand': total_demand,
                'coverage_percentage': coverage_pct,
                'time': time.time() - start_time
            }
            
            results.append(result)
            
            print(f"  覆盖率: {coverage_pct:.1f}%")
            print(f"  选定设施数: {len(selected_indices)}")
            print(f"  用时: {result['time']:.3f}s")
            
            # 可视化
            visualize_solution(
                instance, 
                selected_indices, 
                coverage_pct, 
                K,
                save_path=f'solution_instance_{idx}_K{K}.png'
            )
            
        except Exception as e:
            print(f"  求解失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --------------------------------------------------------
    # 4. 结果汇总
    # --------------------------------------------------------
    if results:
        print("\n" + "=" * 50)
        print("结果汇总:")
        print("=" * 50)
        
        coverages = [r['coverage_percentage'] for r in results]
        times = [r['time'] for r in results]
        
        print(f"成功求解实例数: {len(results)}/{len(test_instances)}")
        print(f"平均覆盖率: {np.mean(coverages):.1f}%")
        print(f"覆盖率范围: {min(coverages):.1f}% - {max(coverages):.1f}%")
        print(f"平均用时: {np.mean(times):.3f}s")
        
        # 保存结果
        with open('mclp_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存: mclp_results.pkl")
        
        # 绘制汇总图
        plt.figure(figsize=(10, 6))
        
        # 条形图
        x_pos = np.arange(len(results))
        bars = plt.bar(x_pos, coverages, alpha=0.7)
        
        # 标注
        for i, (bar, cov) in enumerate(zip(bars, coverages)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{cov:.1f}%', ha='center', va='bottom')
            
            # 实例名称
            instance_name = results[i]['instance_name']
            plt.text(bar.get_x() + bar.get_width()/2., -5, 
                    instance_name[:10], ha='center', va='top', rotation=45)
        
        plt.axhline(y=np.mean(coverages), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(coverages):.1f}%')
        
        plt.ylabel('覆盖率 (%)')
        plt.title(f'MCLP求解结果汇总 (K={K})')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig('results_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("结果汇总图已保存: results_summary.png")
        
    else:
        print("\n没有成功求解的实例")
    
    print("\n" + "=" * 80)
    print("MCLP求解完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()