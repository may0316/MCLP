import time
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from create_more import build_mclp_graph, load_dataset
from self_supervised_mclp import OptimizedSelfSupervisedMCLPWrapper

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def enhanced_visualize_solution(instance, selected_indices, coverage_pct, K, save_path=None, scores=None):
    """增强的可视化解决方案"""
    points = instance['points']
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    
    if selected_indices is None or len(selected_indices) == 0:
        print("警告: 没有选中的设施")
        return
    
    selected_points = points[selected_indices]
    
    plt.figure(figsize=(12, 10))
    
    # 1. 所有点，按是否被覆盖着色
    if isinstance(points, np.ndarray):
        points_tensor = torch.from_numpy(points)
    else:
        points_tensor = points
    
    # 计算覆盖情况
    r = instance['coverage_radius']
    dist_matrix = torch.cdist(points_tensor, points_tensor)
    covered_mask = torch.any(
        dist_matrix[:, selected_indices] <= r,
        dim=1
    ).cpu().numpy()
    
    # 绘制未覆盖的点
    uncovered_points = points[~covered_mask]
    plt.scatter(uncovered_points[:, 0], uncovered_points[:, 1], 
               s=40, alpha=0.5, c='lightgray', label='未覆盖点')
    
    # 绘制覆盖的点
    covered_points = points[covered_mask]
    plt.scatter(covered_points[:, 0], covered_points[:, 1],
               s=50, alpha=0.8, c='lightgreen', label='覆盖点')
    
    # 2. 选中的设施
    plt.scatter(selected_points[:, 0], selected_points[:, 1],
               s=300, c='red', marker='X', label=f'选定设施 (K={K})', 
               edgecolors='darkred', linewidths=2, zorder=5)
    
    # 3. 覆盖范围圆
    for i, idx in enumerate(selected_indices):
        circle = plt.Circle(points[idx], r, 
                          color='red', alpha=0.1, linewidth=1.5, linestyle='--')
        plt.gca().add_patch(circle)
        
        # 标注设施编号
        plt.text(points[idx, 0], points[idx, 1], 
                f'F{i+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkred')
    
    # 4. 设施分数（如果有）
    if scores is not None and len(scores) == len(points):
        # 归一化分数用于颜色映射
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        cmap = plt.cm.RdYlGn
        
        # 在设施点周围添加分数环
        for i, idx in enumerate(selected_indices):
            color = cmap(norm_scores[idx])
            circle_inner = plt.Circle(points[idx], r * 0.15, 
                                     color=color, alpha=0.7, linewidth=0)
            plt.gca().add_patch(circle_inner)
    
    # 5. 覆盖半径示意图
    legend_circle = plt.Circle((0, 0), r, color='red', alpha=0.2, linewidth=0)
    legend_circle.set_visible(False)
    plt.gca().add_patch(legend_circle)
    
    # 计算统计信息
    n_covered = len(covered_points)
    n_total = len(points)
    coverage_ratio = n_covered / n_total * 100
    
    # 设施间最小距离
    if len(selected_points) > 1:
        from scipy.spatial.distance import pdist
        facility_dists = pdist(selected_points)
        min_facility_dist = np.min(facility_dists) if len(facility_dists) > 0 else 0
        avg_facility_dist = np.mean(facility_dists) if len(facility_dists) > 0 else 0
    else:
        min_facility_dist = 0
        avg_facility_dist = 0
    
    plt.title(f"{instance.get('name', '实例')} | K={K}\n"
              f"覆盖率: {coverage_pct:.1f}% ({n_covered}/{n_total}点)\n"
              f"设施最小间距: {min_facility_dist:.2f} | 平均间距: {avg_facility_dist:.2f}",
              fontsize=14, fontweight='bold')
    
    plt.legend(loc='upper right', fontsize=10)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 添加坐标轴标签
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
        print(f"可视化已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()

def run_hyperparameter_tuning(wrapper, graph, K_values):
    """运行超参数调优"""
    print("\n运行超参数调优...")
    
    best_params = {}
    best_results = {}
    
    # 定义超参数搜索空间
    param_grid = {
        'temperature': [ 0.1],
        'iterations': [5],
        'alpha': [ 0.3]
    }
    
    for temp in param_grid['temperature']:
        for iters in param_grid['iterations']:
            for alpha in param_grid['alpha']:
                print(f"  测试: temp={temp}, iters={iters}, alpha={alpha}")
                
                # 保存原始参数
                original_state = wrapper.model.training
                wrapper.model.eval()
                
                # 临时修改模型参数（需要在模型中实现这些参数）
                if hasattr(wrapper, 'temperature'):
                    wrapper.temperature = temp
                if hasattr(wrapper, 'alpha'):
                    wrapper.alpha = alpha
                
                K_results = {}
                for K in K_values:
                    try:
                        selected, coverage, scores = wrapper.solve_mclp(graph, K, iterations=iters)
                        
                        if hasattr(graph, 'total_weights'):
                            total_demand = torch.sum(graph.total_weights).item()
                        else:
                            total_demand = graph.num_nodes
                        
                        coverage_pct = (coverage / total_demand) * 100
                        K_results[K] = coverage_pct
                    except:
                        K_results[K] = 0
                
                avg_coverage = np.mean(list(K_results.values()))
                
                # 更新最佳结果
                if not best_results or avg_coverage > np.mean(list(best_results.values())):
                    best_results = K_results
                    best_params = {'temperature': temp, 'iterations': iters, 'alpha': alpha}
                    print(f"    新最佳: {avg_coverage:.1f}%")
                
                # 恢复模型状态
                wrapper.model.train(original_state)
    
    print(f"最佳参数: {best_params}")
    print(f"最佳覆盖率: {np.mean(list(best_results.values())):.1f}%")
    
    return best_params, best_results

def main():
    print("\n" + "=" * 80)
    print("文旅MCLP求解（优化完整版）")
    print("=" * 80)
    
    # --------------------------------------------------------
    # 1. 加载增强数据
    # --------------------------------------------------------
    print("\n1. 加载增强数据...")
    
    dataset_files = ['enhanced_mclp_dataset.pkl', 'simple_mclp_dataset.pkl', 'mclp_tourism_train_improved.pkl']
    dataset = None
    
    for file in dataset_files:
        if os.path.exists(file):
            print(f"加载数据集: {file}")
            try:
                dataset = load_dataset(file)
                if dataset is not None:
                    break
            except Exception as e:
                print(f"加载失败: {e}")
                try:
                    with open(file, 'rb') as f:
                        dataset = pickle.load(f)
                    break
                except:
                    continue
    
    if dataset is None:
        print("未找到数据集，创建新的增强数据集...")
        from pre_mclp import create_enhanced_dataset
        dataset = create_enhanced_dataset(num_instances=15, num_nodes=50)
    
    print(f"加载了 {len(dataset)} 个实例")
    
    # 选择实例进行测试
    test_indices = list(range(min(5, len(dataset))))
    test_instances = [dataset[i] for i in test_indices]
    
    # --------------------------------------------------------
    # 2. 初始化优化模型
    # --------------------------------------------------------
    print("\n2. 初始化优化模型...")
    
    wrapper = OptimizedSelfSupervisedMCLPWrapper(device=device)
    
    # 获取输入维度
    test_instance = test_instances[0]
    try:
        test_graph = build_mclp_graph(test_instance)
        input_dim = test_graph.x.shape[1]
        print(f"成功构建图，输入维度: {input_dim}")
    except Exception as e:
        print(f"构建图失败: {e}")
        input_dim = 2  # 坐标
        if 'tourism_features' in test_instance:
            input_dim += test_instance['tourism_features'].shape[1]
        if 'total_weights' in test_instance:
            input_dim += 1
        print(f"使用估计的输入维度: {input_dim}")
    
    wrapper.initialize_model(input_dim=input_dim, hidden_dim=128, output_dim=64)
    
    # 尝试加载预训练模型
    model_files = ['best_mclp_model.pth', 'optimized_mclp_model.pth', 
                   'stable_mclp_model.pth', 'moco_mclp_pretrained.pth']
    model_loaded = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"加载预训练模型: {model_file}")
            try:
                wrapper.load_model(model_file)
                model_loaded = True
                
                # 快速验证模型
                try:
                    test_graph = build_mclp_graph(test_instances[0])
                    selected, coverage, _ = wrapper.solve_mclp(test_graph, K=10, iterations=5)
                    print(f"  模型验证: 覆盖率 = {(coverage/test_graph.num_nodes*100):.1f}%")
                except:
                    print("  模型验证跳过")
                
                break
            except Exception as e:
                print(f"  加载失败: {e}")
                continue
    
    if not model_loaded:
        print("未找到可用的预训练模型，进行快速训练...")
        print("快速训练中...")
        
        # 使用少量实例快速训练
        train_instances = test_instances[:min(3, len(test_instances))]
        for i, instance in enumerate(train_instances):
            try:
                graph = build_mclp_graph(instance)
                print(f"  训练实例 {i+1}/{len(train_instances)}...")
                losses = wrapper.train_on_instance(graph, epochs=10, K=10)
                print(f"    平均损失: {np.mean(losses):.4f}")
            except Exception as e:
                print(f"    训练失败: {e}")
                continue
        
        wrapper.save_model('quick_trained_model.pth')
    
    # --------------------------------------------------------
    # 3. 超参数调优（可选）
    # --------------------------------------------------------
    if len(test_instances) >= 2:
        try:
            tuning_graph = build_mclp_graph(test_instances[1])
            K_values = [5, 10, 15]
            best_params, best_results = run_hyperparameter_tuning(wrapper, tuning_graph, K_values)
            print(f"超参数调优完成")
        except Exception as e:
            print(f"超参数调优跳过: {e}")
    
    # --------------------------------------------------------
    # 4. 求解MCLP（多K值测试）
    # --------------------------------------------------------
    print("\n3. 求解MCLP（多K值测试）...")
    print("=" * 60)
    
    all_results = []
    K_values = [5, 8, 10]
    
    for idx, instance in enumerate(test_instances):
        print(f"\n实例 {idx}: {instance.get('name', '未知')}")
        print(f"  节点数: {len(instance['points'])} | 覆盖半径: {instance.get('coverage_radius', '未知')}")
        
        try:
            # 构建图
            graph = build_mclp_graph(instance)
            print(f"  图构建成功: {graph.num_nodes}节点, {graph.edge_index.shape[1]}边")
            
            instance_results = {}
            
            for K in K_values:
                start_time = time.time()
                
                # 求解（使用多次迭代取最佳）
                selected_indices, coverage, scores = wrapper.solve_mclp(graph, K, iterations=5)
                
                # 计算覆盖率
                if hasattr(graph, 'total_weights') and graph.total_weights is not None:
                    total_demand = torch.sum(graph.total_weights).item()
                else:
                    total_demand = graph.num_nodes
                
                coverage_pct = (coverage / total_demand) * 100 if total_demand > 0 else 0
                solve_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'instance_id': idx,
                    'instance_name': instance.get('name', f'instance_{idx}'),
                    'K': K,
                    'selected_indices': selected_indices.tolist() if selected_indices is not None else [],
                    'coverage': coverage,
                    'total_demand': total_demand,
                    'coverage_percentage': coverage_pct,
                    'time': solve_time,
                    'n_facilities_selected': len(selected_indices) if selected_indices is not None else 0
                }
                
                all_results.append(result)
                instance_results[K] = result
                
                print(f"    K={K:2d}: 覆盖率={coverage_pct:5.1f}% | "
                      f"设施数={len(selected_indices):2d} | 用时={solve_time:.3f}s")
            
            # 为最佳K值可视化
            best_K = max(instance_results.keys(), 
                        key=lambda k: instance_results[k]['coverage_percentage'])
            best_result = instance_results[best_K]
            
            if best_result['coverage_percentage'] > 40:  # 只可视化合理的结果
                enhanced_visualize_solution(
                    instance,
                    best_result['selected_indices'],
                    best_result['coverage_percentage'],
                    best_K,
                    save_path=f'optimized_solution_{idx}_K{best_K}.png',
                    scores=scores if 'scores' in locals() else None
                )
            
        except Exception as e:
            print(f"  求解失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --------------------------------------------------------
    # 4. 高级结果分析
    # --------------------------------------------------------
    if all_results:
        print("\n" + "=" * 60)
        print("高级结果分析")
        print("=" * 60)
        
        # 按K值分组
        results_by_K = {}
        for result in all_results:
            K = result['K']
            if K not in results_by_K:
                results_by_K[K] = []
            results_by_K[K].append(result)
        
        print(f"\n成功求解实例数: {len(set(r['instance_id'] for r in all_results))}/{len(test_instances)}")
        print(f"总求解次数: {len(all_results)}")
        
        # 打印按K值的统计
        print("\n按K值统计:")
        print("-" * 50)
        print("K值 | 平均覆盖率 | 标准差 | 最佳 | 最差 | 实例数")
        print("-" * 50)
        
        for K in sorted(results_by_K.keys()):
            K_results = results_by_K[K]
            coverages = [r['coverage_percentage'] for r in K_results]
            times = [r['time'] for r in K_results]
            
            if coverages:
                print(f"{K:3d} | {np.mean(coverages):9.1f}% | {np.std(coverages):6.1f} | "
                      f"{max(coverages):5.1f}% | {min(coverages):5.1f}% | {len(K_results):6d}")
        
        # 总体统计
        all_coverages = [r['coverage_percentage'] for r in all_results]
        all_times = [r['time'] for r in all_results]
        
        print("\n总体统计:")
        print(f"平均覆盖率: {np.mean(all_coverages):.1f}%")
        print(f"覆盖率范围: {min(all_coverages):.1f}% - {max(all_coverages):.1f}%")
        print(f"覆盖率标准差: {np.std(all_coverages):.1f}%")
        print(f"平均用时: {np.mean(all_times):.3f}s")
        print(f"中位用时: {np.median(all_times):.3f}s")
        
        # 覆盖率分布
        print("\n覆盖率分布:")
        thresholds = [80, 70, 60, 50, 40]
        for threshold in thresholds:
            count = len([c for c in all_coverages if c >= threshold])
            percentage = count / len(all_coverages) * 100
            print(f"  ≥{threshold}%: {count:3d}次 ({percentage:.1f}%)")
        
        # 保存结果
        results_summary = {
            'all_results': all_results,
            'summary': {
                'avg_coverage': np.mean(all_coverages),
                'min_coverage': min(all_coverages),
                'max_coverage': max(all_coverages),
                'std_coverage': np.std(all_coverages),
                'avg_time': np.mean(all_times),
                'results_by_K': results_by_K
            }
        }
        
        with open('optimized_mclp_final_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)
        print(f"\n详细结果已保存: optimized_mclp_final_results.pkl")
        
        # 绘制增强的汇总图
        plot_enhanced_summary(all_results, results_by_K)
        
        # 性能提升分析
        print("\n" + "=" * 60)
        print("性能提升分析")
        print("=" * 60)
        
        original_avg = 45.3  # 原始平均覆盖率
        optimized_avg = np.mean(all_coverages)
        improvement = ((optimized_avg - original_avg) / original_avg) * 100
        
        print(f"原始平均覆盖率: {original_avg:.1f}%")
        print(f"优化后平均覆盖率: {optimized_avg:.1f}%")
        print(f"提升幅度: {improvement:+.1f}%")
        
        if optimized_avg >= 70:
            print(f"✅ 优化成功！达到预期目标（≥70%）")
        elif optimized_avg >= 65:
            print(f"⚠️  接近目标，建议进一步调优")
        else:
            print(f"❌ 未达到预期目标，建议检查数据或增加训练")
        
    else:
        print("\n没有成功求解的实例")
    
    print("\n" + "=" * 80)
    print("MCLP求解完成！")
    print("=" * 80)

def plot_enhanced_summary(all_results, results_by_K):
    """绘制增强的汇总图"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 主图：按K值的覆盖率箱线图+散点图
    ax1 = plt.subplot(2, 2, 1)
    
    K_values = sorted(results_by_K.keys())
    positions = range(1, len(K_values) + 1)
    
    # 箱线图
    box_data = []
    for K in K_values:
        coverages = [r['coverage_percentage'] for r in results_by_K[K]]
        box_data.append(coverages)
    
    bp = ax1.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    # 设置箱线图颜色
    colors = plt.cm.Set2(np.linspace(0, 1, len(K_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加散点（jitter）
    for i, K in enumerate(K_values):
        coverages = [r['coverage_percentage'] for r in results_by_K[K]]
        jitter = np.random.normal(0, 0.05, len(coverages))
        ax1.scatter([positions[i] + j for j in jitter], coverages, 
                   alpha=0.5, s=30, color=colors[i], edgecolor='black', linewidth=0.5)
    
    # 添加均值线
    for i, K in enumerate(K_values):
        coverages = [r['coverage_percentage'] for r in results_by_K[K]]
        mean_coverage = np.mean(coverages)
        ax1.hlines(mean_coverage, positions[i]-0.3, positions[i]+0.3, 
                  colors='red', linewidths=2, label='均值' if i == 0 else "")
    
    ax1.set_xlabel('K值')
    ax1.set_ylabel('覆盖率 (%)')
    ax1.set_title('不同K值的覆盖率分布')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'K={K}' for K in K_values])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 热力图：实例×K值的覆盖率
    ax2 = plt.subplot(2, 2, 2)
    
    # 获取所有实例ID
    instance_ids = sorted(set(r['instance_id'] for r in all_results))
    
    # 创建热力图数据
    heatmap_data = np.full((len(instance_ids), len(K_values)), np.nan)
    
    for i, instance_id in enumerate(instance_ids):
        for j, K in enumerate(K_values):
            # 查找对应的结果
            for result in all_results:
                if result['instance_id'] == instance_id and result['K'] == K:
                    heatmap_data[i, j] = result['coverage_percentage']
                    break
    
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    
    # 添加数值标注
    for i in range(len(instance_ids)):
        for j in range(len(K_values)):
            if not np.isnan(heatmap_data[i, j]):
                ax2.text(j, i, f'{heatmap_data[i, j]:.0f}', 
                        ha='center', va='center', fontsize=8,
                        color='black' if heatmap_data[i, j] > 50 else 'white')
    
    ax2.set_xlabel('K值')
    ax2.set_ylabel('实例ID')
    ax2.set_title('覆盖率热力图（实例×K值）')
    ax2.set_xticks(range(len(K_values)))
    ax2.set_xticklabels([f'K={K}' for K in K_values])
    ax2.set_yticks(range(len(instance_ids)))
    ax2.set_yticklabels([f'Inst_{id}' for id in instance_ids])
    
    plt.colorbar(im, ax=ax2, label='覆盖率 (%)')
    
    # 3. 时间分析
    ax3 = plt.subplot(2, 2, 3)
    
    time_by_K = {}
    for K in K_values:
        times = [r['time'] for r in results_by_K[K]]
        time_by_K[K] = times
    
    time_data = [time_by_K[K] for K in K_values]
    violin_parts = ax3.violinplot(time_data, positions=positions, showmeans=True)
    
    # 设置小提琴图颜色
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    ax3.set_xlabel('K值')
    ax3.set_ylabel('求解时间 (秒)')
    ax3.set_title('求解时间分布')
    ax3.set_xticks(positions)
    ax3.set_xticklabels([f'K={K}' for K in K_values])
    ax3.grid(True, alpha=0.3)
    
    # 4. 覆盖率提升趋势
    ax4 = plt.subplot(2, 2, 4)
    
    # 计算每个K值的平均覆盖率
    avg_coverages = []
    for K in K_values:
        coverages = [r['coverage_percentage'] for r in results_by_K[K]]
        avg_coverages.append(np.mean(coverages))
    
    # 拟合趋势线
    if len(K_values) >= 3:
        z = np.polyfit(K_values, avg_coverages, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(K_values), max(K_values), 100)
        y_smooth = p(x_smooth)
        ax4.plot(x_smooth, y_smooth, '--', color='red', alpha=0.7, label='趋势线')
    
    ax4.plot(K_values, avg_coverages, 'o-', linewidth=2, markersize=8, 
            color='blue', label='平均覆盖率')
    
    # 添加边际效益标注
    if len(avg_coverages) > 1:
        for i in range(1, len(avg_coverages)):
            marginal_gain = avg_coverages[i] - avg_coverages[i-1]
            ax4.annotate(f'+{marginal_gain:.1f}%', 
                        xy=((K_values[i] + K_values[i-1])/2, (avg_coverages[i] + avg_coverages[i-1])/2),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color='green' if marginal_gain > 0 else 'red')
    
    ax4.set_xlabel('K值（设施数量）')
    ax4.set_ylabel('平均覆盖率 (%)')
    ax4.set_title('边际效益分析')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('MCLP求解综合性能分析报告', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('enhanced_results_summary.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("增强汇总图已保存: enhanced_results_summary.png")

if __name__ == '__main__':
    main()