import time
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from load_real_data import load_osm_poi_data

from create_more import (
    MCLPDatasetGenerator,
    build_mclp_graph,
    load_dataset,
    save_dataset
)
from self_supervised_mclp import SelfSupervisedMCLPWrapper

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("文旅场景 MCLP 求解（自监督 Soft-MCLP → Hard 推理）")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 加载或生成数据
    # --------------------------------------------------------
    print("\n加载文旅 MCLP 数据集...")
    dataset_file = 'mclp_tourism_test_50.pkl'
    
    shared_points, shared_tourism_features = load_osm_poi_data(
    max_points=200,
    device=device
    )
    if not os.path.exists(dataset_file):
        print("未找到数据集，生成新数据...")
        generator = MCLPDatasetGenerator(
            num_nodes=200,
            num_instances=20,
            device=device,
            include_tourism_features=True,
            tourism_hotspots=8,
            shared_points=shared_points,
            shared_tourism_features=shared_tourism_features
        )
        dataset = generator.generate_dataset()
        save_dataset(dataset, dataset_file)
    else:
        dataset = load_dataset(dataset_file)

    print(f"数据集实例数: {len(dataset)}")

    # 测试阶段只用少量
    dataset = dataset[:5]

    # --------------------------------------------------------
    # 2. 初始化自监督 MCLP 模型
    # --------------------------------------------------------
    print("\n初始化 Self-Supervised MCLP 模型...")
    wrapper = SelfSupervisedMCLPWrapper(device=device)

    test_graph = build_mclp_graph(dataset[0])
    input_dim = test_graph.x.shape[1]

    wrapper.initialize_model(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=64
    )

    model_path = 'moco_mclp_pretrained.pth'
    if os.path.exists(model_path):
        wrapper.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        print("已加载预训练模型")
    else:
        print(" 未找到预训练模型，使用随机初始化")

    # --------------------------------------------------------
    # 3. 求解 MCLP
    # --------------------------------------------------------
    K = 10
    all_results = []

    print(f"\n开始求解 MCLP（K={K}）")
    print("=" * 70)

    for idx, instance in enumerate(dataset):
        print(f"\n处理实例 {idx}: {instance['name']}")
        start_time = time.time()

        try:
            # 构建图
            graph = build_mclp_graph(instance)

            # -----------------------------
            # 核心：Hard 推理（Top-K）
            # -----------------------------
            wrapper.model.eval()
            with torch.no_grad():
                _, scores = wrapper.model(graph)
                selected_indices = torch.topk(scores, K).indices.cpu().numpy()

            # -----------------------------
            # 覆盖率计算（真实 MCLP）
            # -----------------------------
            dist_matrix = graph.distance_matrix
            coverage_radius = graph.coverage_radius

            dist_to_sel = dist_matrix[:, selected_indices]
            min_dist = torch.min(dist_to_sel, dim=1)[0]
            covered_mask = (min_dist <= coverage_radius)

            if hasattr(graph, 'total_weights') and graph.total_weights is not None:
                covered = torch.sum(graph.total_weights[covered_mask]).item()
                total = torch.sum(graph.total_weights).item()
            else:
                covered = covered_mask.float().sum().item()
                total = graph.num_nodes

            coverage_pct = 100.0 * covered / max(total, 1)

            result = {
                'instance_id': idx,
                'instance_name': instance['name'],
                'selected_indices': selected_indices.tolist(),
                'coverage': covered,
                'total_demand': total,
                'coverage_percentage': coverage_pct,
                'n_nodes': graph.num_nodes,
                'K': K,
                'coverage_radius': coverage_radius,
                'time': time.time() - start_time
            }

            all_results.append(result)

            print(f"  覆盖率: {coverage_pct:.2f}%")
            print(f"  用时: {result['time']:.2f}s")

            visualize_result(instance, selected_indices, result)

        except Exception as e:
            print(f"实例 {idx} 失败: {e}")
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------
    # 4. 汇总结果
    # --------------------------------------------------------
    if all_results:
        print("\n" + "=" * 70)
        coverages = [r['coverage_percentage'] for r in all_results]
        times = [r['time'] for r in all_results]

        print(f"平均覆盖率: {np.mean(coverages):.2f}% ± {np.std(coverages):.2f}%")
        print(f"平均用时: {np.mean(times):.2f}s")

        with open('mclp_tourism_results.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        visualize_statistics(all_results)


# ============================================================
# 可视化
# ============================================================

def visualize_result(instance, selected_indices, result):
    points = instance['points'].cpu().numpy()
    selected_points = points[selected_indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.6, label='候选点')
    ax.scatter(
        selected_points[:, 0],
        selected_points[:, 1],
        s=150,
        c='red',
        marker='X',
        label='选定设施'
    )

    r = result['coverage_radius']
    for idx in selected_indices:
        circle = plt.Circle(points[idx], r, color='red', alpha=0.08)
        ax.add_patch(circle)

    ax.set_title(f"{instance['name']} | 覆盖率 {result['coverage_percentage']:.1f}%")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.show()


def visualize_statistics(all_results):
    coverages = [r['coverage_percentage'] for r in all_results]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(coverages)), coverages)
    plt.ylabel("覆盖率 (%)")
    plt.xlabel("实例")
    plt.title("文旅 MCLP 覆盖率对比")
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


# ============================================================
if __name__ == '__main__':
    main()
