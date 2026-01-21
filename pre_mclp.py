import numpy as np
import torch
import torch.optim as optim
import pickle
import os
import time
from create_more import MCLPDatasetGenerator, build_mclp_graph, save_dataset, load_dataset
from self_supervised_mclp import SelfSupervisedMCLPWrapper
from load_real_data import load_osm_poi_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
def main():
    print("文旅MCLP自监督预训练")
    print("=" * 60)

    # =====================================================
    # 1. 加载真实 OSM POI（只做一次！！！）
    # =====================================================
    print("加载真实 OSM POI（只做一次）...")
    shared_points, shared_tourism_features = load_osm_poi_data(
        max_points=200,
        device=device
    )
    print(f"已加载 POI 数量: {shared_points.shape[0]}")

    # =====================================================
    # 2. 生成或加载数据集
    # =====================================================
    print("\n处理数据集...")
    dataset_file = 'mclp_tourism_train_100.pkl'

    if not os.path.exists(dataset_file):
        print("生成新的文旅数据集...")

        generator = MCLPDatasetGenerator(
            num_nodes=200,
            num_instances=100,
            device=device,
            include_tourism_features=True,
            tourism_hotspots=8,
            # ✅ 关键：传入真实 POI
            shared_points=shared_points,
            shared_tourism_features=shared_tourism_features
        )

        dataset = generator.generate_dataset()
        save_dataset(dataset, dataset_file)

    else:
        print("从本地加载已有数据集...")
        dataset = load_dataset(dataset_file)

    print(f"数据集包含 {len(dataset)} 个实例")

    if len(dataset) == 0:
        print("数据集为空，退出")
        return

    # =====================================================
    # 3. 初始化自监督模型
    # =====================================================
    print("\n初始化自监督模型...")
    wrapper = SelfSupervisedMCLPWrapper(device=device)

    # 用第一个实例确定输入维度
    test_instance = dataset[0]
    test_graph = build_mclp_graph(test_instance)
    input_dim = test_graph.x.shape[1]

    wrapper.initialize_model(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=64
    )

    # =====================================================
    # 4. 预训练
    # =====================================================
    print("\n开始预训练...")
    all_losses = []
    start_time = time.time()

    train_instances = dataset[:20]  # 用前20个实例训练

    for i, instance in enumerate(train_instances):
        print(f"\n训练实例 {i+1}/{len(train_instances)}: {instance['name']}")

        try:
            graph = build_mclp_graph(instance)

            losses = wrapper.train_on_instance(
                graph=graph,
                epochs=50,
                batch_size=32
            )

            all_losses.extend(losses)
            avg_loss = np.mean(losses) if losses else 0
            print(f"  平均损失: {avg_loss:.4f}")

            if (i + 1) % 5 == 0:
                torch.save(
                    wrapper.model.state_dict(),
                    f'moco_mclp_checkpoint_{i+1}.pth'
                )
                print("  检查点已保存")

        except Exception as e:
            print(f"  训练失败: {e}")
            continue

    # =====================================================
    # 5. 保存最终模型
    # =====================================================
    if wrapper.model is not None:
        print("\n保存最终模型...")
        torch.save(
            wrapper.model.state_dict(),
            'moco_mclp_pretrained.pth'
        )
        print("模型已保存")

    # =====================================================
    # 6. 测试
    # =====================================================
    print("\n测试模型性能...")
    test_results = []

    test_instances = dataset[-10:] if len(dataset) > 10 else dataset

    for i, instance in enumerate(test_instances):
        try:
            graph = build_mclp_graph(instance)

            for K in [5, 8, 10]:
                selected_indices, coverage, scores = wrapper.solve_mclp(
                    graph=graph,
                    K=K
                )

                n_nodes = len(instance['points'])
                coverage_percentage = (coverage / n_nodes) * 100

                test_results.append({
                    'instance_id': i,
                    'instance_name': instance['name'],
                    'K': K,
                    'coverage': coverage,
                    'coverage_percentage': coverage_percentage,
                    'selected_indices': selected_indices.tolist()
                })

                print(
                    f"  实例 {instance['name']}, K={K}: "
                    f"覆盖率={coverage_percentage:.1f}%"
                )

        except Exception as e:
            print(f"  测试失败: {e}")
            continue

    # =====================================================
    # 7. 保存结果
    # =====================================================
    if test_results:
        results_summary = {
            'all_losses': all_losses,
            'test_results': test_results,
            'training_time': time.time() - start_time,
            'num_train_instances': len(train_instances),
            'num_test_instances': len(test_instances)
        }

        with open('mclp_pretrain_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)

        avg_coverage = np.mean(
            [r['coverage_percentage'] for r in test_results]
        )
        print(f"\n平均测试覆盖率: {avg_coverage:.1f}%")

    print("\n预训练完成！")

if __name__ == '__main__':
    main()
    