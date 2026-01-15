import numpy as np
import torch
import torch.optim as optim
import pickle
import os
import time
from create_more import MCLPDatasetGenerator, build_mclp_graph, save_dataset, load_dataset
from self_supervised_mclp import SelfSupervisedMCLPWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def main():
    print("文旅MCLP自监督预训练")
    print("="*60)
    
    # 1. 生成或加载数据
    print("处理数据集...")
    dataset_file = 'mclp_tourism_train_100.pkl'
    
    if not os.path.exists(dataset_file):
        print("生成新的文旅数据集...")
        generator = MCLPDatasetGenerator(
            num_nodes=200,
            num_instances=100,
            device=device,
            include_tourism_features=True,
            tourism_hotspots=8
        )
        dataset = generator.generate_dataset()
        save_dataset(dataset, dataset_file)
    else:
        dataset = load_dataset(dataset_file)
    
    print(f"数据集包含 {len(dataset)} 个实例")
    
    if len(dataset) == 0:
        print("数据集为空，退出")
        return
    
    # 2. 初始化模型
    print("\n初始化自监督模型...")
    wrapper = SelfSupervisedMCLPWrapper(device=device)
    
    # 使用第一个实例确定输入维度
    test_instance = dataset[0]
    test_graph = build_mclp_graph(test_instance)
    input_dim = test_graph.x.shape[1]
    
    wrapper.initialize_model(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=64
    )
    
    # 3. 预训练
    print("\n开始预训练...")
    all_losses = []
    start_time = time.time()
    
    # 使用前20个实例训练
    train_instances = dataset[:20]
    
    for i, instance in enumerate(train_instances):
        print(f"\n训练实例 {i+1}/{len(train_instances)}: {instance['name']}")
        
        try:
            # 构建图
            graph = build_mclp_graph(instance)
            
            # 训练
            losses = wrapper.train_on_instance(
                graph=graph,
                epochs=50,  # 每个实例训练50轮
                batch_size=32
            )
            
            all_losses.extend(losses)
            avg_loss = np.mean(losses) if losses else 0
            print(f"  平均损失: {avg_loss:.4f}")
            
            # 每5个实例保存一次模型
            if (i + 1) % 5 == 0:
                torch.save(wrapper.model.state_dict(), f'moco_mclp_checkpoint_{i+1}.pth')
                print(f"  检查点已保存")
                
        except Exception as e:
            print(f"  训练失败: {e}")
            continue
    
    # 4. 保存最终模型
    if wrapper.model is not None:
        print("\n保存最终模型...")
        torch.save(wrapper.model.state_dict(), 'moco_mclp_pretrained.pth')
        print("模型已保存")
    
    # 5. 测试
    print("\n测试模型性能...")
    test_results = []
    
    # 使用后10个实例测试
    test_instances = dataset[-10:] if len(dataset) > 10 else dataset
    
    for i, instance in enumerate(test_instances):
        try:
            graph = build_mclp_graph(instance)
            
            # 测试不同K值
            for K in [5, 8, 10]:
                selected_indices, coverage, scores = wrapper.solve_mclp(
                    graph=graph,
                    K=K
                )
                
                n_nodes = len(instance['points'])
                coverage_percentage = (coverage / n_nodes) * 100
                
                result = {
                    'instance_id': i,
                    'instance_name': instance['name'],
                    'K': K,
                    'n_nodes': n_nodes,
                    'coverage': coverage,
                    'coverage_percentage': coverage_percentage,
                    'selected_indices': selected_indices.tolist()
                }
                test_results.append(result)
                
                print(f"  实例 {instance['name']}, K={K}: 覆盖率={coverage_percentage:.1f}%")
                
        except Exception as e:
            print(f"  测试失败: {e}")
            continue
    
    # 6. 保存结果
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
        
        # 计算平均覆盖率
        avg_coverage = np.mean([r['coverage_percentage'] for r in test_results if 'coverage_percentage' in r])
        print(f"\n平均测试覆盖率: {avg_coverage:.1f}%")
    
    print("\n预训练完成!")

if __name__ == '__main__':
    main()