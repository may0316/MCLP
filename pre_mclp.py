import numpy as np
import torch
import pickle
import os
from create_more import MCLPDatasetGenerator, save_dataset, load_dataset
from self_supervised_mclp import SelfSupervisedMCLPWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def main():
    print("自监督MCLP预训练")
    print("="*60)
    
    # 1. 生成或加载数据
    print("处理数据集...")
    dataset_file = 'mclp_beijing_test_20.pkl'  # 使用更小的数据集
    
    if not os.path.exists(dataset_file):
        print("生成新的数据集...")
        generator = MCLPDatasetGenerator(
            num_nodes=100,  # 更少的节点
            num_instances=20,  # 更少的实例
            device=device
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
    self_supervised = SelfSupervisedMCLPWrapper(device=device)
    
    # 3. 训练
    print("\n开始训练...")
    all_losses = []
    
    # 只使用前3个实例训练，减少复杂度
    train_instances = dataset[:3]
    
    for i, instance in enumerate(train_instances):
        print(f"\n训练实例 {i}: {instance['name']}")
        
        try:
            losses = self_supervised.train_on_instance(instance, epochs=10)
            all_losses.extend(losses)
            print(f"  训练完成，损失数: {len(losses)}")
        except Exception as e:
            print(f"  训练失败: {e}")
            continue
    
    # 4. 保存模型
    if self_supervised.model is not None:
        print("\n保存模型...")
        torch.save(self_supervised.model.state_dict(), 'self_supervised_mclp_simple.pth')
        print("模型已保存")
    
    # 5. 测试
    print("\n测试模型...")
    if len(dataset) > 3:
        test_instance = dataset[3]
        try:
            selected_indices, coverage = self_supervised.solve_mclp(test_instance, K=5)
            n_nodes = len(test_instance['points'])
            coverage_percentage = coverage / n_nodes * 100
            print(f"测试实例覆盖率: {coverage_percentage:.1f}%")
        except Exception as e:
            print(f"测试失败: {e}")
    
    print("\n预训练完成!")

if __name__ == '__main__':
    main()