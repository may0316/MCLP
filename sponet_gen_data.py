# sponet_gen_data.py
import argparse
import os
import pickle
import torch
import numpy as np
from config import MCLPConfig

def check_extension(filename):
    """检查文件扩展名，确保是.pkl格式"""
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    """保存数据集为pkl文件"""
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        os.makedirs(filedir)
    
    filename_with_ext = check_extension(filename)
    with open(filename_with_ext, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"数据集已保存到: {filename_with_ext}")
    print(f"包含 {len(dataset)} 个MCLP实例，每个实例有 {dataset[0]['loc'].shape[0]} 个节点")


def generate_MCLP_data(n_samples, n_users, p, radius, demand_dist='uniform'):
    """生成MCLP模拟数据集
    
    Args:
        n_samples: 生成的实例数量
        n_users: 每个实例中的节点数量（需求点/候选点）
        p: 需要选择的设施数量
        radius: 覆盖半径
        demand_dist: 需求权重分布类型 ('uniform', 'normal', 'exponential')
    
    Returns:
        list: 包含n_samples个字典的列表，每个字典包含:
              - loc: 节点坐标 (n_users, 2) 的tensor
              - demand: 需求权重 (n_users,) 的tensor
              - p: 设施数量
              - radius: 覆盖半径
    """
    print(f"正在生成 {n_samples} 个MCLP实例...")
    print(f"  节点数量: {n_users}")
    print(f"  设施数量: {p}")
    print(f"  覆盖半径: {radius}")
    print(f"  需求权重分布: {demand_dist}")
    
    data = []
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 个实例")
        
        # 生成随机需求权重
        if demand_dist == 'uniform':
            demand = torch.FloatTensor(n_users).uniform_(0, 1)
        elif demand_dist == 'normal':
            demand = torch.randn(n_users).abs()  # 取绝对值保证非负
            demand = demand / demand.max()  # 归一化到[0,1]
        elif demand_dist == 'exponential':
            demand = torch.exp(torch.randn(n_users) * 0.5)
            demand = demand / demand.max()
        else:
            demand = torch.ones(n_users)  # 默认均匀权重
        
        instance = {
            'loc': torch.FloatTensor(n_users, 2).uniform_(0, 1),  # 在[0,1]正方形内随机生成坐标
            'demand': demand,
            'p': p,
            'radius': radius
        }
        data.append(instance)
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成MCLP（最大覆盖选址问题）模拟数据')
    parser.add_argument('--dataset_size', type=int, default=None,
                        help='要生成的实例数量 (默认使用config中的值)')
    parser.add_argument('--graph_size', type=int, default=None,
                        help='每个实例中的节点数量 (默认使用config中的值)')
    parser.add_argument('--p', type=int, default=None,
                        help='需要选择的设施数量 (默认使用config中的值)')
    parser.add_argument('--radius', type=float, default=None,
                        help='覆盖半径 (默认使用config中的值)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据保存目录 (默认使用config中的值)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='随机种子 (默认: 1234)')
    parser.add_argument('--filename', type=str, default=None,
                        help='自定义文件名 (可选)')
    parser.add_argument('--demand_dist', type=str, default=None,
                        choices=['uniform', 'normal', 'exponential'],
                        help='需求权重分布类型 (默认使用config中的值)')
    
    args = parser.parse_args()
    
    # 创建配置实例
    # 如果提供了参数，则覆盖默认配置
    config_kwargs = {}
    if args.graph_size is not None:
        config_kwargs['graph_size'] = args.graph_size
    if args.p is not None:
        config_kwargs['p'] = args.p
    if args.radius is not None:
        config_kwargs['radius'] = args.radius
    if args.demand_dist is not None:
        config_kwargs['demand_dist'] = args.demand_dist
    if args.data_dir is not None:
        config_kwargs['data_dir'] = args.data_dir
    if args.dataset_size is not None:
        config_kwargs['dataset_size'] = args.dataset_size
    
    config = MCLPConfig(**config_kwargs)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"随机种子设置为: {args.seed}")
    print(f"\n{config}")
    
    # 生成数据
    dataset = generate_MCLP_data(
        n_samples=config.dataset_size,
        n_users=config.graph_size,
        p=config.p,
        radius=config.radius,
        demand_dist=config.demand_dist
    )
    
    # 确定文件名
    if args.filename:
        filename = args.filename
    else:
        filename = config.filename
    
    # 完整保存路径
    save_path = os.path.join(config.data_dir, filename)
    
    # 保存数据
    save_dataset(dataset, save_path)
    
    print("\n数据生成完成！")
    print(f"示例实例:")
    sample = dataset[0]
    print(f"  节点坐标形状: {sample['loc'].shape}")
    print(f"  需求权重形状: {sample['demand'].shape}")
    print(f"  需求权重范围: [{sample['demand'].min():.4f}, {sample['demand'].max():.4f}]")
    print(f"  需求权重和: {sample['demand'].sum():.4f}")