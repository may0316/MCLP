# config.py
import os

class MCLPConfig:
    """MCLP模型统一配置类"""
    def __init__(self, graph_size=50, p=8, radius=0.2, demand_dist='uniform', 
                 dataset_size=10000, data_dir='./data'):
        """
        初始化配置
        
        Args:
            graph_size: 节点数量 (n)
            p: 设施数量
            radius: 覆盖半径
            demand_dist: 需求权重分布类型 ('uniform', 'normal', 'exponential')
            dataset_size: 数据集大小
            data_dir: 数据保存目录
        """
        self.graph_size = graph_size
        self.p = p
        self.radius = radius
        self.demand_dist = demand_dist
        self.dataset_size = dataset_size
        self.data_dir = data_dir
        
        # 自动生成文件名
        self.filename = f"MCLP_{graph_size}_{p}_{radius:.2f}_{demand_dist}.pkl"
        self.data_path = os.path.join(data_dir, self.filename)
        
        # 模型参数
        self.hidden_dim = 128
        self.out_dim = 64
        self.num_epochs_pretrain = 10
        self.num_epochs_solve = 10
        self.batch_size = 16
        self.lr = 0.001
        
        # MoCo参数
        self.momentum = 0.99
        self.queue_size = 256
        
        # 预训练模型保存路径
        self.pretrained_path = f'pretrained_mclp_{graph_size}_{p}_{radius:.2f}.pth'
        
    def __str__(self):
        return (f"MCLPConfig:\n"
                f"  graph_size: {self.graph_size}\n"
                f"  p: {self.p}\n"
                f"  radius: {self.radius}\n"
                f"  demand_dist: {self.demand_dist}\n"
                f"  dataset_size: {self.dataset_size}\n"
                f"  data_path: {self.data_path}\n"
                f"  pretrained_path: {self.pretrained_path}")

# 创建默认配置
default_config = MCLPConfig()

# 如果需要不同配置，可以创建新实例
# small_config = MCLPConfig(graph_size=20, p=4, radius=0.3)
# large_config = MCLPConfig(graph_size=100, p=15, radius=0.15)