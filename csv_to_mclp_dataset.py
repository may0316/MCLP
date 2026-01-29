# csv_to_mclp_dataset.py
"""
将高德地图CSV转换为MCLP数据集的完整工具
python csv_to_mclp_dataset.py --input amap_pois.csv --output mclp_dataset.pkl
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVToMCLPConverter:
    """将高德地图CSV转换为MCLP数据集的转换器"""
    
    def __init__(self, 
                 coord_range: Tuple[float, float] = (0.0, 100.0),
                 device: torch.device = torch.device('cpu')):
        """
        初始化转换器
        
        Args:
            coord_range: 坐标归一化范围
            device: 计算设备
        """
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        self.device = device
        
        # POI类型到特征的映射
        self.poi_feature_mapping = {
            # 旅游景点类
            '风景名胜': [0.85, 0.45, 0.35],
            '公园广场': [0.75, 0.55, 0.45],
            '博物馆': [0.90, 0.65, 0.55],
            '纪念馆': [0.80, 0.45, 0.35],
            '寺庙道观': [0.70, 0.45, 0.25],
            '动物园': [0.85, 0.55, 0.45],
            '植物园': [0.75, 0.45, 0.35],
            '度假村': [0.90, 0.60, 0.70],
            '游乐园': [0.95, 0.65, 0.55],
            '水族馆': [0.85, 0.50, 0.40],
            
            # 餐饮类
            '中餐厅': [0.55, 0.35, 0.75],
            '外国餐厅': [0.65, 0.45, 0.85],
            '快餐厅': [0.35, 0.65, 0.85],
            '咖啡厅': [0.45, 0.55, 0.75],
            '茶艺馆': [0.50, 0.45, 0.65],
            '火锅店': [0.60, 0.50, 0.80],
            '小吃店': [0.40, 0.55, 0.70],
            
            # 住宿类
            '宾馆酒店': [0.65, 0.75, 0.85],
            '旅馆招待所': [0.45, 0.55, 0.65],
            '公寓式酒店': [0.70, 0.70, 0.80],
            
            # 购物类
            '购物中心': [0.55, 0.85, 0.95],
            '百货商场': [0.50, 0.80, 0.90],
            '超市': [0.25, 0.65, 0.85],
            '便利店': [0.15, 0.75, 0.95],
            '市场': [0.30, 0.70, 0.80],
            
            # 交通类
            '地铁站': [0.25, 0.95, 0.55],
            '公交车站': [0.15, 0.85, 0.45],
            '停车场': [0.15, 0.75, 0.65],
            '加油站': [0.10, 0.70, 0.60],
            '火车站': [0.30, 0.90, 0.50],
            '飞机场': [0.40, 0.95, 0.70],
            
            # 服务设施类
            '医院': [0.15, 0.65, 0.95],
            '银行': [0.15, 0.55, 0.85],
            '公共厕所': [0.10, 0.35, 0.95],
            '公安局': [0.05, 0.45, 0.75],
            '消防局': [0.05, 0.40, 0.70],
            '邮局': [0.10, 0.50, 0.80],
            
            # 休闲娱乐类
            '电影院': [0.65, 0.75, 0.85],
            '剧院': [0.70, 0.65, 0.75],
            'KTV': [0.60, 0.70, 0.80],
            '酒吧': [0.70, 0.65, 0.85],
            '体育场馆': [0.75, 0.60, 0.70],
            '健身房': [0.55, 0.65, 0.80],
            '游泳馆': [0.60, 0.55, 0.75],
        }
    
    def load_and_clean_csv(self, csv_path: str, max_points: Optional[int] = None) -> pd.DataFrame:
        """加载并清洗CSV数据"""
        logger.info(f"加载CSV文件: {csv_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            logger.info(f"原始数据: {len(df)} 条记录")
            
            # 检查必要的列
            required_cols = ['lon', 'lat']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV缺少必要列: {missing_cols}")
            
            # 数据清洗
            df_clean = df.copy()
            
            # 1. 去除坐标为空的行
            df_clean = df_clean.dropna(subset=['lon', 'lat'])
            
            # 2. 去除重复坐标
            df_clean = df_clean.drop_duplicates(subset=['lon', 'lat'])
            
            # 3. 去除异常坐标
            df_clean = df_clean[
                (df_clean['lon'] >= 73) & (df_clean['lon'] <= 136) &
                (df_clean['lat'] >= 18) & (df_clean['lat'] <= 54)
            ]
            
            # 4. 限制点数
            if max_points is not None and len(df_clean) > max_points:
                df_clean = df_clean.sample(n=max_points, random_state=42)
            
            logger.info(f"清洗后数据: {len(df_clean)} 条记录")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"加载CSV失败: {e}")
            raise
    
    def normalize_coordinates(self, df: pd.DataFrame) -> torch.Tensor:
        """将经纬度坐标归一化到指定范围"""
        lon = df['lon'].values.astype(np.float32)
        lat = df['lat'].values.astype(np.float32)
        
        # 计算实际地理范围（米）
        avg_lat = np.mean(lat)
        lon_scale = 111.32 * np.cos(np.radians(avg_lat)) * 1000  # 米/度
        lat_scale = 111.32 * 1000  # 米/度
        
        # 转换为平面坐标（米）
        x_meters = (lon - lon.min()) * lon_scale
        y_meters = (lat - lat.min()) * lat_scale
        
        # 归一化到目标范围
        x_normalized = x_meters / max(x_meters.max(), 1) * self.range_size + self.min_val
        y_normalized = y_meters / max(y_meters.max(), 1) * self.range_size + self.min_val
        
        # 转换为张量
        points = torch.tensor(
            np.stack([x_normalized, y_normalized], axis=1),
            dtype=torch.float32,
            device=self.device
        )
        
        logger.info(f"坐标归一化完成: {len(points)} 个点")
        logger.info(f"坐标范围: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] × "
                   f"[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
        
        return points
    
    def extract_features(self, df: pd.DataFrame) -> torch.Tensor:
        """从CSV数据中提取或计算特征"""
        n = len(df)
        features = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        
        # 检查CSV是否已有特征列
        feature_cols = ['tourism_score', 'traffic_score', 'facility_score']
        
        if all(col in df.columns for col in feature_cols):
            # 直接使用CSV中的特征
            logger.info("使用CSV中的特征数据")
            
            for i, col in enumerate(feature_cols):
                if col in df.columns:
                    col_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                    # 归一化到 [0, 1]
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max > col_min:
                        features[:, i] = torch.tensor(
                            (col_data - col_min) / (col_max - col_min),
                            dtype=torch.float32,
                            device=self.device
                        )
            
        else:
            # 从POI类型推断特征
            logger.info("从POI类型推断特征")
            
            for idx, row in df.iterrows():
                poi_type = str(row.get('type', '')).strip()
                poi_name = str(row.get('name', '')).lower() if pd.notna(row.get('name')) else ""
                
                # 优先匹配类型
                matched = False
                for type_key, weights in self.poi_feature_mapping.items():
                    if type_key in poi_type:
                        features[idx] = torch.tensor(weights, dtype=torch.float32, device=self.device)
                        matched = True
                        break
                
                # 如果没有匹配到，根据名称关键词推断
                if not matched:
                    # 关键词映射
                    keyword_weights = {
                        'tourism': ['景点', '景区', '旅游', '观光', '游览', '名胜', '古迹', '公园', '广场'],
                        'traffic': ['车站', '地铁', '公交', '停车', '交通', '枢纽', '客运', '机场', '码头'],
                        'facility': ['餐厅', '饭店', '酒店', '宾馆', '商场', '超市', '医院', '银行', '学校']
                    }
                    
                    # 计算特征
                    tourism_score = 0.2
                    traffic_score = 0.3
                    facility_score = 0.4
                    
                    for keyword in keyword_weights['tourism']:
                        if keyword in poi_name:
                            tourism_score += 0.15
                    
                    for keyword in keyword_weights['traffic']:
                        if keyword in poi_name:
                            traffic_score += 0.2
                    
                    for keyword in keyword_weights['facility']:
                        if keyword in poi_name:
                            facility_score += 0.15
                    
                    # 限制范围
                    tourism_score = min(tourism_score, 0.9)
                    traffic_score = min(traffic_score, 0.9)
                    facility_score = min(facility_score, 0.95)
                    
                    features[idx] = torch.tensor(
                        [tourism_score, traffic_score, facility_score],
                        dtype=torch.float32,
                        device=self.device
                    )
        
        # 如果CSV中有rating和cost列，用它们微调特征
        if 'rating' in df.columns:
            ratings = pd.to_numeric(df['rating'], errors='coerce').fillna(0).values
            for i, rating in enumerate(ratings):
                if rating > 0:
                    features[i, 0] += min(rating / 5.0 * 0.2, 0.2)
        
        if 'cost' in df.columns:
            costs = pd.to_numeric(df['cost'], errors='coerce').fillna(0).values
            for i, cost in enumerate(costs):
                if cost > 0 and 30 < cost < 150:
                    features[i, 2] += 0.1
        
        # 确保特征在 [0, 1] 范围内
        features = torch.clamp(features, 0.0, 1.0)
        
        logger.info(f"特征提取完成: {features.shape}")
        logger.info(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
        
        return features
    
    def generate_spatial_labels(self, points: torch.Tensor) -> torch.Tensor:
        """生成空间聚类标签"""
        points_np = points.cpu().numpy()
        
        # 确定聚类数量
        n_clusters = min(8, max(2, len(points) // 25))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points_np)
        
        spatial_labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        logger.info(f"生成空间标签: {spatial_labels.unique().numel()} 个聚类")
        
        return spatial_labels
    
    def generate_weights(self, points: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成各种权重"""
        n = len(points)
        
        # 1. 人口密度权重（基于特征和位置）
        population_weights = torch.zeros(n, device=self.device)
        
        # 旅游热点区域权重更高
        population_weights += features[:, 0] * 0.6
        
        # 设施密集区域权重更高
        population_weights += features[:, 2] * 0.4
        
        # 归一化
        if population_weights.max() > population_weights.min():
            population_weights = (population_weights - population_weights.min()) / \
                               (population_weights.max() - population_weights.min()) * 0.8 + 0.1
        
        # 2. 道路可达性权重
        road_weights = torch.zeros(n, device=self.device)
        
        # 交通便利地区权重更高
        road_weights += features[:, 1] * 0.7
        
        # 添加随机道路网络效果
        for _ in range(3):  # 模拟3条主要道路
            start_x = torch.rand(1, device=self.device) * self.range_size + self.min_val
            start_y = torch.rand(1, device=self.device) * self.range_size + self.min_val
            end_x = torch.rand(1, device=self.device) * self.range_size + self.min_val
            end_y = torch.rand(1, device=self.device) * self.range_size + self.min_val
            
            # 计算点到道路的距离
            road_vec = torch.tensor([end_x - start_x, end_y - start_y], device=self.device)
            road_len = torch.norm(road_vec)
            
            if road_len > 0:
                road_vec = road_vec / road_len
                
                for i in range(n):
                    point_vec = points[i] - torch.tensor([start_x, start_y], device=self.device)
                    proj_len = torch.dot(point_vec, road_vec)
                    proj_len = torch.clamp(proj_len, 0, road_len)
                    
                    nearest_point = torch.tensor([start_x, start_y], device=self.device) + proj_len * road_vec
                    dist = torch.norm(points[i] - nearest_point)
                    
                    # 距离道路越近，权重越高
                    road_weights[i] += torch.exp(-dist / (self.range_size * 0.1)) * 0.3
        
        # 归一化
        if road_weights.max() > road_weights.min():
            road_weights = (road_weights - road_weights.min()) / \
                         (road_weights.max() - road_weights.min()) * 0.8 + 0.1
        
        # 3. 总权重（综合权重）
        total_weights = torch.ones(n, device=self.device) * 0.3
        total_weights += population_weights * 0.4
        total_weights += road_weights * 0.3
        total_weights += features[:, 2] * 0.3  # 设施需求度
        
        # 归一化
        total_weights = torch.clamp(total_weights, 0.1, 1.0)
        if total_weights.max() > total_weights.min():
            total_weights = (total_weights - total_weights.min()) / \
                          (total_weights.max() - total_weights.min()) * 0.9 + 0.1
        
        weights = {
            'population_weights': population_weights,
            'road_weights': road_weights,
            'total_weights': total_weights
        }
        
        logger.info(f"权重生成完成")
        logger.info(f"人口权重范围: [{population_weights.min():.3f}, {population_weights.max():.3f}]")
        logger.info(f"道路权重范围: [{road_weights.min():.3f}, {road_weights.max():.3f}]")
        logger.info(f"总权重范围: [{total_weights.min():.3f}, {total_weights.max():.3f}]")
        
        return weights
    
    def generate_regions(self, points: torch.Tensor) -> torch.Tensor:
        """生成区域标签"""
        points_np = points.cpu().numpy()
        
        # 根据数据量确定区域数量
        n_regions = min(10, max(3, len(points) // 30))
        
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        region_labels = kmeans.fit_predict(points_np)
        
        regions = torch.tensor(region_labels, dtype=torch.long, device=self.device)
        
        logger.info(f"生成区域标签: {regions.unique().numel()} 个区域")
        
        return regions
    
    def calculate_distance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """计算距离矩阵"""
        # 使用欧几里得距离
        A = points.unsqueeze(dim=-2)  # shape: (n, 1, 2)
        B = points.unsqueeze(dim=-3)  # shape: (1, n, 2)
        
        dist = (A - B) ** 2.0
        dist = dist.sum(dim=-1)
        dist = torch.sqrt(dist)
        
        logger.info(f"距离矩阵计算完成: {dist.shape}")
        logger.info(f"平均距离: {dist[dist > 0].mean():.2f}")
        logger.info(f"最大距离: {dist.max():.2f}")
        
        return dist
    
    def calculate_coverage_radius(self, distance_matrix: torch.Tensor) -> float:
        """计算覆盖半径"""
        if distance_matrix.numel() == 0:
            return self.range_size * 0.15
        
        # 获取非零距离
        non_zero_dists = distance_matrix[distance_matrix > 0]
        
        if len(non_zero_dists) == 0:
            return self.range_size * 0.15
        
        # 计算平均距离
        avg_dist = non_zero_dists.mean().item()
        
        # 覆盖半径设置为平均距离的0.25倍
        coverage_radius = avg_dist * 0.25
        
        # 限制在合理范围内
        min_radius = self.range_size * 0.05
        max_radius = self.range_size * 0.3
        
        coverage_radius = max(min(coverage_radius, max_radius), min_radius)
        
        logger.info(f"计算覆盖半径: {coverage_radius:.2f} (平均距离: {avg_dist:.2f})")
        
        return coverage_radius
    
    def create_mclp_instances(self, 
                             points: torch.Tensor,
                             features: torch.Tensor,
                             spatial_labels: torch.Tensor,
                             num_instances: int = 50,
                             instance_variation: float = 0.02) -> List[Dict]:
        """创建多个MCLP实例（添加实例级变化）"""
        logger.info(f"创建 {num_instances} 个MCLP实例...")
        
        instances = []
        
        for i in range(num_instances):
            instance_start_time = time.time()
            
            # 为每个实例创建随机种子
            seed = 42 + i * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 1. 基础数据（添加轻微扰动）
            instance_points = points.clone()
            
            # 添加随机扰动（模拟不同时间/条件下的变化）
            noise_scale = self.range_size * instance_variation
            noise = torch.randn_like(instance_points) * noise_scale
            instance_points = instance_points + noise
            instance_points = torch.clamp(instance_points, self.min_val, self.max_val)
            
            # 2. 特征（轻微变化）
            instance_features = features.clone()
            feature_noise = torch.randn_like(instance_features) * 0.05
            instance_features = instance_features + feature_noise
            instance_features = torch.clamp(instance_features, 0.0, 1.0)
            
            # 3. 重新计算权重（每个实例略有不同）
            weights = self.generate_weights(instance_points, instance_features)
            
            # 4. 计算距离矩阵
            distance_matrix = self.calculate_distance_matrix(instance_points)
            
            # 5. 计算覆盖半径
            coverage_radius = self.calculate_coverage_radius(distance_matrix)
            
            # 6. 生成区域标签
            region_labels = self.generate_regions(instance_points)
            
            # 7. 构建实例
            instance = {
                'name': f'mclp_real_{i:04d}',
                'instance_id': i,
                'points': instance_points,
                'spatial_labels': spatial_labels,
                'tourism_features': instance_features,
                'population_weights': weights['population_weights'],
                'road_weights': weights['road_weights'],
                'total_weights': weights['total_weights'],
                'region_labels': region_labels,
                'distance_matrix': distance_matrix,
                'coverage_radius': coverage_radius,
                'avg_distance': distance_matrix[distance_matrix > 0].mean().item() if len(distance_matrix[distance_matrix > 0]) > 0 else 0.0,
                'diameter': distance_matrix.max().item(),
                'num_nodes': len(instance_points),
                'data_source': 'real_amap',
                'generation_time': time.time() - instance_start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            instances.append(instance)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  已创建 {i + 1}/{num_instances} 个实例")
        
        logger.info(f"MCLP实例创建完成: {len(instances)} 个实例")
        
        return instances
    
    def convert(self, 
               csv_path: str,
               output_path: str,
               num_instances: int = 50,
               max_points: Optional[int] = None,
               instance_variation: float = 0.02) -> Dict[str, Any]:
        """
        主转换函数
        
        Args:
            csv_path: 输入CSV文件路径
            output_path: 输出文件路径
            num_instances: 生成的实例数量
            max_points: 最大POI点数
            instance_variation: 实例间变化程度
        
        Returns:
            转换后的数据集信息
        """
        total_start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("开始转换高德地图CSV到MCLP数据集")
        logger.info(f"输入文件: {csv_path}")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"实例数量: {num_instances}")
        logger.info(f"最大点数: {max_points}")
        logger.info("=" * 60)
        
        try:
            # 1. 加载和清洗CSV数据
            df = self.load_and_clean_csv(csv_path, max_points)
            
            # 2. 归一化坐标
            points = self.normalize_coordinates(df)
            
            # 3. 提取特征
            features = self.extract_features(df)
            
            # 4. 生成空间标签
            spatial_labels = self.generate_spatial_labels(points)
            
            # 5. 创建MCLP实例
            instances = self.create_mclp_instances(
                points, features, spatial_labels, 
                num_instances, instance_variation
            )
            
            # 6. 保存数据集
            dataset_info = self.save_dataset(instances, output_path, df)
            
            total_time = time.time() - total_start_time
            
            logger.info("=" * 60)
            logger.info("转换完成!")
            logger.info(f"总耗时: {total_time:.2f}秒")
            logger.info(f"实例数: {len(instances)}")
            logger.info(f"平均点数: {np.mean([inst['num_nodes'] for inst in instances]):.0f}")
            logger.info(f"平均覆盖半径: {np.mean([inst['coverage_radius'] for inst in instances]):.2f}")
            logger.info(f"输出文件: {output_path}")
            logger.info("=" * 60)
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"转换失败: {e}")
            raise
    
    def save_dataset(self, instances: List[Dict], output_path: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """保存数据集到文件"""
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存主数据集 (pickle格式)
        with open(output_path, 'wb') as f:
            pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 2. 保存数据集信息 (JSON格式)
        info_path = output_path.replace('.pkl', '_info.json')
        
        dataset_info = {
            'num_instances': len(instances),
            'num_points': instances[0]['num_nodes'] if instances else 0,
            'coord_range': [self.min_val, self.max_val],
            'coverage_radius_range': [
                min(inst['coverage_radius'] for inst in instances),
                max(inst['coverage_radius'] for inst in instances)
            ],
            'avg_distance_range': [
                min(inst['avg_distance'] for inst in instances),
                max(inst['avg_distance'] for inst in instances)
            ],
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_csv': os.path.basename(original_df.attrs.get('source', 'unknown')) if hasattr(original_df, 'attrs') else 'unknown',
            'original_points': len(original_df),
            'poi_type_distribution': original_df['type'].value_counts().head(20).to_dict() if 'type' in original_df.columns else {}
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        # 3. 保存示例CSV (便于查看)
        csv_path = output_path.replace('.pkl', '_sample.csv')
        if instances:
            # 取第一个实例的数据
            first_instance = instances[0]
            
            sample_data = []
            for i in range(min(100, len(first_instance['points']))):
                point = first_instance['points'][i].cpu().numpy() if hasattr(first_instance['points'][i], 'cpu') else first_instance['points'][i]
                feature = first_instance['tourism_features'][i].cpu().numpy() if hasattr(first_instance['tourism_features'][i], 'cpu') else first_instance['tourism_features'][i]
                
                sample_data.append({
                    'point_id': i,
                    'x': point[0],
                    'y': point[1],
                    'tourism_score': feature[0],
                    'traffic_score': feature[1],
                    'facility_score': feature[2],
                    'total_weight': first_instance['total_weights'][i].item() if hasattr(first_instance['total_weights'][i], 'item') else first_instance['total_weights'][i],
                    'region_label': first_instance['region_labels'][i].item() if hasattr(first_instance['region_labels'][i], 'item') else first_instance['region_labels'][i]
                })
            
            sample_df = pd.DataFrame(sample_data)
            sample_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"数据集保存完成:")
        logger.info(f"  主文件: {output_path}")
        logger.info(f"  信息文件: {info_path}")
        logger.info(f"  示例文件: {csv_path}")
        
        return dataset_info

def load_mclp_dataset(file_path: str) -> List[Dict]:
    """加载MCLP数据集"""
    logger.info(f"加载MCLP数据集: {file_path}")
    
    with open(file_path, 'rb') as f:
        instances = pickle.load(f)
    
    logger.info(f"加载完成: {len(instances)} 个实例")
    
    # 转换回CPU（如果原来是GPU张量）
    for instance in instances:
        for key, value in instance.items():
            if torch.is_tensor(value):
                instance[key] = value.cpu()
    
    return instances

def analyze_dataset(dataset: List[Dict], max_instances: int = 3):
    """分析数据集"""
    if not dataset:
        logger.warning("数据集为空")
        return
    
    logger.info(f"数据集包含 {len(dataset)} 个实例")
    
    for i, instance in enumerate(dataset[:max_instances]):
        logger.info(f"实例 {i}: {instance['name']}")
        logger.info(f"  节点数: {instance['num_nodes']}")
        logger.info(f"  覆盖半径: {instance['coverage_radius']:.2f}")
        logger.info(f"  平均距离: {instance['avg_distance']:.2f}")
        logger.info(f"  直径: {instance['diameter']:.2f}")
        logger.info(f"  总权重范围: [{instance['total_weights'].min():.3f}, {instance['total_weights'].max():.3f}]")
        
        if i < max_instances - 1:
            logger.info("  " + "-" * 40)

# 命令行接口
def main():
    parser = argparse.ArgumentParser(description='将高德地图CSV转换为MCLP数据集')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default='mclp_real_dataset.pkl',
                       help='输出文件路径（默认: mclp_real_dataset.pkl）')
    parser.add_argument('--num-instances', '-n', type=int, default=50,
                       help='生成的实例数量（默认: 50）')
    parser.add_argument('--max-points', '-m', type=int, default=None,
                       help='最大POI点数（默认: 使用所有点）')
    parser.add_argument('--variation', '-v', type=float, default=0.02,
                       help='实例间变化程度（默认: 0.02）')
    parser.add_argument('--coord-range', '-c', type=float, nargs=2, default=[0.0, 100.0],
                       help='坐标归一化范围（默认: 0.0 100.0）')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备（默认: cpu）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建转换器
    converter = CSVToMCLPConverter(
        coord_range=tuple(args.coord_range),
        device=device
    )
    
    # 执行转换
    try:
        dataset_info = converter.convert(
            csv_path=args.input,
            output_path=args.output,
            num_instances=args.num_instances,
            max_points=args.max_points,
            instance_variation=args.variation
        )
        
        # 加载并分析数据集
        dataset = load_mclp_dataset(args.output)
        analyze_dataset(dataset, max_instances=3)
        
    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 使用示例函数
def example_usage():
    """使用示例"""
    
    # 示例1: 基本使用
    converter = CSVToMCLPConverter()
    
    # 转换CSV文件
    dataset_info = converter.convert(
        csv_path="amap_pois.csv",
        output_path="mclp_real_data.pkl",
        num_instances=100,
        max_points=500
    )
    
    # 加载数据集
    dataset = load_mclp_dataset("mclp_real_data.pkl")
    
    # 分析数据集
    analyze_dataset(dataset)
    
    return dataset

def batch_convert(csv_files: List[str], output_dir: str = "datasets"):
    """批量转换多个CSV文件"""
    converter = CSVToMCLPConverter()
    
    for csv_file in csv_files:
        try:
            # 生成输出文件名
            base_name = os.path.basename(csv_file).replace('.csv', '')
            output_file = os.path.join(output_dir, f"{base_name}_mclp.pkl")
            
            logger.info(f"处理文件: {csv_file}")
            
            # 转换
            converter.convert(
                csv_path=csv_file,
                output_path=output_file,
                num_instances=30,
                max_points=300
            )
            
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 失败: {e}")
            continue