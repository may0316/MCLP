import osmnx as ox
import torch
import numpy as np

def load_osm_poi_data(
    max_points=200,
    device=None
):
    import torch
    import numpy as np

    print("【POI】使用离线真实尺度模拟 POI（稳定版）")

    # 模拟杭州西湖附近 2km × 2km 区域（真实量级）
    # 单位：米
    np.random.seed(42)

    N = max_points
    x = np.random.uniform(0, 2000, size=N)
    y = np.random.uniform(0, 2000, size=N)

    points = torch.tensor(
        np.stack([x, y], axis=1),
        dtype=torch.float,
        device=device
    )

    # tourism / amenity / service 三类特征（和你模型结构匹配）
    tourism_features = torch.ones(N, 3, device=device)

    print(f"【POI】生成 POI 数量: {N}")

    return points, tourism_features