# amap_api.py
import requests
import pandas as pd
import numpy as np
import time
import json
import hashlib
import os
import sys
from typing import List, Dict, Tuple, Optional, Any
import threading
from queue import Queue
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AMapAPI:
    """高德地图API客户端 - 简单实用版"""
    
    def __init__(self, api_key: str = None):
        """
        初始化高德地图API客户端
        
        Args:
            api_key: 高德地图API Key，如果为None则从环境变量或配置文件读取
        """
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError("未提供高德地图API Key！请参考文档获取：https://lbs.amap.com/")
        
        self.base_url = "https://restapi.amap.com/v3"
        
        # API限制（免费版限制）
        self.rate_limit = 10  # 每秒最多10次请求
        self.last_request_time = 0
        self.request_count = 0
        self.request_limit = 2000  # 每日免费额度
        
        # POI类型映射表（简化版）
        self.poi_types = {
            '风景名胜': '风景名胜',
            '公园广场': '公园广场',
            '博物馆': '博物馆',
            '纪念馆': '纪念馆',
            '寺庙道观': '寺庙道观',
            '中餐厅': '中餐厅',
            '外国餐厅': '外国餐厅',
            '快餐厅': '快餐厅',
            '咖啡厅': '咖啡厅',
            '宾馆酒店': '宾馆酒店',
            '购物中心': '购物中心',
            '超市': '超市',
            '电影院': '电影院',
            '体育场馆': '体育场馆',
            '医院': '医院',
            '银行': '银行',
            '停车场': '停车场',
            '公共厕所': '公共厕所',
            '地铁站': '地铁站',
            '公交车站': '公交车站'
        }
    
    def _get_api_key(self):
        """从多个来源获取API Key"""
        # 1. 从环境变量获取
        api_key = os.environ.get('AMAP_API_KEY')
        if api_key:
            return api_key
        
        # 2. 从配置文件获取
        config_file = 'amap_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('api_key')
            except:
                pass
        
        # 3. 从用户输入获取
        print("\n" + "="*60)
        print("高德地图API Key配置")
        print("="*60)
        print("请访问 https://lbs.amap.com/ 注册并创建应用")
        print("在控制台创建Key，选择'Web服务'")
        print("="*60)
        
        api_key = input("请输入您的API Key: ").strip()
        
        if api_key:
            # 保存到配置文件
            try:
                config = {'api_key': api_key}
                with open('amap_config.json', 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                print("✅ API Key已保存到 amap_config.json")
            except:
                print("⚠️  无法保存配置文件，请记住您的API Key")
        
        return api_key
    
    def _wait_for_rate_limit(self):
        """遵守API速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # 免费版限制：每秒最多10次请求
        if time_since_last < 1.0 / self.rate_limit:
            wait_time = (1.0 / self.rate_limit) - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # 检查每日限额
        if self.request_count >= self.request_limit * 0.9:
            logger.warning(f"⚠️  已使用 {self.request_count}/{self.request_limit} 次请求，接近每日限额")
    
    def _make_request(self, endpoint: str, params: Dict, max_retries: int = 3):
        """发送HTTP请求"""
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        params['key'] = self.api_key
        params['output'] = 'JSON'
        
        for retry in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get('status') == '1':
                    logger.debug(f"✅ 请求成功: {endpoint}")
                    return result
                else:
                    error_msg = result.get('info', '未知错误')
                    error_code = result.get('infocode', '')
                    
                    if 'DAILY_QUERY_OVER_LIMIT' in error_msg or '10044' in error_code:
                        logger.error(f"❌ 每日请求超限！错误: {error_msg}")
                        raise Exception("每日请求额度已用完")
                    
                    if 'INVALID_USER_KEY' in error_msg or '10001' in error_code:
                        logger.error(f"❌ API Key无效！错误: {error_msg}")
                        raise Exception("API Key无效")
                    
                    logger.warning(f"⚠️  API错误: {error_msg}, 重试 {retry + 1}/{max_retries}")
                    
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)  # 指数退避
                    else:
                        raise Exception(f"API请求失败: {error_msg}")
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  网络错误: {e}, 重试 {retry + 1}/{max_retries}")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)
                else:
                    raise
        
        raise Exception("请求失败，超过最大重试次数")
    
    def search_poi_by_keyword(self, keyword: str, city: str = None, page: int = 1, page_size: int = 20):
        """
        通过关键词搜索POI
        
        Args:
            keyword: 搜索关键词
            city: 城市名称（可选）
            page: 页码
            page_size: 每页数量（最大50）
        
        Returns:
            API响应结果
        """
        params = {
            'keywords': keyword,
            'city': city if city else '全国',
            'citylimit': 'true' if city else 'false',
            'page': page,
            'offset': min(page_size, 50)  # 高德限制每页最多50条
        }
        
        return self._make_request('place/text', params)
    
    def search_poi_by_type(self, poi_type: str, city: str, page: int = 1, page_size: int = 20):
        """
        通过POI类型搜索
        
        Args:
            poi_type: POI类型
            city: 城市名称
            page: 页码
            page_size: 每页数量
        
        Returns:
            API响应结果
        """
        return self.search_poi_by_keyword(poi_type, city, page, page_size)
    
    def search_around(self, location: Tuple[float, float], radius: int = 3000, poi_type: str = None, page_size: int = 20):
        """
        周边搜索
        
        Args:
            location: (经度, 纬度)
            radius: 搜索半径（米），最大50000
            poi_type: POI类型（可选）
            page_size: 每页数量
        
        Returns:
            API响应结果
        """
        params = {
            'location': f"{location[0]},{location[1]}",
            'radius': min(radius, 50000),
            'offset': min(page_size, 50)
        }
        
        if poi_type:
            params['types'] = poi_type
        
        return self._make_request('place/around', params)
    
    def get_city_suggestions(self, keyword: str):
        """获取城市建议"""
        params = {
            'keywords': keyword,
            'type': 'city'
        }
        return self._make_request('assistant/inputtips', params)
    
    def parse_poi_data(self, api_result: Dict) -> pd.DataFrame:
        """解析API返回的POI数据为DataFrame"""
        pois = []
        
        if 'pois' not in api_result:
            logger.warning("API返回结果中没有'pois'字段")
            return pd.DataFrame()
        
        for poi in api_result['pois']:
            try:
                # 解析坐标
                location = poi.get('location', '')
                lon, lat = 0.0, 0.0
                if location:
                    coords = location.split(',')
                    if len(coords) >= 2:
                        lon, lat = float(coords[0]), float(coords[1])
                
                # 提取商圈信息
                business_area = poi.get('business_area', '')
                if isinstance(business_area, list):
                    business_area = ','.join(business_area)
                
                poi_info = {
                    'id': poi.get('id', ''),
                    'name': poi.get('name', ''),
                    'type': poi.get('type', ''),
                    'type_code': poi.get('typecode', ''),
                    'address': poi.get('address', ''),
                    'location': location,
                    'lon': lon,
                    'lat': lat,
                    'tel': poi.get('tel', ''),
                    'pname': poi.get('pname', ''),  # 省名称
                    'cityname': poi.get('cityname', ''),  # 城市名称
                    'adname': poi.get('adname', ''),  # 区县名称
                    'business_area': business_area,
                    'tag': poi.get('tag', ''),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 尝试提取评分信息
                biz_ext = poi.get('biz_ext', {})
                if isinstance(biz_ext, dict):
                    poi_info['rating'] = float(biz_ext.get('rating', '0'))
                    poi_info['cost'] = float(biz_ext.get('cost', '0'))
                elif isinstance(biz_ext, str):
                    try:
                        biz_dict = json.loads(biz_ext)
                        poi_info['rating'] = float(biz_dict.get('rating', '0'))
                        poi_info['cost'] = float(biz_dict.get('cost', '0'))
                    except:
                        poi_info['rating'] = 0.0
                        poi_info['cost'] = 0.0
                else:
                    poi_info['rating'] = 0.0
                    poi_info['cost'] = 0.0
                
                pois.append(poi_info)
                
            except Exception as e:
                logger.warning(f"解析POI数据时出错: {e}, 跳过该POI")
                continue
        
        if not pois:
            return pd.DataFrame()
        
        df = pd.DataFrame(pois)
        
        # 计算特征分数
        df = self._calculate_features(df)
        
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算POI特征分数"""
        if df.empty:
            return df
        
        # 初始化特征列
        df['tourism_score'] = 0.0
        df['traffic_score'] = 0.0
        df['facility_score'] = 0.0
        
        # 基于类型分配分数
        for idx, row in df.iterrows():
            poi_type = str(row['type']).lower()
            poi_name = str(row['name']).lower()
            
            # 旅游相关特征
            tourism_keywords = [
                '风景', '公园', '广场', '博物', '展览', '纪念',
                '寺庙', '教堂', '道观', '度假', '农家', '乐园',
                '动物园', '植物园', '景区', '景点', '旅游', '观光',
                '古镇', '古城', '老街', '遗址', '故居'
            ]
            
            # 交通相关特征
            traffic_keywords = [
                '车站', '地铁', '火车', '高铁', '汽车', '机场',
                '停车', '加油', '充电', '公交', '出租', '交通',
                '枢纽', '客运', '码头', '港口', '轨道', '线路'
            ]
            
            # 设施相关特征
            facility_keywords = [
                '医院', '诊所', '药店', '卫生', '银行', 'atm',
                '邮政', '电信', '移动', '联通', '公安', '消防',
                '警察', '法院', '政府', '机关', '厕所', '洗手间',
                '商场', '超市', '市场', '百货', '便利店', '店铺',
                '餐厅', '饭店', '酒楼', '餐馆', '食堂', '快餐',
                '酒店', '宾馆', '旅馆', '住宿', '客栈', '招待所'
            ]
            
            # 计算旅游特征
            for keyword in tourism_keywords:
                if keyword in poi_type or keyword in poi_name:
                    df.at[idx, 'tourism_score'] += 0.3
            
            # 计算交通特征
            for keyword in traffic_keywords:
                if keyword in poi_type or keyword in poi_name:
                    df.at[idx, 'traffic_score'] += 0.4
            
            # 计算设施特征
            for keyword in facility_keywords:
                if keyword in poi_type or keyword in poi_name:
                    df.at[idx, 'facility_score'] += 0.3
            
            # 根据评分调整
            if 'rating' in df.columns and row['rating'] > 0:
                df.at[idx, 'tourism_score'] += min(row['rating'] / 5.0 * 0.3, 0.3)
            
            # 根据价格调整（便宜的设施可能更受欢迎）
            if 'cost' in df.columns and row['cost'] > 0:
                if row['cost'] < 100:  # 便宜
                    df.at[idx, 'facility_score'] += 0.2
        
        # 归一化到0-1
        for feature in ['tourism_score', 'traffic_score', 'facility_score']:
            if feature in df.columns and df[feature].max() > 0:
                df[feature] = df[feature] / df[feature].max()
                df[feature] = df[feature].clip(0, 1)  # 确保在0-1范围内
        
        return df
    
    def collect_pois_by_city(self, city: str, max_pois: int = 200, poi_types: List[str] = None):
        """
        收集指定城市的POI数据
        
        Args:
            city: 城市名称（如"北京市", "上海市"）
            max_pois: 最大POI数量
            poi_types: POI类型列表，如果为None则使用默认类型
        
        Returns:
            包含POI数据的DataFrame
        """
        if poi_types is None:
            # 使用简化的POI类型
            poi_types = ['风景名胜', '公园广场', '博物馆', '中餐厅', '宾馆酒店', 
                        '购物中心', '电影院', '医院', '银行', '地铁站']
        
        logger.info(f"开始收集 {city} 的POI数据，目标数量: {max_pois}")
        
        all_pois = []
        collected_count = 0
        
        for poi_type in poi_types:
            if collected_count >= max_pois:
                break
            
            logger.info(f"  收集类型: {poi_type}")
            page = 1
            page_size = 25  # 每次获取25条
            
            while collected_count < max_pois:
                try:
                    # 获取当前类型的数据
                    result = self.search_poi_by_type(poi_type, city, page, page_size)
                    
                    if result.get('status') != '1':
                        logger.warning(f"    获取 {poi_type} 失败: {result.get('info')}")
                        break
                    
                    # 解析数据
                    df_page = self.parse_poi_data(result)
                    
                    if df_page.empty:
                        logger.info(f"    {poi_type} 没有更多数据")
                        break
                    
                    # 添加到总数据
                    all_pois.append(df_page)
                    collected_count += len(df_page)
                    
                    logger.info(f"    第{page}页: 获取到 {len(df_page)} 条，总计 {collected_count}/{max_pois}")
                    
                    # 检查是否还有更多页
                    total_count = int(result.get('count', 0))
                    if page * page_size >= total_count:
                        break
                    
                    # 翻页
                    page += 1
                    
                    # 避免请求过快
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"    获取 {poi_type} 第{page}页时出错: {e}")
                    break
        
        # 合并所有数据
        if all_pois:
            df_all = pd.concat(all_pois, ignore_index=True)
            
            # 去重（基于ID）
            df_all = df_all.drop_duplicates(subset=['id'])
            
            # 限制数量
            if len(df_all) > max_pois:
                df_all = df_all.head(max_pois)
            
            logger.info(f"✅ 收集完成！共获取 {len(df_all)} 个POI")
            
            # 统计信息
            if not df_all.empty:
                logger.info(f"  坐标范围: 经度 [{df_all['lon'].min():.6f}, {df_all['lon'].max():.6f}]")
                logger.info(f"           纬度 [{df_all['lat'].min():.6f}, {df_all['lat'].max():.6f}]")
                
                if 'tourism_score' in df_all.columns:
                    logger.info(f"  特征平均分: 旅游 {df_all['tourism_score'].mean():.3f}, "
                              f"交通 {df_all['traffic_score'].mean():.3f}, "
                              f"设施 {df_all['facility_score'].mean():.3f}")
            
            return df_all
        else:
            logger.warning("❌ 未收集到任何POI数据")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """保存DataFrame到CSV文件"""
        if df.empty:
            logger.warning("没有数据可保存")
            return False
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # 保存文件
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 数据已保存到: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存文件失败: {e}")
            return False
    
    def save_to_json(self, df: pd.DataFrame, filename: str):
        """保存DataFrame到JSON文件"""
        if df.empty:
            logger.warning("没有数据可保存")
            return False
        
        try:
            # 转换为字典列表
            data_dict = df.to_dict(orient='records')
            
            # 保存文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 数据已保存到JSON: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存JSON文件失败: {e}")
            return False
    
    def display_summary(self, df: pd.DataFrame):
        """显示数据摘要"""
        if df.empty:
            print("❌ 数据为空")
            return
        
        print("\n" + "="*60)
        print("POI数据摘要")
        print("="*60)
        print(f"总数量: {len(df)} 个POI")
        print(f"城市分布: {df['cityname'].value_counts().to_dict()}")
        
        print("\nPOI类型分布 (前10):")
        type_counts = df['type'].value_counts().head(10)
        for type_name, count in type_counts.items():
            print(f"  {type_name[:20]:20s}: {count:4d}")
        
        if 'tourism_score' in df.columns:
            print("\n特征统计:")
            print(f"  旅游特征: 平均 {df['tourism_score'].mean():.3f}, "
                  f"最小 {df['tourism_score'].min():.3f}, "
                  f"最大 {df['tourism_score'].max():.3f}")
            print(f"  交通特征: 平均 {df['traffic_score'].mean():.3f}, "
                  f"最小 {df['traffic_score'].min():.3f}, "
                  f"最大 {df['traffic_score'].max():.3f}")
            print(f"  设施特征: 平均 {df['facility_score'].mean():.3f}, "
                  f"最小 {df['facility_score'].min():.3f}, "
                  f"最大 {df['facility_score'].max():.3f}")
        
        print(f"\n坐标范围:")
        print(f"  经度: [{df['lon'].min():.6f}, {df['lon'].max():.6f}]")
        print(f"  纬度: [{df['lat'].min():.6f}, {df['lat'].max():.6f}]")
        print("="*60)

class AMapPOICollector:
    """POI数据采集器 - 简单接口"""
    
    def __init__(self, api_key: str = None):
        self.api = AMapAPI(api_key)
        self.data = pd.DataFrame()
    
    def collect(self, city: str, max_pois: int = 200, poi_types: List[str] = None):
        """收集POI数据"""
        print(f"🔄 开始收集 {city} 的POI数据...")
        
        self.data = self.api.collect_pois_by_city(city, max_pois, poi_types)
        
        if not self.data.empty:
            print(f"✅ 收集完成！共获取 {len(self.data)} 个POI")
            self.api.display_summary(self.data)
        
        return self.data
    
    def save(self, filename: str = None, format: str = 'csv'):
        """保存数据"""
        if self.data.empty:
            print("❌ 没有数据可保存")
            return False
        
        if filename is None:
            # 生成默认文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            city = self.data['cityname'].iloc[0] if 'cityname' in self.data.columns and not self.data.empty else 'unknown'
            filename = f"poi_data_{city}_{timestamp}.{format}"
        
        if format.lower() == 'csv':
            return self.api.save_to_csv(self.data, filename)
        elif format.lower() == 'json':
            return self.api.save_to_json(self.data, filename)
        else:
            print(f"❌ 不支持的格式: {format}")
            return False
    
    def get_training_data(self):
        """获取训练数据格式（坐标+特征）"""
        if self.data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 提取坐标
        coords_df = self.data[['lon', 'lat']].copy()
        coords_df.columns = ['x', 'y']
        
        # 提取特征
        features_df = pd.DataFrame()
        if 'tourism_score' in self.data.columns:
            features_df['tourism_score'] = self.data['tourism_score']
        if 'traffic_score' in self.data.columns:
            features_df['traffic_score'] = self.data['traffic_score']
        if 'facility_score' in self.data.columns:
            features_df['facility_score'] = self.data['facility_score']
        
        # 如果特征列缺失，创建默认值
        if features_df.empty:
            features_df = pd.DataFrame({
                'tourism_score': np.random.random(len(coords_df)) * 0.8,
                'traffic_score': np.random.random(len(coords_df)) * 0.6,
                'facility_score': np.random.random(len(coords_df)) * 0.7
            })
        
        return coords_df, features_df

def main():
    """主程序 - 可以直接运行"""
    print("\n" + "="*60)
    print("高德地图POI数据采集工具")
    print("="*60)
    
    print("\n请选择操作:")
    print("1. 采集单个城市POI数据")
    print("2. 查看API Key状态")
    print("3. 退出")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == '1':
        # 获取城市名称
        city = input("请输入城市名称 (如: 北京市, 上海市, 杭州市): ").strip()
        if not city:
            city = "北京市"  # 默认
        
        # 获取POI数量
        try:
            max_pois = int(input("请输入最大POI数量 (默认200): ").strip() or "200")
            max_pois = min(max_pois, 1000)  # 限制最大数量
        except:
            max_pois = 200
        
        print(f"\n开始采集 {city} 的POI数据，目标 {max_pois} 个...")
        
        # 创建采集器
        collector = AMapPOICollector()
        
        try:
            # 采集数据
            df = collector.collect(city, max_pois)
            
            if not df.empty:
                # 询问是否保存
                save_choice = input("\n是否保存数据？(y/n): ").strip().lower()
                if save_choice == 'y':
                    format_choice = input("保存格式 (csv/json, 默认csv): ").strip().lower() or 'csv'
                    
                    if format_choice not in ['csv', 'json']:
                        format_choice = 'csv'
                    
                    filename = input(f"文件名 (默认自动生成): ").strip()
                    
                    if collector.save(filename, format_choice):
                        print("✅ 保存成功！")
                    else:
                        print("❌ 保存失败")
                
                # 显示前几条数据
                show_data = input("\n是否显示前5条数据？(y/n): ").strip().lower()
                if show_data == 'y':
                    print("\n前5条POI数据:")
                    print(df.head().to_string())
        
        except Exception as e:
            print(f"❌ 采集过程中出错: {e}")
            print("可能的原因:")
            print("  1. API Key无效或过期")
            print("  2. 网络连接问题")
            print("  3. 每日请求额度已用完")
            print("  4. 城市名称不正确")
    
    elif choice == '2':
        # 测试API Key
        try:
            api = AMapAPI()
            print(f"\n✅ API Key状态正常")
            print(f"   当前Key: {api.api_key[:8]}...{api.api_key[-4:]}")
            print(f"   已使用请求: {api.request_count}")
            print(f"   每日限额: {api.request_limit}")
        except Exception as e:
            print(f"\n❌ API Key状态异常: {e}")
    
    elif choice == '3':
        print("退出程序")
        return
    
    else:
        print("❌ 无效选择")
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    # 直接运行示例
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        print("请检查网络连接和API Key配置")

# ============================================
# 使用示例代码（可以直接复制到其他文件中使用）
# ============================================

def example_usage():
    """使用示例"""
    
    # 示例1: 基本用法
    print("示例1: 基本用法")
    collector = AMapPOICollector()
    df = collector.collect("杭州市", max_pois=100)
    collector.save("hangzhou_pois.csv")
    
    # 示例2: 获取训练数据
    print("\n示例2: 获取训练数据")
    coords_df, features_df = collector.get_training_data()
    print(f"坐标数据形状: {coords_df.shape}")
    print(f"特征数据形状: {features_df.shape}")
    
    # 示例3: 直接使用API
    print("\n示例3: 直接使用API")
    api = AMapAPI()
    
    # 搜索特定类型的POI
    result = api.search_poi_by_type("公园广场", "北京市", page=1, page_size=10)
    df = api.parse_poi_data(result)
    print(f"获取到 {len(df)} 个公园广场")
    
    # 周边搜索
    # result = api.search_around((116.397428, 39.90923), radius=1000, poi_type="餐厅")
    # df = api.parse_poi_data(result)
    
    return df

def create_test_data():
    """创建测试数据（当API不可用时使用）"""
    print("创建测试数据...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_points = 200
    
    # 模拟杭州市范围
    lons = np.random.uniform(120.0, 120.5, n_points)
    lats = np.random.uniform(30.0, 30.5, n_points)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'id': [f'test_{i}' for i in range(n_points)],
        'name': [f'测试POI_{i}' for i in range(n_points)],
        'type': np.random.choice(['公园', '餐厅', '酒店', '商场', '医院'], n_points),
        'lon': lons,
        'lat': lats,
        'cityname': '杭州市',
        'tourism_score': np.random.random(n_points) * 0.8,
        'traffic_score': np.random.random(n_points) * 0.6,
        'facility_score': np.random.random(n_points) * 0.7
    })
    
    # 保存测试数据
    df.to_csv('test_pois.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 测试数据已保存到 test_pois.csv")
    
    return df

# 如果直接运行此文件，执行主程序
if __name__ == "__main__":
    main()