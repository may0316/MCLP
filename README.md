# 生成 n=20, p=4, radius=0.3 的数据
python sponet_gen_data.py --graph_size 20 --p 4 --radius 0.3 --dataset_size 10000
python sponet_gen_data.py --graph_size 50 --p 8 --radius 0.2 --dataset_size 10000
python sponet_gen_data.py --graph_size 100 --p 15 --radius 0.15 --dataset_size 10000

# 预训练对应规模的模型
python pre_FLP.py 20 4 0.3
python pre_FLP.py 50 8 0.2
python pre_FLP.py 100 15 0.15

# 求解对应规模的问题
python FLP.py 20 4 0.3
python FLP.py 50 8 0.2
python FLP.py 100 15 0.15