# 配置 DensityGridSampler的参数
sampler = dict(
    type='DensityGridSampler', # 采样器类型
    update_den_freq=16,        # 更新密度网格的频率
)

# 配置编码器参数
encoder = dict(
    pos_encoder = dict(
        type='HashEncoder', # 位置编码器类型为HashEncoder
    ),
    dir_encoder = dict(
        type='SHEncoder',   # 方向编码器为SHEncoder
    ),
)

# 配置模型参数
model = dict(
    type='NGPNetworks', # 模型类型为NGPNetworks
    use_fully=True,     # 是否使用全连接层
)

# 配置损失函数参数
loss = dict(
    type='HuberLoss',   # 损失函数类型为 HuberLoss
    delta=0.1,          # 阈值delta
)

# 配置优化器参数
optim = dict(
    type='Adam',
    lr=1e-1,
    eps=1e-15,
    betas=(0.9,0.99),
)

# 配置 EMA(Exponential Moving Average) 参数
ema = dict(
    type='EMA',         # EMA 类型
    decay=0.95,         # EMA 的衰减率
)

# 配置指数衰减参数
expdecay=dict(
    type='ExpDecay',        # 指数衰减类型
    decay_start=20_000,     # 开始衰减的部署
    decay_interval=10_000,  # 衰减的间隔步数
    decay_base=0.33,        # 衰减的基数
    decay_end=None          # 衰减结束步数（None 表示无限制）
)
dataset_type = 'NerfDataset'
dataset_dir = 'data/fox'
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='train',
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode='test',
        preload_shuffle=False,
    ),
)

exp_name = "fox"
log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1<<18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt=False
# Use fp16 for faster training
fp16 = True