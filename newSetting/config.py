import torch as t

class Config:
    profile="pro"
    
    # 生产环境
    pro_params={
        # 个体参数设置
        "max_unit_length":5,
        "min_unit_length":3,
        "image_channel":3,
        "out_channel":64,
        "in_size":224,
        "max_total_SENet_amount":30,# 一个个体中SENet的最大数量（SENet涉及参数量，参数量不宜过多）
        "min_total_SENet_amount":10,# 一个个体中SENet的最小数量
        "max_ResNeXt_amount":4,# 一个unit中包含ResNeXt block的最大数量
        "min_ResNeXt_amount":3,# 一个unit中包含ResNeXt block的最小数量
        "cardinality": [2]*5+[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # cardinality列表
        "group_width":[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # group_width列表
        "init_c_and_gw_index":-5, # 生成个体时选择cardinality和group_width的范围[:init_c_and_gw_index]
        "cardinality_with_group_width_up_limit":64*16, # cardinality*group_width的上限
        "cardinality_with_group_width_down_limit":8*8, # cardinality*group_width的下限
        
        # 训练设置
        "lr":0.1,
        "device":t.device("cuda:0"),
        "evolve_epochs":160, # 进化过程中的epoch
        "ntk_batch_size":32, # 评估ntk的batch_size
        "acc_batch_size":16, # 评估acc的batch_size
        "retrain_epochs":300, # 最优个体的重训练epoch
        "retrain_batch_size":128, # 最优个体的重训练batch_size
        "dataset":2, # 1表示CIFAR10,2表示CIFAR100
        "train_top_k":10, # 进化结束后训练ntk前k个的个体，评估个体的acc

        # 种群设置
        # "pop_size":40, # 种群中的个体数
        # "gen":50, # 进化轮次
        # "s":5, # 随机选取s个个体
        # "ntk_t":3, # 基于ntk选择个体的时候，从s个个体中选取的最优个体个数进行变异
        # "spantime_t":1, # 基于寿命选择个体的时候，从s个个体中选取的最优个体个数进行变异
        # "gen_division1":15, #  基于首次适应度评估和寿命评估的分割代次
        # "gen_division2":45, #  基于寿命评估和第二次适应度评估的分割代次

        "pop_size":4, # 种群中的个体数
        "gen":5, # 进化轮次
        "s":2, # 随机选取s个个体
        "ntk_t":1, # 基于ntk选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "spantime_t":1, # 基于寿命选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "gen_division1":2, #  基于首次适应度评估和寿命评估的分割代次
        "gen_division2":3, #  基于寿命评估和第二次适应度评估的分割代次

        # 其他设置
        "log_path":"../Output/v2.15"
    }

    # 开发环境
    dev_params={
        # 个体参数设置
        "max_unit_length":5,
        "min_unit_length":3,
        "image_channel":3,
        "out_channel":64,
        "in_size":32,
        "max_total_SENet_amount":30,# 一个个体中SENet的最大数量（SENet涉及参数量，参数量不宜过多）
        "min_total_SENet_amount":10,# 一个个体中SENet的最小数量
        "max_ResNeXt_amount":4,# 一个unit中包含ResNeXt block的最大数量
        "min_ResNeXt_amount":3,# 一个unit中包含ResNeXt block的最小数量
        "cardinality": [2]*5+[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # cardinality列表
        "group_width":[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # group_width列表
        "init_c_and_gw_index":-5, # 生成个体时选择cardinality和group_width的范围[:init_c_and_gw_index]
        "cardinality_with_group_width_up_limit":64*16, # cardinality*group_width的上限
        "cardinality_with_group_width_down_limit":8*8, # cardinality*group_width的下限
        
        # 训练设置
        "lr":0.1,
        "gpu":"0,1", #"None"表示单GPU，表示多GPU
        "device":t.device("cuda:0"),
        "evolve_epochs":350, # 进化过程中的epoch
        "ntk_batch_size":32, # 评估ntk的batch_size
        "acc_batch_size":256, # 评估acc的batch_size
        "retrain_epochs":300, # 最优个体的重训练epoch
        "retrain_baNVIDIA-SMtch_size":128, # 最优个体的重训练batch_size
        "dataset":1, # 1表示CIFAR10,2表示CIFAR100
        "train_top_k":10, # 进化结束后训练ntk前k个的个体，评估个体的acc

        # 种群设置
        "pop_size":40, # 种群中的个体数
        "gen":50, # 进化轮次
        "s":5, # 随机选取s个个体
        "ntk_t":3, # 基于ntk选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "spantime_t":1, # 基于寿命选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "gen_division1":15, #  基于首次适应度评估和寿命评估的分割代次
        "gen_division2":30, #  基于寿命评估和第二次适应度评估的分割代次

        # 其他设置
        "log_path":"../Output/v5.6_1"
    }
    # 测试环境
    test_params={
        # 个体参数设置
        "max_unit_length":5,
        "min_unit_length":3,
        "image_channel":3,
        "out_channel":64,
        "in_size":32,
        "max_total_SENet_amount":30,# 一个个体中SENet的最大数量（SENet涉及参数量，参数量不宜过多）
        "min_total_SENet_amount":10,# 一个个体中SENet的最小数量
        "max_ResNeXt_amount":4,# 一个unit中包含ResNeXt block的最大数量
        "min_ResNeXt_amount":3,# 一个unit中包含ResNeXt block的最小数量
        "cardinality": [2]*5+[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # cardinality列表
        "group_width":[4]*5+[8]*5+[16]*5+[32]*3+[64]*3, # group_width列表
        "init_c_and_gw_index":-5, # 生成个体时选择cardinality和group_width的范围[:init_c_and_gw_index]
        "cardinality_with_group_width_up_limit":64*16, # cardinality*group_width的上限
        "cardinality_with_group_width_down_limit":8*8, # cardinality*group_width的下限
        
        # 训练设置
        "lr":0.1,
        "device":t.device("cuda:1"),
        "evolve_epochs":300, # 进化过程中的epoch
        "ntk_batch_size":64, # 评估ntk的batch_size
        "acc_batch_size":128, # 评估acc的batch_size
        "retrain_epochs":300, # 最优个体的重训练epoch
        "retrain_batch_size":128, # 最优个体的重训练batch_size
        "dataset":2, # 1表示CIFAR10,2表示CIFAR100
        "train_top_k":10, # 进化结束后训练ntk前k个的个体，评估个体的acc

        # 种群设置
        "pop_size":40, # 种群中的个体数
        "gen":50, # 进化轮次
        "s":5, # 随机选取s个个体
        "ntk_t":3, # 基于ntk选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "spantime_t":1, # 基于寿命选择个体的时候，从s个个体中选取的最优个体个数进行变异
        "gen_division1":15, #  基于首次适应度评估和寿命评估的分割代次
        "gen_division2":30, #  基于寿命评估和第二次适应度评估的分割代次

        # 其他设置
        "log_path":"../Output/v5.20_1"
    }

    
"""全局配置

说明
- 通过 `profile` 选择不同环境参数集（生产/开发/测试）
- 参数包含：结构搜索范围、训练超参、种群规模与分期策略、日志路径等
"""
