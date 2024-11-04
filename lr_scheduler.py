import math
import functools
import torch

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:          #预热期
        multiplier = iteration / warmup_iterations      #学习率从0线性增加到1
    else:           #余弦退火期
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))        #使用余弦函数将学习率从1逐渐降到0
    return multiplier
'''
学习率调度策略的优点:

预热期帮助模型在训练初期稳定,避免大学习率导致的不稳定
余弦退火使学习率平滑下降,有助于模型在训练后期更好地收敛
'''
def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup, 
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler