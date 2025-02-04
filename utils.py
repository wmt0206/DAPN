import torch
import numpy as np
import torch.optim as optim

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
def adjust_learning_rate(optimizer, epoch, lr=0.01, step1=30, step2=60, step3=90):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= step3:
        lr = lr * 0.001
    elif epoch >= step2:
        lr = lr * 0.01
    elif epoch >= step1:
        lr = lr * 0.1
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    # 用来重置AverageMeter的所有状态，将值(val)、平均值(avg)、总和(sum)和计数(count)都设为0。
    # 通常在一个新的评估周期开始前（如每个epoch开始）调用，以清空之前累计的数据。
    def reset(self):
        self.val = 0
        self.avg = 0      
        self.sum = 0
        self.count = 0

    # 用于更新当前的值以及基于这个新值和一个计数(n)来重新计算总和与平均值。

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 